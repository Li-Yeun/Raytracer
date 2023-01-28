uint RandomUInt(__global uint* seeds, int id)
{
	seeds[id] ^= seeds[id] << 13;
	seeds[id] ^= seeds[id] >> 17;
	seeds[id] ^= seeds[id] << 5;
	return seeds[id];
}

//float RandomFloatSeed(uint seed) { return RandomUIntSeed(seed) * 2.3283064365387e-10f; }
float RandomFloat(__global uint* seeds, int id) { return RandomUInt(seeds, id) * 2.3283064365387e-10f; }
float Rand(__global uint* seeds, int id, float range) { return RandomFloat(seeds, id) * range; }

float magnitude(float4 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}


// TODO CHECK IF THIS FUNCTION EXACTLY MATCHES WITH THE CPU FUNCTION
float4 GetAlbedo(float4 I, float4 N, __global uint* texture)
{
    if (N.y == 1)
    {
        // floor albedo: checkerboard
        int ix = (int)(I.x * 2 + 96.01f);
        int iz = (int)(I.z * 2 + 96.01f);
        // add deliberate aliasing to two tile
        if (ix == 98 && iz == 98) ix = (int)(I.x * 32.01f), iz = (int)(I.z * 32.01f);
        if (ix == 94 && iz == 98) ix = (int)(I.x * 64.01f), iz = (int)(I.z * 64.01f);
        float r = ((ix + iz) & 1) ? 1 : 0.3f;
        return (float4) (r, r, r, r);
    }
    else if (N.z == -1)
    {
        // back wall: logo
        int ix = (int)((I.x + 4) * (128.0f / 8));
        int iy = (int)((2 - I.y) * (64.0f / 3));
        uint p = texture[(ix & 127) + (iy & 63) * 128];
        float4 i3 = (float4) ((float)((p >> 16) & 255), (float)((p >> 8) & 255), (float)(p & 255), 0.0f);
        return i3 * (1.0f / 255.0f);
    }
    return (float4)(0.93f, 0.93f, 0.93f, 0.0f);
}

__kernel void Shade(__global int* rayCounter, __global int* pixelIdxs, __global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs, // Primary Rays
__global __read_only float4* albedos, __global __read_only int* materials, __global float4* __read_only primNorms, __global __read_only float* sphereInvrs, float4 primStartIdx, float4 primCount, __global __read_only uint* texture,       // Primitives
__global float4* lightCorners, float A, float s, float4 emission,                                                                                                                // Light Source(s)
__global float4* energies, __global float4* transmissions,                                                                                                                       // E & T
__global int* shadowCounter, __global int* shadowPixelIdxs, __global float4* shadowOrigins, __global float4* shadowDirections, __global float* shadowDistances,                  // Shadow Rays
__global int* bounceCounter, __global int* bouncePixelIdxs,                                                                                                                      // Bounce Rays
__global uint* seeds)  // Maybe make seed a pointer and atomically increment it after creating a seed                                                                                        // Random CPU seed
{   
    int threadId = get_global_id(0);

    if(threadId >= *rayCounter)
    {
        return;
    }

    int rayPixelIdx = pixelIdxs[threadId];

    if (primIdxs[rayPixelIdx] == -1)
        return;

     // TODO CHECK IF MATERIAL IS LIGHT (PROBABLY DO THIS IN EXTEND KERNEL ALREADY)
    if(materials[primIdxs[rayPixelIdx]] == 4) // IF MAT IS A LIGHT SOURCE
        return;
    
    seeds[rayPixelIdx] += threadId;
    
    float4 I = origins[rayPixelIdx] + directions[rayPixelIdx] * distances[rayPixelIdx];
    float4 N;

    if(primIdxs[rayPixelIdx] >= (int)primStartIdx.x && primIdxs[rayPixelIdx] < (int)primStartIdx.x + (int)primCount.x) // If primitive = sphere
        N = (I - primNorms[primIdxs[rayPixelIdx]]) * sphereInvrs[primIdxs[rayPixelIdx] - (int)primStartIdx.x];
    else
        N = primNorms[primIdxs[rayPixelIdx]];
        
    if (dot( N, directions[rayPixelIdx] ) > 0)
        N = -N; // hit backside / inside

    float4 albedo;
    if (primIdxs[rayPixelIdx] >= (int)primStartIdx.y && primIdxs[rayPixelIdx] < (int)primStartIdx.y + (int)primCount.y) // If primitive = plane
    {
        albedo = GetAlbedo(I, N, texture);
    }
    else
        albedo = albedos[primIdxs[rayPixelIdx]];

    float4 BRDF = albedo / M_PI_F;

        // Pick random position
    float4 c1c2 = normalize(lightCorners[0] - lightCorners[1]);
    float randomLength = RandomFloat(seeds, rayPixelIdx) * s;
    float4 u = c1c2 * randomLength;

    float4 c2c3 = normalize(lightCorners[2] - lightCorners[1]);
    randomLength = RandomFloat(seeds, rayPixelIdx) * s;
    float4 v = c2c3 * randomLength;

    float4 light_point = lightCorners[1] + u + v - (float4)(0.0f, 0.01f, 0.0f, 0.0f);
    
    float4 L = light_point - I;
    float dist = magnitude(L); // CHECK IF OPENCL DISTANCE DOES THE SAME
	L /= dist;

    if (dot(N, L) > 0 && dot(primNorms[0], -L) > 0) {

        int si = atomic_inc(shadowCounter);

        shadowOrigins[si] = I + L * 0.001f;
        shadowDirections[si] = L;
        shadowDistances[si] = dist - 2.0f * 0.001f;

        shadowPixelIdxs[si] = rayPixelIdx; 

        float solidAngle = (dot(primNorms[0], -L) * A) / (dist * dist);
        float lightPDF = 1.0f / solidAngle;

        // DOUBLE CHECK
        energies[si] = transmissions[rayPixelIdx] * (dot(N, L) / lightPDF) * BRDF * emission; 
    }
    
    // Russian Roulette
    
    float p = clamp(max(albedo.z, max(albedo.x, albedo.y)), 0.0f, 1.0f);
    //printf("float: %f", p);
    if (p >= RandomFloat(seeds, rayPixelIdx)) {
        // continue random walk
        int ei = atomic_inc(bounceCounter);
        bouncePixelIdxs[ei] = rayPixelIdx;

        transmissions[rayPixelIdx] *= 1.0f / p;

        // Calculate a diffuse reflection
        float4 R = (float4) (Rand(seeds, rayPixelIdx, 2.0f) - 1.0f, Rand(seeds, rayPixelIdx, 2.0f) - 1.0f, Rand(seeds, rayPixelIdx, 2.0f) - 1.0f, 0);

        while (magnitude(R) > 1.0f)
        {
            R = (float4)(Rand(seeds, rayPixelIdx, 2.0f) - 1.0f, Rand(seeds, rayPixelIdx, 2.0f) - 1.0f, Rand(seeds, rayPixelIdx, 2.0f) - 1.0f, 0);
        }

        R = normalize(R);

        if (dot(N, R) < 0) 
            R = -R;

        float hemiPDF = 1.0f / (M_PI_F * 2.0f);
        origins[rayPixelIdx] = I + R * 0.001f;
        directions[rayPixelIdx] = R;

        transmissions[rayPixelIdx] *= (dot(N, R) / hemiPDF) * BRDF;
    }

}