uint RandomUInt(uint seed)
{
    uint localSeed = seed;
	localSeed ^= localSeed << 13;
	localSeed ^= localSeed >> 17;
	localSeed ^= localSeed << 5;
	return localSeed;
}


float RandomFloat(uint seed) { return seed * 2.3283064365387e-10f; }
float Rand(uint seed, float range) { return RandomFloat(seed) * range; }

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

__kernel void Shade(__global int* rayCounter, __global int* pixelIdxs, __global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs, __global int* lastSpecular, __global int* insides, // Primary Rays
__global float4* albedos, __global int* materials, __global float4* primNorms, __global float* sphereInvrs, float4 primStartIdx, float4 primCount, __global uint* texture, __global float* refractiveIndices, __global float4* absorptions,       // Primitives
__global float4* lightCorners, float A, float s, float4 emission,                                                                                                                // Light Source(s)
__global float4* energies, __global float4* transmissions,                                                                                                                       // E & T
__global int* shadowBounceCounterBuffer, 
__global int* shadowPixelIdxs, __global float4* shadowOrigins, __global float4* shadowDirections, __global float* shadowDistances,                  // Shadow Rays
__global int* bouncePixelIdxs,   
__global float4* accumulator,                                                                                                                   // Bounce Rays
__global uint* seeds)  // Maybe make seed a pointer and atomically increment it after creating a seed                                                                                        // Random CPU seed
{   
    int threadId = get_global_id(0);
    
    int rayPixelIdx = pixelIdxs[threadId];

    if (primIdxs[rayPixelIdx] == -1)
        return;

     // TODO CHECK IF MATERIAL IS LIGHT (PROBABLY DO THIS IN EXTEND KERNEL ALREADY)
    if(materials[primIdxs[rayPixelIdx]] == 4) // IF MAT IS A LIGHT SOURCE
    {
        if(lastSpecular[rayPixelIdx] == 1)
        {
            accumulator[rayPixelIdx] += energies[rayPixelIdx] + transmissions[rayPixelIdx] * emission;
        }

        return;
    }
    
    uint localSeed = seeds[rayPixelIdx] + threadId;

    lastSpecular[rayPixelIdx] = 0;

    
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

    __global int* shadowCounter = &shadowBounceCounterBuffer[0];
    __global int* bounceCounter = &shadowBounceCounterBuffer[1];

    if (materials[primIdxs[rayPixelIdx]] == 1)
    {   
        float p = 0.93f;
        localSeed = RandomUInt(localSeed);
        seeds[rayPixelIdx] = localSeed;

        if (p < RandomFloat(localSeed))
            return;
            
        transmissions[rayPixelIdx] *= 1.0f / p;

        int ei = atomic_inc(bounceCounter);
        bouncePixelIdxs[ei] = rayPixelIdx;

        float3 R = directions[rayPixelIdx].xyz - 2.0f * (dot(directions[rayPixelIdx].xyz, N.xyz)) * N.xyz;
        origins[rayPixelIdx] = I + (float4)(R, 0.0f) * 0.001f;
        directions[rayPixelIdx] = (float4)(R, 0.0f);
        lastSpecular[rayPixelIdx] = 1;
        return;
    }
    else if (materials[primIdxs[rayPixelIdx]] == 2)
    {
        float p = 0.93f;
        localSeed = RandomUInt(localSeed);
        seeds[rayPixelIdx] = localSeed;

        if (p < RandomFloat(localSeed))
            return;

        transmissions[rayPixelIdx] *= 1.0f / p;

        // Compute Refraction & Absoption
        float air_refractive_index = 1.0003f;
        float n1, n2, refraction_ratio;

        float4 absorption = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
        if (insides[rayPixelIdx] == 1)
        {
            absorption = (float4)(exp(-absorptions[primIdxs[rayPixelIdx]].x * distances[rayPixelIdx]), exp(-absorptions[primIdxs[rayPixelIdx]].y * distances[rayPixelIdx]), exp(-absorptions[primIdxs[rayPixelIdx]].z * distances[rayPixelIdx]), 0.0f);
            n1 = refractiveIndices[primIdxs[rayPixelIdx]];
            n2 = air_refractive_index;
        }
        else
        {
            n1 = air_refractive_index;
            n2 = refractiveIndices[primIdxs[rayPixelIdx]];
        }

        refraction_ratio = n1 / n2;

        float incoming_angle = dot(N.xyz, -directions[rayPixelIdx].xyz);
        float k = 1.0f - (refraction_ratio * refraction_ratio) * (1.0f - (incoming_angle * incoming_angle));
        // Compute Freshnel 
        float3 refraction_direction = refraction_ratio * directions[rayPixelIdx].xyz + N.xyz * (refraction_ratio * incoming_angle - sqrt(k));

        float outcoming_angle = dot(-N.xyz, refraction_direction);

        double leftFracture_half = (n1 * incoming_angle - refractiveIndices[primIdxs[rayPixelIdx]] * outcoming_angle) / (n1 * incoming_angle + n2 * outcoming_angle);

        double rightFracture_half = (n1 * outcoming_angle - refractiveIndices[primIdxs[rayPixelIdx]] * incoming_angle) / (n1 * outcoming_angle + n2 * incoming_angle);

        float Fr = 0.5f * (leftFracture_half * leftFracture_half + rightFracture_half * rightFracture_half);

        lastSpecular[rayPixelIdx] = 1;

        int ei = atomic_inc(bounceCounter);
        bouncePixelIdxs[ei] = rayPixelIdx;

        transmissions[rayPixelIdx] *= albedo * absorption;

        float3 R = refraction_direction;

        localSeed = RandomUInt(localSeed);
        seeds[rayPixelIdx] = localSeed;

        if (k < 0 || RandomFloat(localSeed) <= Fr)
        {
            // Compute reflection
            R = directions[rayPixelIdx].xyz - 2.0f * (dot(directions[rayPixelIdx].xyz, N.xyz)) * N.xyz;   
        }
        else
        {
            // refraction
            insides[rayPixelIdx] = -insides[rayPixelIdx];
        }
        origins[rayPixelIdx] = I + (float4)(R, 0.0f) * 0.001f;
        directions[rayPixelIdx] = (float4)(R, 0.0f);

        return;
    }
    float4 BRDF = albedo / M_PI_F;

        // Pick random position
    float4 c1c2 = normalize(lightCorners[0] - lightCorners[1]);
    localSeed = RandomUInt(localSeed);

    float randomLength = RandomFloat(localSeed) * s;
    float4 u = c1c2 * randomLength;

    float4 c2c3 = normalize(lightCorners[2] - lightCorners[1]);
    localSeed = RandomUInt(localSeed);
    randomLength = RandomFloat(localSeed) * s;
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
    localSeed = RandomUInt(localSeed);
    
    if (p >= RandomFloat(localSeed)) {
        // continue random walk
        int ei = atomic_inc(bounceCounter);
        bouncePixelIdxs[ei] = rayPixelIdx;

        transmissions[rayPixelIdx] *= 1.0f / p;
            
        localSeed = RandomUInt(localSeed);
        float x = Rand(localSeed, 2.0f);
        localSeed = RandomUInt(localSeed);
        float y = Rand(localSeed, 2.0f);
        localSeed = RandomUInt(localSeed);
        float z = Rand(localSeed, 2.0f);
        // Calculate a diffuse reflection
        float4 R = (float4) (x - 1.0f, y - 1.0f, z - 1.0f, 0);

        while (magnitude(R) > 1.0f)
        {
            localSeed = RandomUInt(localSeed);
            float x = Rand(localSeed, 2.0f);
            localSeed = RandomUInt(localSeed);
            float y = Rand(localSeed, 2.0f);
            localSeed = RandomUInt(localSeed);
            float z = Rand(localSeed, 2.0f);

            R = (float4)(x - 1.0f, y - 1.0f, z - 1.0f, 0);
        }

        R = normalize(R);

        if (dot(N, R) < 0) 
            R = -R;

        float hemiPDF = 1.0f / (M_PI_F * 2.0f);
        origins[rayPixelIdx] = I + R * 0.001f;
        directions[rayPixelIdx] = R;

        transmissions[rayPixelIdx] *= (dot(N, R) / hemiPDF) * BRDF;
    }
    
    seeds[rayPixelIdx] = localSeed;

}