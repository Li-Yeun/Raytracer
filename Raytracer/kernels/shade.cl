int localSeed;

uint RandomUIntSeed(uint seed)
{
    localSeed = seed;
	localSeed ^= seed << 13;
	localSeed ^= localSeed >> 17;
	localSeed ^= localSeed << 5;
	return localSeed;
}

uint RandomUInt()
{
	localSeed ^= localSeed << 13;
	localSeed ^= localSeed >> 17;
	localSeed ^= localSeed << 5;
	return localSeed;
}

float RandomFloatSeed(uint seed) { return RandomUIntSeed(seed) * 2.3283064365387e-10f; }
float RandomFloat() { return RandomUInt() * 2.3283064365387e-10f; }
float Rand(float range) { return RandomFloat() * range; }

float magnitude(float3 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__kernel void Shade(__global int* rayCounter, __global int* pixelIdxs, __global float3* origins, __global float3* directions, __global float* distances, __global int* primIdxs, // Primary Rays
__global float3* albedos, __global float3* primNorms, __global float* sphereInvrs, int sphereStartIdx, int sphereCount,                                                          // Primitives
__global float3* lightCorners, float A, float s, float3 emission,                                                                                                                // Light Source(s)
__global float3* energies, __global float3* transmissions,                                                                                                                       // E & T
__global int* shadowCounter, __global int* shadowPixelIdxs, __global float3* shadowOrigins, __global float3* shadowDirections, __global float* shadowDistances,                  // Shadow Rays
__global int* bounceCounter, __global int* bouncePixelIdxs, __global float3* bounceOrigins, __global float3* bounceDirections,                                                   // Bounce Rays
int seed)  // Maybe make seed a pointer and atomically increment it after creating a seed                                                                                        // Random CPU seed
{   
    int threadId = get_global_id(0);

    if(threadId >= *rayCounter) // TODO CHECK IF THIS IS NEED WHEN USING GPU if(threadId >= *rayCounter - 1)
    {
        return;
    }

     // TODO CHECK IF MATERIAL IS LIGHT (PROBABLY DO THIS IN EXTEND KERNEL ALREADY)

    
    float3 I = origins[threadId] + directions[threadId] * distances[threadId];
    float3 N;

    if(primIdxs[threadId] >= sphereStartIdx && primIdxs[threadId] < sphereStartIdx + sphereCount)
        N = (I - primNorms[primIdxs[threadId]]) * sphereInvrs[primIdxs[threadId] - sphereStartIdx];
    else
        N = primNorms[primIdxs[threadId]];
    
    if (dot( N, directions[threadId] ) > 0)
        N = -N; // hit backside / inside

    float3 BRDF = albedos[primIdxs[threadId]] / M_PI_F;

        // Pick random position
    float3 c1c2 = normalize(lightCorners[0] - lightCorners[1]);
    float randomLength = RandomFloatSeed(seed + threadId * get_local_id(0)) * s;  // Better alternative = RandomFloatSeed(seed + threadId * get_local_id(0) + number0) * s;
    float3 u = c1c2 * randomLength;

    float3 c2c3 = normalize(lightCorners[2] - lightCorners[1]);
    randomLength = RandomFloat() * s;
    float3 v = c2c3 * randomLength;

    float3 light_point = lightCorners[1] + u + v - (float3)(0, 0.01f, 0);
    
    float3 L = light_point - I;
    float dist = magnitude(L); // CHECK IF OPENCL DISTANCE DOES THE SAME
	L /= dist;

    if (dot(N, L) > 0 && dot(primNorms[0], -L) > 0) {

        int si = atomic_inc(shadowCounter);

        shadowOrigins[si] = I + L * 0.001f;
        shadowDirections[si] = L;
        shadowDistances[si] = dist - 2.0f * 0.001f;

        shadowPixelIdxs[si] = pixelIdxs[threadId]; 

        float solidAngle = (dot(primNorms[0], -L) * A) / (dist * dist);
        float lightPDF = 1.0f / solidAngle;

        // DOUBLE CHECK
        energies[si] = transmissions[pixelIdxs[threadId]] * (dot(N, L) / lightPDF) * BRDF * emission; 
    }
    
    // Russian Roulette
    
    float p = clamp(max(albedos[primIdxs[threadId]].z, max(albedos[primIdxs[threadId]].x, albedos[primIdxs[threadId]].y)), 0.0f, 1.0f);
    //printf("float: %f", p);
    if (p >= RandomFloat()) {
        // continue random walk
        int ei = atomic_inc(bounceCounter);
        transmissions[pixelIdxs[threadId]] *= 1.0f / p;

        // Calculate a diffuse reflection
        float3 R = (float3) (Rand(2.0f) - 1.0f, Rand(2.0f) - 1.0f, Rand(2.0f) - 1.0f);

        while (magnitude(R) > 1.0f)
        {
            R = (float3)(Rand(2.0f) - 1.0f, Rand(2.0f) - 1.0f, Rand(2.0f) - 1.0f);
        }

        R = normalize(R);

        if (dot(N, R) < 0) 
            R = -R;

        float hemiPDF = 1.0f / (M_PI_F * 2.0f);
        bounceOrigins[ei] = I + R * 0.001f;
        bounceDirections[ei] = R;

        bouncePixelIdxs[ei] = pixelIdxs[threadId];
                
        transmissions[pixelIdxs[threadId]] *= (dot(N, R) / hemiPDF) * BRDF;
    }

}