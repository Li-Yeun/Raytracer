float RandomFloat(int seed) { return seed * 2.3283064365387e-10f; }

uint RandomUInt(int seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

__kernel void Shade(__global int* pixelIdxs, __global float3* origins, __global float3* directions, __global float* distances, __global int* primIdxs, 
__global float3* albedos, __global float3* primNorms, __global float* sphereInvrs, int sphereStartIdx, int sphereSize,
__global float3* lightCorners, __global float A, __global float s, __global float3 emission,
__global int* shadowPixelIdxs, __global float3* shadowOrigins, __global float3* shadowDirections, __global float* shadowDistances, __global float3* energies,
__global int* bouncePixelIdxs, __global float3* bounceOrigins, __global float3* bounceDirections,
__global seed)
{   
    int threadId = get_global_id(0);

    // TODO MIGHT CHANGE LATER
    if(primIdxs[threadId] == -1)
        return;
    
    float3 I = origins[threadId] + directions[threadId] * distances[threadId];

    float3 N;
    if(primIdxs[threadId] >= sphereStartIdx && primIdxs[threadId] < sphereIdx + sphereSize)
        N = (I - primNorms[primIdxs[threadId]]) * sphereInvrs[primIdxs[threadId] - sphereStartIdx];

    N = primNorms[primIdxs[threadId]];
    
    if (dot( N, directions[threadId] ) > 0) N = -N; // hit backside / inside

    float3 BRDF = albedos[primIdxs[threadId]] / PI;

    // Pick random position
    float3 c1c2 = normalize(lightCorners[0] - lightCorners[1]);
    
    uint newSeed = RandomUInt(seed);

    float randomLength = RandomFloat(newSeed) * s;
    float3 u = c1c2 * randomLength;
    float3 c2c3 = normalize(lightCorners[2] - lightCorners[1]);

    newSeed = RandomUInt(newSeed);
    randomLength = RandomFloat(newSeed) * s;
    float3 v = c2c3 * randomLength;

    float3 light_point = lightCorners[1] + u + v - (float3)(0, 0.01f, 0);
    
    float3 L = light_point - I;
    float dist = sqrtf(L[0] * L[0] + L[1] * L[1] + L[2] * L[2])
	L = normalize(L);

    if (dot(N, L) > 0 && dot(primNorms[0], -L) > 0) {

        int si = atomicInc( shadowRayIdx )

        shadowOrigins[si] = I + L * 0.001f
        shadowDirections[si] = L;
        shadowDistances[si] = dist - 2.0f * 0.001f;

        shadowPixelIdxs[si] = pixelIdxs[threadId]; 

        float solidAngle = (dot(Nl, -L) * A) / sqrf(dist);
        float lightPDF = 1.0f / solidAngle;

        // TODO (T) transmissions buffer
        float3 E = T * (dot(normal, L) / lightPDF) * BRDF * std::get<3>(result);
        energies[si] = E; 
        
        // TO BE CONTINUED
        shadowBuffer[si] = ShadowRay( … )
    }
    
    
    // Russian Roulette
    newSeed = RandomUInt(newSeed);
    
    float p = clamp(max(albedos[primIdxs[threadId]][2], max(albedos[primIdxs[threadId]][0], albedo[primIdxs[threadId]][1])), 0.0f, 1.0f); // TODO CHECK IF DOUBLE INDEX IS VALID
    if (p < RandomFloat(newSeed)) return; else T *= 1.0f / p;

    // continue random walk
    float3 R = scene.DiffuseReflection(normal);
    float hemiPDF = 1.0f / (PI * 2.0f);
    ray = Ray(intersection + R * 0.001f, R);
    T *= (dot(normal, R) / hemiPDF) * BRDF;

    if (bounce) {
    // TO BE CONTINUED
    int ei = atomicInc( extensionRayIdx )
    newRayBuffer[ei] = ExtensionRay( … )
    }
}