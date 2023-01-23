#define SCRWIDTH 200//1280
#define SCRHEIGHT 200//720

__global int counter = 0; // Check if this has to be 0 or -1

__kernel void GenerateInitialPrimaryRays(__global int* pixelIdxs, __global float3* origins, __global float3* directions, __global float* distances, __global int* primIdxs,  // Primary Rays
 __global float3* energies, __global float3* transmissions,                                                                                                                  // E & T
float aspect, float3 camPos)                                                                                                                                                 // Camera Properties
{
    int threadId = get_global_id(0);
    
    int x = threadId % SCRWIDTH;
    int y = threadId / SCRWIDTH;

    float u = (float)x * (1.0f / SCRWIDTH);
	float v = (float)y * (1.0f / SCRHEIGHT);
    
    float3 topLeft = (float3) ( -aspect, 1.0f, 0.0f );
	float3 topRight = (float3)( aspect, 1.0f, 0.0f );
	float3 bottomLeft = (float3) ( -aspect, -1.0f, 0.0f );

    float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);

    origins[threadId] = camPos;
    directions[threadId] = normalize(P - camPos);

    distances[threadId] = INFINITY;
    primIdxs[threadId] = -1;

    // Reset Buffers to initial values
    pixelIdxs[threadId] = threadId;
    energies[threadId] = (float3) (0.0f, 0.0f, 0.0f);
    transmissions[threadId] = (float3) (1.0f, 1.0f, 1.0f);
}

__kernel void GeneratePrimaryRays(__global int* rayCounter, __global int* pixelIdxs,  __global float3* origins, __global float3* directions, __global float* distances, __global int* primIdxs, // Primary Rays
__global int* shadowCounter,                                                                                                                                                                    // Shadow Rays  
__global int* bounceCounter, __global int* bouncePixelIdxs, __global float3* bounceOrigins, __global float3* bounceDirections)                                                                  // Bounce Rays                                                                                                                                                                  // Camera Properties
{
    int threadId = get_global_id(0);
    
    if(threadId >= *bounceCounter) 
        return;

    origins[threadId] = bounceOrigins[threadId];
    directions[threadId] = bounceDirections[threadId];

    distances[threadId] = INFINITY;
    primIdxs[threadId] = -1;

    pixelIdxs[threadId] = bouncePixelIdxs[threadId];

    int ri = atomic_inc(&counter);

    if(ri == *bounceCounter - 1) 
    {
        atomic_xchg(&counter, 0);
        atomic_xchg(rayCounter, *bounceCounter);
        atomic_xchg(bounceCounter, 0);
        atomic_xchg(shadowCounter, 0);
    }
}