#define SCRWIDTH 1280
#define SCRHEIGHT 720

__global int counter = 0;

__kernel void GenerateInitialPrimaryRays(__global int* pixelIdxs, __global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs, __global int* lastSpecular, __global int* insides,// Primary Rays
__global float4* energies, __global float4* transmissions,                                                                                                                  // E & T
float aspect, float4 camPos)                                                                                                                                                 // Camera Properties
{
    int threadId = get_global_id(0);
    
    int x = threadId % SCRWIDTH;
    int y = threadId / SCRWIDTH;

    float u = (float)x * (1.0f / SCRWIDTH);
	float v = (float)y * (1.0f / SCRHEIGHT);
    
    float4 topLeft = (float4) ( -aspect, 1.0f, 0.0f, 0.0f );
	float4 topRight = (float4)( aspect, 1.0f, 0.0f, 0.0f );
	float4 bottomLeft = (float4) ( -aspect, -1.0f, 0.0f, 0.0f );

    float4 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);

    origins[threadId] = camPos;

    directions[threadId] = normalize(P - camPos);

    distances[threadId] = INFINITY; // ONNODIG DOET EXTEND KERNEL AL
    primIdxs[threadId] = -1;        // ONNODIG DOET EXTEND KERNEL AL

    lastSpecular[threadId] = 1;
    insides[threadId] = -1;

    // Reset Buffers to initial values
    pixelIdxs[threadId] = threadId;
    energies[threadId] = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    transmissions[threadId] = (float4) (1.0f, 1.0f, 1.0f, 0.0f); // Rename to throughput
    
}

__kernel void GeneratePrimaryRays(__global int* rayCounter, __global int* pixelIdxs,  // Primary Rays
__global int* shadowBounceCounterBuffer, __global int* bouncePixelIdxs)                                                                                                                                     // Bounce Rays                                                                                                                                                                  // Camera Properties
{
    int threadId = get_global_id(0);
    
    __global int* bounceCounter = &shadowBounceCounterBuffer[1];

    pixelIdxs[threadId] = bouncePixelIdxs[threadId];

    int ri = atomic_inc(&counter);

    if(ri == *bounceCounter - 1) 
    {
        atomic_xchg(&counter, 0);
        atomic_xchg(rayCounter, *bounceCounter);

        __global int* shadowCounter = &shadowBounceCounterBuffer[0];
        atomic_xchg(shadowCounter, 0);
        atomic_xchg(bounceCounter, 0);
    }
}