#define SCRWIDTH 1280
#define SCRHEIGHT 720

__kernel void GeneratePrimaryRays(__global int* pixelIdxs, __global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs, __global int* lastSpecular, __global int* insides,// Primary Rays
__global float4* energies, __global float4* transmissions,                                                                                                                  // E & T
__global float4* camProp)                                                                                                                                                 // Camera Properties
{
    int threadId = get_global_id(0);
    
    int x = threadId % SCRWIDTH;
    int y = threadId / SCRWIDTH;

    float u = (float)x * (1.0f / SCRWIDTH);
	float v = (float)y * (1.0f / SCRHEIGHT);

    float4 P = camProp[1] + u * (camProp[2] - camProp[1]) + v * (camProp[3] - camProp[1]);

    origins[threadId] = camProp[0];

    directions[threadId] = normalize(P - camProp[0]);

    distances[threadId] = INFINITY; // ONNODIG DOET EXTEND KERNEL AL
    primIdxs[threadId] = -1;        // ONNODIG DOET EXTEND KERNEL AL

    lastSpecular[threadId] = 1;
    insides[threadId] = -1;

    // Reset Buffers to initial values
    pixelIdxs[threadId] = threadId;
    energies[threadId] = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    transmissions[threadId] = (float4) (1.0f, 1.0f, 1.0f, 0.0f); // Rename to throughput
    
}