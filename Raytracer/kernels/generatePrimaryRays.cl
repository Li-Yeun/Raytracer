#define SCRWIDTH 1280
#define SCRHEIGHT 720

__kernel void GenerateInitialPrimaryRays(__global int* pixelIdxs, __global float3* origins, __global float3* directions, __global float* distances, __global int* primIdxs, 
 __global float3* energies, __global float3* transmissions,
float aspect, float3 camPos)
{
    int threadId = get_global_id(0);
    
    int x = threadId / SCRWIDTH;
    int y = threadId % SCRWIDTH;

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

__kernel void GeneratePrimaryRays(__global int* pixelIdxs, __global int* bouncePixelIdxs, __global float3* origins, __global float3* directions, __global float* distances, __global int* primIdxs, 
float aspect, float3 camPos)
{
    int threadId = get_global_id(0);
    
    // TODO CHANGE AND DELETE
    if(distances[threadId] == -1.0f)
        return;
    
    int x = threadId / SCRWIDTH;
    int y = threadId % SCRWIDTH;

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

    pixelIdxs[threadId] = bouncePixelIdxs[threadId];
}