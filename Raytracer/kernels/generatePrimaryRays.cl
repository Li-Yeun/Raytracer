#define SCRWIDTH 1280
#define SCRHEIGHT 720

__kernel void GeneratePrimaryRays(__global float4* accumulator)
{
    int threadId = get_global_id(0);
    
    if(threadId >= SCRHEIGHT / 2 * SCRWIDTH)
        accumulator[threadId] += (float4) (1.0f, 1.0f, 1.0f, 1.0f);
    else
        accumulator[threadId] += (float4) (0, 0, 0, 0);

}