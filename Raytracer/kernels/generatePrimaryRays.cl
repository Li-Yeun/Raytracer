#define SCRWIDTH 1280
#define SCRHEIGHT 720

__kernel void GeneratePrimaryRays(__global float4* accumulator)
{
    int id = get_global_id(0);
    
    if(id >= SCRHEIGHT / 2 * SCRWIDTH)
        accumulator[id] = (float4) (1.0f, 1.0f, 1.0f, 1.0f);
    else
        accumulator[id] = (float4) (0, 0, 0, 0);

}