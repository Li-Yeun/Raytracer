#define SCRWIDTH 1280
#define SCRHEIGHT 720

uint RGBF32_to_RGB8( float3 v, int f )
{
	uint r = (uint)(255.0f * min( 1.0f, v.x / f));
	uint g = (uint)(255.0f * min( 1.0f, v.y / f));
	uint b = (uint)(255.0f * min( 1.0f, v.z / f));
	return (r << 16) + (g << 8) + b;
}

__kernel void Finalize(__global uint* pixels, __global float3* accumulator, int accumulatedFrames)
{   
    int threadId = get_global_id(0);
    pixels[threadId] = RGBF32_to_RGB8(accumulator[threadId], accumulatedFrames);
}