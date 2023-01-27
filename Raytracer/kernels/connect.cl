float3 MultiplyMatrix(float4 b, float16 a)
{
    return (float3)(a.s0 * b.x + a.s1 * b.y + a.s2 * b.z + a.s3 * b.w,
        a.s4 * b.x + a.s5 * b.y + a.s6 * b.z + a.s7 * b.w,
		a.s8 * b.x + a.s9 * b.y + a.sA * b.z + a.sB * b.w);
}

__kernel void Connect(__global int* shadowCounter, __global int* shadowPixelIdxs, __global float4* shadowOrigins, __global float4* shadowDirections, __global float* shadowDistances,
int quads_size, int spheres_size, int cubes_size, int planes_size, int triangles_size,
__global float16* quadMatrices, __global float* quadSizes, __global float4* sphereInfos, __global float4* primNorms, __global float4* triangleInfos,
__global float4* energies, __global float4* accumulator)
{
    int threadId = get_global_id(0);

    if(threadId >= *shadowCounter) 
    {
        return;
    }

    float rayT = shadowDistances[threadId];
    float3 rayO = shadowOrigins[threadId].xyz;
    float3 rayD = shadowDirections[threadId].xyz;

    bool isOccluded = false;

    for(int i = 0; i < quads_size; i++)
    {
        float3 O = MultiplyMatrix((float4)(rayO, 1), quadMatrices[i]);
        float3 D = MultiplyMatrix((float4)(rayD, 0), quadMatrices[i]);
        float t = O.y / -D.y;

        if (t < rayT && t > 0)
        {
            float3 I = O + t * D;
            if (I.x > -quadSizes[i] && I.x < quadSizes[i] && I.z > -quadSizes[i] && I.z < quadSizes[i])
            {
                isOccluded = true;
                break;
            }
        }
    }

    for (int i = 0; i < spheres_size; i++)
    {
        if(isOccluded)
            break;

        float3 pos = sphereInfos[i].xyz;
        float r2 = sphereInfos[i].w;

        float3 oc = rayO - pos;
        float b = dot(oc, rayD);
        float c = dot(oc, oc) - r2;
        float t;
        float d = b * b - c;
        if (d <= 0) continue;
        d = sqrt(d);
        t = -b - d;

        if (t < rayT && t > 0)
        {
            isOccluded = true;
            break;
        }
        t = d - b;
        if (t < rayT && t > 0)
        {
            isOccluded = true;
            break;
        } 
    }

    for (int i = 0; i < cubes_size; i++)
    {
        //TODO
        if(isOccluded)
            break;
    }

    for (int i = 0; i < planes_size; i++)
    {
        if(isOccluded)
            break;

        float3 N = primNorms[quads_size + spheres_size + cubes_size + i].xyz;
        float d = primNorms[quads_size + spheres_size + cubes_size + i].w;
        float t = -(dot(rayO, N) + d) / (dot(rayD, N));

        if (t < rayT && t > 0) 
        {   
            isOccluded = true;
            break;
        }
    }

    for (int i = 0; i < triangles_size; i++)
    {
        if(isOccluded)
            break;

        float3 pos1 = triangleInfos[i * 3].xyz;
        float3 pos2 = triangleInfos[i * 3 + 1].xyz;
        float3 pos3 = triangleInfos[i * 3 + 2].xyz;

        // No intersection if ray and plane are parallel
        float3 edge1 = pos2 - pos1;
        float3 edge2 = pos3 - pos1;
        float3 h = cross(rayD, edge2);
        float a = dot(edge1, h);
        if (a > -0.0001f && a < 0.0001f) 
            continue; // ray parallel to triangle

        float f = 1 / a;
        float3 s = rayO - pos1;
        float u = f * dot(s, h);
        if (u < 0 || u > 1) 
            continue;
        float3 q = cross(s, edge1);
        float v = f * dot(rayD, q);
        if (v < 0 || u + v > 1) 
            continue;
        float t = f * dot(edge2, q);

        if (t > 0.0001f && t < rayT) {
            isOccluded = true;
            break;
        }
    }

    if(!isOccluded)
        accumulator[shadowPixelIdxs[threadId]] += energies[threadId];
}