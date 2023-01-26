// int type;        // shpere: 0,       triangle: 1,        quad: 2,        plane: 3
// float3 info1;    // shpere: pos,     triangle: pos1,     quad: _,        plane: N
// float3 info2;    // shpere: _,       triangle: pos2,     quad: _,        plane: _
// float3 info3;    // shpere: _,       triangle: pos3,     quad: _,        plane: _
// float infofloat; // shpere: r2,      triangle: _,        quad: size,     plane: d
// mat4 infoMatrix; // shpere: _,       triangle: _,        quad: invT,     plane: _

int planeRootNode = 1, primitveRootNodeIdx = 2;
int spheres_size = 2, quads_size = 1, planes_size = 6, cube_size = 1; //TODO: Not hardcode this info?

// Multiply a 4x4 matrix by a 4x1 vector
float4 MultiplyMatrix(float4 a, float16 matrix)
{
    float4 result = (float4)(0, 0, 0, 0);
    for (int i = 0; i < 4; i++)
    {
        result.x += a[i] * matrix[i];
        result.y += a[i] * matrix[i + 4];
        result.z += a[i] * matrix[i + 8];
        result.w += a[i] * matrix[i + 12];
    }
    return (float4)result;
}

bool IntersectAABB(const float4 rayOrigin, const float4 rayDirection, const float rayDistance, const float4 bmin, const float4 bmax)
	{
        float3 rD = (float3)(1 / rayDirection.x, 1 / rayDirection.y, 1 / rayDirection.z);

		float tx1 = (bmin.x - rayOrigin.x) * rD.x, tx2 = (bmax.x - rayOrigin.x) * rD.x;
		float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
		float ty1 = (bmin.y - rayOrigin.y) * rD.y, ty2 = (bmax.y - rayOrigin.y) * rD.y;
		tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
		float tz1 = (bmin.z - rayOrigin.z) * rD.z, tz2 = (bmax.z - rayOrigin.z) * rD.z;
		tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
		return tmax >= tmin && tmin < rayDistance && tmax > 0;
	}


__kernel void Extend(__global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs, // ray data
                    int primCount, __global int* primTypes, //Primitive metadata
                    __global float4* primInfo1, __global float4* primInfo2, __global float4* primInfo3, // Primitive float3 data
                    __global float* primInfofloat, __global float16* primInfoMatrix, // primitive misc data
                    __global float4* aabbMin, __global float4* aabbMax, __global uint* leftFirst, __global uint* primitiveCount, __global int* bvhPrimitiveIdx) // BVH data 
{   
    int i = get_global_id(0);
    float3 direction = (float3)(directions[i].x, directions[i].y, directions[i].z);
    float3 origin = (float3)(origins[i].x, origins[i].y, origins[i].z);


    // traverse BVH
    for (uint planeIndex = 0; planeIndex < primitiveCount[planeRootNode]; planeIndex++)
    {
        float3 info1 = (float3)(primInfo1[planeIndex].x, primInfo1[planeIndex].y, primInfo1[planeIndex].z);

        float t = -(dot(origin, info1) + primInfofloat[planeIndex]) / (dot(direction, info1));
        if (t < distances[i] && t > 0) distances[i] = t, primIdxs[i] = planeIndex;
    }

    // traverse BVH iteratively
    int stack[64];
    int* stackPtr = stack;
    *stackPtr++ = primitveRootNodeIdx; // push
    int stackSize = 1;

    
while (stackSize > 0)
    {
        int nodeIdx = *--stackPtr; // pop
        stackSize--;

        if (!IntersectAABB(origins[i], directions[i], distances[i], aabbMin[nodeIdx], aabbMax[nodeIdx])) continue;
        // if node is leaf
        if (primitiveCount[nodeIdx] > 0)
        {
            for (int i = 0; i < primitiveCount[nodeIdx]; i++)
            {
                int primId = bvhPrimitiveIdx[leftFirst[nodeIdx] + i];
                float3 info1 = (float3)(primInfo1[primId].x, primInfo1[primId].y, primInfo1[primId].z);
                float3 info2 = (float3)(primInfo2[primId].x, primInfo2[primId].y, primInfo2[primId].z);
                float3 info3 = (float3)(primInfo3[primId].x, primInfo3[primId].y, primInfo3[primId].z);

                float3 origin3 = (float3)(origin.x, origin.y, origin.z);
                float3 direction3 = (float3)(direction.x, direction.y, direction.z);


                if ( primId < spheres_size) {
                    // Intersect sphere
                    float3 oc = origin3 - info1;
                    float b = dot(oc, direction3);
                    float c = dot(oc, oc) - primInfofloat[primId];
                    float t, d = b * b - c;
                    if (d <= 0) return;
                    d = sqrt(d), t = -b - d;
                    if (t < distances[i] && t > 0)
                    {
                        distances[i] = t, primIdxs[i] = primId;
                        return;
                    }
                    t = d - b;
                    if (t < distances[i] && t > 0)
                    {
                        distances[i] = t, primIdxs[i] = primId;
                        return;
                    }
                }
                else {
                    //intersect triangle
                    const float3 edge1 = info2 - info1;
                    const float3 edge2 = info3 - info1;
                    const float3 h = cross(direction, edge2);
                    const float a = dot(edge1, h);
                    if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
                    const float f = 1 / a;
                    const float3 s = origin - info1;
                    const float u = f * dot(s, h);
                    if (u < 0 || u > 1) return;
                    const float3 q = cross(s, edge1);
                    const float v = f * dot(direction, q);
                    if (v < 0 || u + v > 1) return;
                    const float t = f * dot(edge2, q);
                    if (t > 0.0001f && t < distances[i]) {
                        distances[i] = t;
                        primIdxs[i] = primId;
                    }
                }
            }
        }
        else
        {
            *stackPtr++ = leftFirst[nodeIdx]; // push
            *stackPtr++ = leftFirst[nodeIdx] + 1; // push
            stackSize += 2;
        }

    }
}