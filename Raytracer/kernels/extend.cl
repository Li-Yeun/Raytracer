// int primMats;    // DIFFUSE = 0, MIRROR = 1, GLASS = 2, SUBSTANCE = 1, LIGHT = 0 

// Multiply a 4x4 matrix by a 4x1 vector and convert it to a 3x1 vector
float3 MultiplyMatrix(float4 b, float16 a)
{
    return (float3)(a.s0 * b.x + a.s1 * b.y + a.s2 * b.z + a.s3 * b.w,
        a.s4 * b.x + a.s5 * b.y + a.s6 * b.z + a.s7 * b.w,
		a.s8 * b.x + a.s9 * b.y + a.sA * b.z + a.sB * b.w);
}

float IntersectAABBStack(float3 rayOrigin, float3 rayDirection, float rayDistance, float3 bmin, float3 bmax)
{
    float3 rD = (float3)(1 / rayDirection.x, 1 / rayDirection.y, 1 / rayDirection.z);

    float tx1 = (bmin.x - rayOrigin.x) * rD.x, tx2 = (bmax.x - rayOrigin.x) * rD.x;
    float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
    float ty1 = (bmin.y - rayOrigin.y) * rD.y, ty2 = (bmax.y - rayOrigin.y) * rD.y;
    tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
    float tz1 = (bmin.z - rayOrigin.z) * rD.z, tz2 = (bmax.z - rayOrigin.z) * rD.z;
    tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
    if (tmax >= tmin && tmin < rayDistance && tmax > 0) return tmin; else return 1e30f;
}

struct GPUBVHNode
{
	float aabbMinx, aabbMiny, aabbMinz;
	float aabbMaxx, aabbMaxy, aabbMaxz;
	int leftFirst, primitiveCount;
};

__kernel void InitialExtend(__global int* pixelIdxs, __global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs,   // Primary Rays
int quads_size, int spheres_size, int cubes_size, int planes_size, int triangles_size,
__global float16* quadMatrices, __global float* quadSizes, __global float4* sphereInfos, __global float16* cubeInvMatrices, __global float4* cubeB, __global float4* primNorms, __global float4* triangleInfos,
__global struct GPUBVHNode* bvhNodes, __global int* bvhPrimitiveIdx)
{
    int threadId = get_global_id(0);
    
    int rayPixelIdx = pixelIdxs[threadId];

    float rayT = INFINITY;
    int rayObjIdx = -1;

    float3 rayO = origins[rayPixelIdx].xyz;
    float3 rayD = directions[rayPixelIdx].xyz;

    // Quad intersection
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
                rayT = t; 
                rayObjIdx = i;
            }
        }
    }

    int cubeStartIdx = quads_size + spheres_size;

    for(int i = 0; i < cubes_size; i++)
    {
        int offset = i * 2;
        float3 O = MultiplyMatrix((float4)(rayO, 1), cubeInvMatrices[i]);
        float3 D = MultiplyMatrix((float4)(rayD, 0), cubeInvMatrices[i]);
        float rDx = 1 / D.x, rDy = 1 / D.y, rDz = 1 / D.z;
        int signx = D.x < 0, signy = D.y < 0, signz = D.z < 0;
        float tmin = (cubeB[signx + offset].x - O.x) * rDx;
        float tmax = (cubeB[1 - signx + offset].x - O.x) * rDx;
        float tymin = (cubeB[signy + offset].y - O.y) * rDy;
        float tymax = (cubeB[1 - signy + offset].y - O.y) * rDy;

        if (tmin > tymax || tymin > tmax) 
            continue;

        tmin = max(tmin, tymin), tmax = min(tmax, tymax);
        float tzmin = (cubeB[signz + offset].z - O.z) * rDz;
        float tzmax = (cubeB[1 - signz + offset].z - O.z) * rDz;

        if (tmin > tzmax || tzmin > tmax) 
            continue;
            
        tmin = max(tmin, tzmin), tmax = min(tmax, tzmax);

        if (tmin > 0)
        {
            if (tmin < rayT) 
            {
                rayT = tmin; 
                rayObjIdx = cubeStartIdx + i;
            }
        }
        else if (tmax > 0)
        {
            if (tmax < rayT) 
            {
                rayT = tmax; 
                rayObjIdx = cubeStartIdx + i;
            }
        }
    }

    int planeStartIdx = cubeStartIdx + cubes_size;
    // Plane intersection
    for(int i = 0; i < planes_size; i++)
    {
        int currentPlaneIdx = i + planeStartIdx;
        float3 N = primNorms[currentPlaneIdx].xyz;
        float d = primNorms[currentPlaneIdx].w;
        float t = -(dot(rayO, N) + d) / (dot(rayD, N));

        if (t < rayT && t > 0) 
        {   
            rayT = t;
            rayObjIdx = currentPlaneIdx;
        }
    }

    struct GPUBVHNode* node = &bvhNodes[2], *stack[32]; // *stack[64] wordt gebruik in cpu code
    uint stackPtr = 0;
    
    int triangleIdxOffset = quads_size + cubes_size + planes_size;

    while (1)
    {
        // if(threadId == 1) printf("start while\n");
        if (node->primitiveCount > 0) // isLeaf()
        {
            for (uint i = 0; i < node->primitiveCount; i++)
            {
                int primId = bvhPrimitiveIdx[node->leftFirst + i];
                
                if (primId < spheres_size)
                {
                    float3 pos = sphereInfos[primId].xyz;
                    float r2 = sphereInfos[primId].w;

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
                        rayT = t; 
                        rayObjIdx = primId + quads_size;
                        continue;
                    }
                    t = d - b;
                    if (t < rayT && t > 0)
                    {
                        rayT = t;
                        rayObjIdx = primId + quads_size;
                        continue;
                    } 

                }
                else
                {
                    int triangleIdx = (primId - spheres_size) * 3;
                    float3 pos1 = triangleInfos[triangleIdx].xyz;
                    float3 pos2 = triangleInfos[triangleIdx + 1].xyz;
                    float3 pos3 = triangleInfos[triangleIdx + 2].xyz;

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
                        rayT = t;   
                        rayObjIdx = primId + triangleIdxOffset;
                    }
                }

            }
            if (stackPtr == 0) break; else node = stack[--stackPtr];
            continue;
        }

        struct GPUBVHNode* child1 = &bvhNodes[node->leftFirst];
        struct GPUBVHNode* child2 = &bvhNodes[node->leftFirst + 1];

        float3 bmin = (float3)(child1->aabbMinx, child1->aabbMiny, child1->aabbMinz);
        float3 bmax = (float3)(child1->aabbMaxx, child1->aabbMaxy, child1->aabbMaxz);

        float dist1 = IntersectAABBStack( rayO, rayD, rayT, bmin, bmax );

        bmin = (float3)(child2->aabbMinx, child2->aabbMiny, child2->aabbMinz);
        bmax = (float3)(child2->aabbMaxx, child2->aabbMaxy, child2->aabbMaxz);
        float dist2 = IntersectAABBStack( rayO, rayD, rayT, bmin, bmax );
        if (dist1 > dist2) 
        { 
            float d = dist1; dist1 = dist2; dist2 = d;
            struct GPUBVHNode* c = child1; child1 = child2; child2 = c; 
        }
        if (dist1 == 1e30f)
        {
            if (stackPtr == 0) break; else node = stack[--stackPtr];
        }
        else
        {
            node = child1;
            if (dist2 != 1e30f) stack[stackPtr++] = child2;
        }
    }

    distances[rayPixelIdx] = rayT;
    primIdxs[rayPixelIdx] = rayObjIdx;
}

__global int counter = 0;
__kernel void Extend(__global int* pixelIdxs, __global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs,   // Primary Rays
int quads_size, int spheres_size, int cubes_size, int planes_size, int triangles_size,
__global float16* quadMatrices, __global float* quadSizes, __global float4* sphereInfos, __global float16* cubeInvMatrices, __global float4* cubeB, __global float4* primNorms, __global float4* triangleInfos,
__global struct GPUBVHNode* bvhNodes, __global int* bvhPrimitiveIdx,
__global int* shadowBounceCounter, __global int* bouncePixelIdxs)
{
    int threadId = get_global_id(0);

    int rayPixelIdx = bouncePixelIdxs[threadId];
    pixelIdxs[threadId] = rayPixelIdx;

    float rayT = INFINITY;
    int rayObjIdx = -1;

    float3 rayO = origins[rayPixelIdx].xyz;
    float3 rayD = directions[rayPixelIdx].xyz;

    // Quad intersection
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
                rayT = t; 
                rayObjIdx = i;
            }
        }
    }

    int cubeStartIdx = quads_size + spheres_size;

    for(int i = 0; i < cubes_size; i++)
    {
        int offset = i * 2;
        float3 O = MultiplyMatrix((float4)(rayO, 1), cubeInvMatrices[i]);
        float3 D = MultiplyMatrix((float4)(rayD, 0), cubeInvMatrices[i]);
        float rDx = 1 / D.x, rDy = 1 / D.y, rDz = 1 / D.z;
        int signx = D.x < 0, signy = D.y < 0, signz = D.z < 0;
        float tmin = (cubeB[signx + offset].x - O.x) * rDx;
        float tmax = (cubeB[1 - signx + offset].x - O.x) * rDx;
        float tymin = (cubeB[signy + offset].y - O.y) * rDy;
        float tymax = (cubeB[1 - signy + offset].y - O.y) * rDy;

        if (tmin > tymax || tymin > tmax) 
            continue;

        tmin = max(tmin, tymin), tmax = min(tmax, tymax);
        float tzmin = (cubeB[signz + offset].z - O.z) * rDz;
        float tzmax = (cubeB[1 - signz + offset].z - O.z) * rDz;

        if (tmin > tzmax || tzmin > tmax) 
            continue;
            
        tmin = max(tmin, tzmin), tmax = min(tmax, tzmax);

        if (tmin > 0)
        {
            if (tmin < rayT) 
            {
                rayT = tmin; 
                rayObjIdx = cubeStartIdx + i;
            }
        }
        else if (tmax > 0)
        {
            if (tmax < rayT) 
            {
                rayT = tmax; 
                rayObjIdx = cubeStartIdx + i;
            }
        }
    }

    int planeStartIdx = cubeStartIdx + cubes_size;
    // Plane intersection
    for(int i = 0; i < planes_size; i++)
    {
        int currentPlaneIdx = i + planeStartIdx;
        float3 N = primNorms[currentPlaneIdx].xyz;
        float d = primNorms[currentPlaneIdx].w;
        float t = -(dot(rayO, N) + d) / (dot(rayD, N));

        if (t < rayT && t > 0) 
        {   
            rayT = t;
            rayObjIdx = currentPlaneIdx;
        }
    }
        
    struct GPUBVHNode* node = &bvhNodes[2], *stack[32]; // *stack[64] wordt gebruik in cpu code
    uint stackPtr = 0;
    
    int triangleIdxOffset = quads_size + cubes_size + planes_size;

    while (1)
    {
        // if(threadId == 1) printf("start while\n");
        if (node->primitiveCount > 0) // isLeaf()
        {
            for (uint i = 0; i < node->primitiveCount; i++)
            {
                int primId = bvhPrimitiveIdx[node->leftFirst + i];
                
                if (primId < spheres_size)
                {
                    float3 pos = sphereInfos[primId].xyz;
                    float r2 = sphereInfos[primId].w;

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
                        rayT = t; 
                        rayObjIdx = primId + quads_size;
                        continue;
                    }
                    t = d - b;
                    if (t < rayT && t > 0)
                    {
                        rayT = t;
                        rayObjIdx = primId + quads_size;
                        continue;
                    } 

                }
                else
                {
                    int triangleIdx = (primId - spheres_size) * 3;
                    float3 pos1 = triangleInfos[triangleIdx].xyz;
                    float3 pos2 = triangleInfos[triangleIdx + 1].xyz;
                    float3 pos3 = triangleInfos[triangleIdx + 2].xyz;

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
                        rayT = t;   
                        rayObjIdx = primId + triangleIdxOffset;
                    }
                }

            }
            if (stackPtr == 0) break; else node = stack[--stackPtr];
            continue;
        }

        struct GPUBVHNode* child1 = &bvhNodes[node->leftFirst];
        struct GPUBVHNode* child2 = &bvhNodes[node->leftFirst + 1];

        float3 bmin = (float3)(child1->aabbMinx, child1->aabbMiny, child1->aabbMinz);
        float3 bmax = (float3)(child1->aabbMaxx, child1->aabbMaxy, child1->aabbMaxz);

        float dist1 = IntersectAABBStack( rayO, rayD, rayT, bmin, bmax );

        bmin = (float3)(child2->aabbMinx, child2->aabbMiny, child2->aabbMinz);
        bmax = (float3)(child2->aabbMaxx, child2->aabbMaxy, child2->aabbMaxz);
        float dist2 = IntersectAABBStack( rayO, rayD, rayT, bmin, bmax );
        if (dist1 > dist2) 
        { 
            float d = dist1; dist1 = dist2; dist2 = d;
            struct GPUBVHNode* c = child1; child1 = child2; child2 = c; 
        }
        if (dist1 == 1e30f)
        {
            if (stackPtr == 0) break; else node = stack[--stackPtr];
        }
        else
        {
            node = child1;
            if (dist2 != 1e30f) stack[stackPtr++] = child2;
        }
    }

    distances[rayPixelIdx] = rayT;
    primIdxs[rayPixelIdx] = rayObjIdx;

    __global int* bounceCounter = &shadowBounceCounter[1];
    int ri = atomic_inc(&counter);
    if(ri == *bounceCounter - 1) 
    {
        counter = 0;

        shadowBounceCounter[0] = 0;
        shadowBounceCounter[1] = 0;
    }
}