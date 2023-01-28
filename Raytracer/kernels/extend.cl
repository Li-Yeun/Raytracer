// int primMats;    // DIFFUSE = 0, MIRROR = 1, GLASS = 2, SUBSTANCE = 1, LIGHT = 0 

/*
float rayT;
int rayObjIdx;
float3 rayO;
float3 rayD;
*/

// Multiply a 4x4 matrix by a 4x1 vector and convert it to a 3x1 vector
float3 MultiplyMatrix(float4 b, float16 a)
{
    return (float3)(a.s0 * b.x + a.s1 * b.y + a.s2 * b.z + a.s3 * b.w,
        a.s4 * b.x + a.s5 * b.y + a.s6 * b.z + a.s7 * b.w,
		a.s8 * b.x + a.s9 * b.y + a.sA * b.z + a.sB * b.w);
}

/*
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
*/


/*
void QuadIntersect()
{
    //TODO
    return;
}

void SphereIntersect(int objIdx, float3 pos, float r2)
{
    float3 oc = rayO - pos;
    float b = dot(oc, rayD);
    float c = dot(oc, oc) - r2;
    float t, d = b * b - c;
    if (d <= 0) return;
    d = sqrt(d), t = -b - d;

    if (t < rayT && t > 0)
    {
        rayT = t; 
        rayObjIdx = objIdx;
        return;
    }
    t = d - b;
    if (t < rayT && t > 0)
    {
        rayT = t;
        rayObjIdx = objIdx;
        return;
    } 
}

void CubeIntersect()
{
    return;
    //TODO
}

void PlaneIntersect(int objIdx, float3 N, float d)
{
    float t = -(dot(rayO, N) + d) / (dot(rayD, N));

    if (t < rayT && t > 0) 
    {   rayT = t;
        rayObjIdx = objIdx;
    }
}

void triangleIntersect(int objIdx, float3 pos1, float3 pos2, float3 pos3)
{
    // No intersection if ray and plane are parallel
    float3 edge1 = pos2 - pos1;
    float3 edge2 = pos3 - pos1;
    float3 h = cross(rayD, edge2);
    float a = dot(edge1, h);
    if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
    float f = 1 / a;
    float3 s = rayO - pos1;
    float u = f * dot(s, h);
    if (u < 0 || u > 1) return;
    float3 q = cross(s, edge1);
    float v = f * dot(rayD, q);
    if (v < 0 || u + v > 1) return;
    float t = f * dot(edge2, q);

    if (t > 0.0001f && t < rayT) {
        rayT = t;   
        rayObjIdx = objIdx;
    }
}*/

__kernel void Extend(__global int* rayCounter, __global int* pixelIdxs, __global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs,   // Primary Rays
int quads_size, int spheres_size, int cubes_size, int planes_size, int triangles_size,
__global __read_only float16* quadMatrices, __global __read_only float* quadSizes, __global __read_only float4* sphereInfos, __global __read_only float4* primNorms, __global __read_only float4* triangleInfos)
{
    int threadId = get_global_id(0);

    if(threadId >= *rayCounter) 
    {
        return;
    }

    int rayPixelIdx = pixelIdxs[threadId];

    float rayT = INFINITY;
    int rayObjIdx = -1;

    float3 rayO = origins[rayPixelIdx].xyz;
    float3 rayD = directions[rayPixelIdx].xyz;

    int currentObjIdx = -1;
    for(int i = 0; i < quads_size; i++)
    {
        currentObjIdx += 1;
        //TODO
        float3 O = MultiplyMatrix((float4)(rayO, 1), quadMatrices[i]);
        float3 D = MultiplyMatrix((float4)(rayD, 0), quadMatrices[i]);
        float t = O.y / -D.y;

        if (t < rayT && t > 0)
        {
            float3 I = O + t * D;
            if (I.x > -quadSizes[i] && I.x < quadSizes[i] && I.z > -quadSizes[i] && I.z < quadSizes[i])
            {
                rayT = t; 
                rayObjIdx = currentObjIdx;
            }
        }
        
        //QuadIntersect();
    }

    for (int i = 0; i < spheres_size; i++)
    {
        currentObjIdx += 1;
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
            rayT = t; 
            rayObjIdx = currentObjIdx;
            continue;
        }
        t = d - b;
        if (t < rayT && t > 0)
        {
            rayT = t;
            rayObjIdx = currentObjIdx;
            continue;
        } 

        //SphereIntersect(currentObjIdx, (float3)(sphereInfos[i].x, sphereInfos[i].y, sphereInfos[i].z), sphereInfos[i].w);
    }

    for (int i = 0; i < cubes_size; i++)
    {
        //TODO
        currentObjIdx += 1;
        //CubeIntersect();
    }

    for (int i = 0; i < planes_size; i++)
    {
        currentObjIdx += 1;

        float3 N = primNorms[currentObjIdx].xyz;
        float d = primNorms[currentObjIdx].w;
        float t = -(dot(rayO, N) + d) / (dot(rayD, N));

        if (t < rayT && t > 0) 
        {   
            rayT = t;
            rayObjIdx = currentObjIdx;
        }
        /*
        PlaneIntersect(currentObjIdx, (float3)(primNorms[currentObjIdx].x, primNorms[currentObjIdx].y, primNorms[currentObjIdx].z), primNorms[currentObjIdx].w);
        currentObjIdx += 1;
        */
    }

    for (int i = 0; i < triangles_size; i++)
    {
        currentObjIdx += 1;
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
            rayT = t;   
            rayObjIdx = currentObjIdx;
        }
        /*
        triangleIntersect(currentObjIdx, 
           triangleInfos1[i].xyz,
           triangleInfos2[i].xyz,
           triangleInfos3[i].xyz);
        */
    }

    distances[rayPixelIdx] = rayT;
    primIdxs[rayPixelIdx] = rayObjIdx;
}