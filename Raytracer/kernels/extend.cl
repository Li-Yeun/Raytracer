// int primMats;    // DIFFUSE = 0, MIRROR = 1, GLASS = 2, SUBSTANCE = 1, LIGHT = 0 

/*
float rayT;
int rayObjIdx;
float3 rayO;
float3 rayD;
*/

/*
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
}*/

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

__kernel void Extend(__global int* rayCounter, __global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs,   // Primary Rays
int quads_size, int spheres_size, int cubes_size, int planes_size, int triangles_size,
__global float16* quadMatrices, __global float* quadSizes, __global float4* sphereInfos, __global float4* primNorms, __global float4* triangleInfos1, __global float4* triangleInfos2,__global float4* triangleInfos3)
{
    int threadId = get_global_id(0);

    if(threadId >= *rayCounter) 
    {
        return;
    }

    float rayT = distances[threadId];
    int rayObjIdx = primIdxs[threadId];

    float3 rayO = origins[threadId].xyz;
    float3 rayD = directions[threadId].xyz;

    int currentObjIdx = -1;
    for(int i = 0; i < quads_size; i++)
    {
        //TODO
        currentObjIdx += 1;
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
        {   rayT = t;
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
        float3 pos1 = triangleInfos1[i].xyz;
        float3 pos2 = triangleInfos2[i].xyz;
        float3 pos3 = triangleInfos3[i].xyz;

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

    distances[threadId] = rayT;
    primIdxs[threadId] = rayObjIdx;
}