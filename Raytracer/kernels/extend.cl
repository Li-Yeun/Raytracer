// int type;        // shpere: 0,       triangle: 1,        quad: 2,        plane: 3
// float3 info1;    // shpere: pos,     triangle: pos1,     quad: _,        plane: N
// float3 info2;    // shpere: _,       triangle: pos2,     quad: _,        plane: _
// float3 info3;    // shpere: _,       triangle: pos3,     quad: _,        plane: _
// float infofloat; // shpere: r2,      triangle: _,        quad: size,     plane: d
// mat4 infoMatrix; // shpere: _,       triangle: _,        quad: invT,     plane: _


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

__kernel void Extend(__global float4* origins, __global float4* directions, __global float* distances, __global int* primIdxs, // ray data
                    int primCount, __global int* primTypes, //Primitive metadata
                    __global float4* primInfo1, __global float4* primInfo2, __global float4* primInfo3, // Primitive float3 data
                    __global float* primInfofloat, __global float16* primInfoMatrix) // primitive misc data
                    
                    //__global float3* aabbMin, __global float3* aabbMax, __global uint* leftFirst, __global uint* primitiveCount) // BVH data 
{   
    int i = get_global_id(0);

    float3 direction = (float3)(directions[i].x, directions[i].y, directions[i].z);
    float3 origin = (float3)(origins[i].x, origins[i].y, origins[i].z);


    for (int primId = 0; primId < primCount; primId++)
    {        
        int type = primTypes[primId];
        float3 info1 = (float3)(primInfo1[primId].x, primInfo1[primId].y, primInfo1[primId].z);
        float3 info2 = (float3)(primInfo2[primId].x, primInfo2[primId].y, primInfo2[primId].z);
        float3 info3 = (float3)(primInfo3[primId].x, primInfo3[primId].y, primInfo3[primId].z);

        // if (i == 0)
        // {
        //     // if(type == 2) printf("matrix: %f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n\n\n", primInfoMatrix[primId][0], primInfoMatrix[primId][1], primInfoMatrix[primId][2], primInfoMatrix[primId][3], primInfoMatrix[primId][4], primInfoMatrix[primId][5], primInfoMatrix[primId][6], primInfoMatrix[primId][7], primInfoMatrix[primId][8], primInfoMatrix[primId][9], primInfoMatrix[primId][10], primInfoMatrix[primId][11], primInfoMatrix[primId][12], primInfoMatrix[primId][13], primInfoMatrix[primId][14], primInfoMatrix[primId][15]);
        //     printf("type: %d, info1: %f, %f, %f, info2: %f, %f, %f, info3: %f, %f, %f, infofloat: %f\n", type, primInfo1[primId].x, primInfo1[primId].y, primInfo1[primId].z, primInfo2[primId].x, primInfo2[primId].y, primInfo2[primId].z, primInfo3[primId].x, primInfo3[primId].y, primInfo3[primId].z, primInfofloat[primId]);
        // }
        // if (i == 1)
        // {
        //     printf(" primCount: %d\n\n", primCount);
        // }

        if (type == 0) // shpere
        {
            float3 oc = origin - info1;
            float b = dot(oc, direction);
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

        if (type == 1) // triangle
        {
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

        if (type == 2) // quad
        {
            const float size = primInfofloat[primId];

            // const float3 O = TransformPosition(origins[i], primInfoMatrix[primId]);
            const float4 O4 = (float4)(origin, 1);
            const float4 Otransformed = MultiplyMatrix(O4, primInfoMatrix[primId]);
            const float3 O = (float3)(Otransformed.x, Otransformed.y, Otransformed.z);

            // const float3 D = TransformVector(ray.D, invT);
            const float4 D4 = (float4)(direction, 0);
            const float4 Dtransformed = MultiplyMatrix(D4, primInfoMatrix[primId]);
            const float3 D = (float3)(Dtransformed.x, Dtransformed.y, Dtransformed.z);

            const float t = O.y / -D.y;
            if (t < distances[i] && t > 0)
            {
                float3 I = O + t * D;
                if (I.x > -size && I.x < size && I.z > -size && I.z < size)
                    distances[i] = t, primIdxs[i] = primId;
            }
        }

        if (type == 3) // plane
        {
            float t = -(dot(origin, info1) + primInfofloat[primId]) / (dot(direction, info1));
            if (t < distances[i] && t > 0) distances[i] = t, primIdxs[i] = primId;
        }
    }

    
}

