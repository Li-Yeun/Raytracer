typedef struct 
{
    int type; // 0 = sphere, 1 = triangle, 2 = quad, 3 = plane
    float3 info1; // shpere: pos,   triangle: pos1,     quad: _,    plane: N
    float3 info2; // shpere: _,     triangle: pos2,     quad: _,    plane: _
    float3 info3; // shpere: _,     triangle: pos3,     quad: _,    plane: _
    float infofloat; // shpere: r2,     triangle: _,     quad: _,    plane: d

} Primitive ;

// TODO: check this function (it's generated), pass invT from quad to Extend kernel (how to we convert matrix to float pointer, i assume just 16 floats) 
// Multiply a 4x4 matrix by a 4x1 vector
float* MultiplyMatrix(float4 a, float* matrix)  
{
    float4 result = (float4)(0, 0, 0, 0);
    for (int i = 0; i < 4; i++)
    {
        result.x += a[i] * matrix[i];
        result.y += a[i] * matrix[i + 4];
        result.z += a[i] * matrix[i + 8];
        result.w += a[i] * matrix[i + 12];
    }
    return (float*)result;
}


// TODO: add buffer for invT matrix for quads (unless we can add them to the primitive struct)
__kernel void Extend(__global float3* origins, __global float3* directions, __global float* distances, __global int* primIdxs, __global Primitive* primitives, int primCount)
{   
    int i = get_global_id(0);

    for (int primId = 0; primId < primCount; primId++)
    {
        int type = primitives[primId].type;
        // type seems to be wrong?? looks like it's only 0 or 2
        // printf("%d", type);

        if (type == 0) // shpere
        {
            float3 oc = origins[i] - primitives[primId].info1;
            float b = dot(oc, directions[i]);
            float c = dot(oc, oc) - primitives[primId].infofloat;
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
            const float3 edge1 = primitives[primId].info2 - primitives[primId].info1;
            const float3 edge2 = primitives[primId].info3 - primitives[primId].info1;
            const float3 h = cross(directions[i], edge2);
            const float a = dot(edge1, h);
            if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
            const float f = 1 / a;
            const float3 s = origins[i] - primitives[primId].info1;
            const float u = f * dot(s, h);
            if (u < 0 || u > 1) return;
            const float3 q = cross(s, edge1);
            const float v = f * dot(directions[i], q);
            if (v < 0 || u + v > 1) return;
            const float t = f * dot(edge2, q);
            if (t > 0.0001f && t < distances[i]) {
                distances[i] = t;
                primIdxs[i] = primId;
            }
        }

        if (type == 2) // quad
        {
            //TODO: check if functions correctly

            // const float3 O = TransformPosition(ray.O, invT);
            const float4 O4 = (float4)(ray.O, 1);
            const float3 O = (float3)(MultiplyMatrix(O4, invT));

            // const float3 D = TransformVector(ray.D, invT);
            const float4 D4 = (float4)(ray.D, 0);
            const float3 D = (float3)(MultiplyMatrix(D4, invT));

            const float t = O.y / -D.y;
            if (t < ray.t && t > 0)
            {
                float3 I = O + t * D;
                if (I.x > -size && I.x < size && I.z > -size && I.z < size)
                    ray.t = t, ray.objIdx = objIdx;
            }
        }

        if (type == 3) // plane
        {
            float t = -(dot(origins[i], primitives[primId].info1) + primitives[primId].infofloat) / (dot(directions[i], primitives[primId].info1));
            if (t < distances[i] && t > 0) distances[i] = t, primIdxs[i] = primId;
        }
    }

    
}

