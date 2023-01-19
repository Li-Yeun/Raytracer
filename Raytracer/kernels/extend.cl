typedef struct 
{
    int type; // 0 = sphere, 1 = triangle, 2 = quad, 3 = plane
    float3 info1; // shpere: pos,   triangle: pos1,     quad: _,    plane: N
    float3 info2; // shpere: _,     triangle: pos2,     quad: _,    plane: _
    float3 info3; // shpere: _,     triangle: pos3,     quad: _,    plane: _
    float infofloat; // shpere: r2,     triangle: _,     quad: _,    plane: d

} Primitive ;

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
            //TODO: implement

        }

        if (type == 3) // plane
        {
            float t = -(dot(origins[i], primitives[primId].info1) + primitives[primId].infofloat) / (dot(directions[i], primitives[primId].info1));
            if (t < distances[i] && t > 0) distances[i] = t, primIdxs[i] = primId;
        }
    }

    
}

