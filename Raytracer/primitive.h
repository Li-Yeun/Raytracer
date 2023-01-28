#pragma once
namespace Tmpl8 {
    __declspec(align(64)) class Ray
    {
    public:
        Ray() = default;
        Ray(float3 origin, float3 direction, float distance = 1e34f)
        {
            O = origin, D = direction, t = distance;
            // calculate reciprocal ray direction for triangles and AABBs
            rD = float3(1 / D.x, 1 / D.y, 1 / D.z);
#ifdef SPEEDTRIX
            d0 = d1 = d2 = 0;
#endif
        }
        float3 IntersectionPoint() { return O + t * D; }
        // ray data
#ifndef SPEEDTRIX
        float3 O, D, rD;
#else
        union { struct { float3 O; float d0; }; __m128 O4; };
        union { struct { float3 D; float d1; }; __m128 D4; };
        union { struct { float3 rD; float d2; }; __m128 rD4; };
#endif
        float t = 1e34f;
        int objIdx = -1;
        bool inside = false; // true when in medium
    };

    // -----------------------------------------------------------
    // Triangle primitive
    // Basic triangle
    // -----------------------------------------------------------
    class Triangle
    {
    public:
        Triangle() = default;
        Triangle(int idx, float3 p1, float3 p2, float3 p3, Material mat) :
            pos1(p1),
            pos2(p2),
            pos3(p3),
            objIdx(idx),
            N(GetNormal(float3(0))),
            material(mat)
        {
            D = -dot(N, pos1);
            centroid = (pos1 + pos2 + pos3) * 0.33333f;
        }

        void Intersect(Ray& ray) const
        {
            // based on https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution

            // No intersection if ray and plane are parallel
            const float3 edge1 = pos2 - pos1;
            const float3 edge2 = pos3 - pos1;
            const float3 h = cross(ray.D, edge2);
            const float a = dot(edge1, h);
            if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
            const float f = 1 / a;
            const float3 s = ray.O - pos1;
            const float u = f * dot(s, h);
            if (u < 0 || u > 1) return;
            const float3 q = cross(s, edge1);
            const float v = f * dot(ray.D, q);
            if (v < 0 || u + v > 1) return;
            const float t = f * dot(edge2, q);
            if (t > 0.0001f && t < ray.t) {
                ray.t = t;
                ray.objIdx = objIdx;
            }
        }

        float3 GetNormal(const float3 intersection) const
        {
            float3 A = pos2 - pos1;
            float3 B = pos3 - pos1;
            return normalize(cross(A, B));
        }
        float3 GetAlbedo(const float3 intersection) const
        {
            return float3(1);
        }

        float3 pos1 = 0;
        float3 pos2 = 0;
        float3 pos3 = 0;
        int objIdx = -1;
        float3 N;
        float D;
        Material material;
        float3 centroid = float3(0);

    };

    // -----------------------------------------------------------
    // Sphere primitive
    // Basic sphere, with explicit support for rays that start
    // inside it. Good candidate for a dielectric material.
    // -----------------------------------------------------------
    class Sphere
    {
    public:
        Sphere() = default;
        Sphere(int idx, float3 p, float r, Material mat) :
            pos(p), r(r), r2(r* r), invr(1 / r), objIdx(idx), material(mat) {
        }
        void Intersect(Ray& ray) const
        {
            float3 oc = ray.O - this->pos;
            float b = dot(oc, ray.D);
            float c = dot(oc, oc) - this->r2;
            float t, d = b * b - c;
            if (d <= 0) return;
            d = sqrtf(d), t = -b - d;
            if (t < ray.t && t > 0)
            {
                ray.t = t, ray.objIdx = objIdx;
                return;
            }
            t = d - b;
            if (t < ray.t && t > 0)
            {
                ray.t = t, ray.objIdx = objIdx;
                return;
            }
        }
        float3 GetNormal(const float3 I) const
        {
            return (I - this->pos) * invr;
        }
        float3 GetAlbedo(const float3 I) const
        {
            return float3(0.93f);
        }

        float3 pos = 0;
        float r = 0, r2 = 0, invr = 0;
        int objIdx = -1;
        Material material;
        
    };

    // -----------------------------------------------------------
    // Plane primitive
    // Basic infinite plane, defined by a normal and a distance
    // from the origin (in the direction of the normal).
    // -----------------------------------------------------------
    class Plane
    {
    public:
        Plane() = default;
        Plane(int idx, float3 normal, float dist, Material mat) : N(normal), d(dist), objIdx(idx), material(mat) {}
        void Intersect(Ray& ray) const
        {
            float t = -(dot(ray.O, this->N) + this->d) / (dot(ray.D, this->N));
            if (t < ray.t && t > 0) ray.t = t, ray.objIdx = objIdx;
        }
        float3 GetNormal(const float3 I) const
        {
            return N;
        }
        float3 GetAlbedo(const float3 I) const
        {
            if (N.y == 1)
            {
                // floor albedo: checkerboard
                int ix = (int)(I.x * 2 + 96.01f);
                int iz = (int)(I.z * 2 + 96.01f);
                // add deliberate aliasing to two tile
                if (ix == 98 && iz == 98) ix = (int)(I.x * 32.01f), iz = (int)(I.z * 32.01f);
                if (ix == 94 && iz == 98) ix = (int)(I.x * 64.01f), iz = (int)(I.z * 64.01f);
                return float3(((ix + iz) & 1) ? 1 : 0.3f);
            }
            else if (N.z == -1)
            {
                // back wall: logo
                static Surface logo("assets/logo.png");
                int ix = (int)((I.x + 4) * (128.0f / 8));
                int iy = (int)((2 - I.y) * (64.0f / 3));
                uint p = logo.pixels[(ix & 127) + (iy & 63) * 128];
                uint3 i3((p >> 16) & 255, (p >> 8) & 255, p & 255);
                return float3(i3) * (1.0f / 255.0f);
            }
            return float3(0.93f);
        }
        float3 N;
        float d;
        int objIdx = -1;
        Material material;
    };

    // -----------------------------------------------------------
    // Cube primitive
    // Oriented cube. Unsure if this will also work for rays that
    // start inside it; maybe not the best candidate for testing
    // dielectrics.
    // -----------------------------------------------------------
    class Cube
    {
    public:
        Cube() = default;
        Cube(int idx, float3 pos, float3 size, Material mat, mat4 transform = mat4::Identity()) : material(mat)
        {
            objIdx = idx;
            b[0] = pos - 0.5f * size, b[1] = pos + 0.5f * size;
            M = transform, invM = transform.FastInvertedTransformNoScale();

        }
        void Intersect(Ray& ray) const
        {
            // 'rotate' the cube by transforming the ray into object space
            // using the inverse of the cube transform.
            float3 O = TransformPosition(ray.O, invM);
            float3 D = TransformVector(ray.D, invM);
            float rDx = 1 / D.x, rDy = 1 / D.y, rDz = 1 / D.z;
            int signx = D.x < 0, signy = D.y < 0, signz = D.z < 0;
            float tmin = (b[signx].x - O.x) * rDx;
            float tmax = (b[1 - signx].x - O.x) * rDx;
            float tymin = (b[signy].y - O.y) * rDy;
            float tymax = (b[1 - signy].y - O.y) * rDy;
            if (tmin > tymax || tymin > tmax) return;
            tmin = max(tmin, tymin), tmax = min(tmax, tymax);
            float tzmin = (b[signz].z - O.z) * rDz;
            float tzmax = (b[1 - signz].z - O.z) * rDz;
            if (tmin > tzmax || tzmin > tmax) return;
            tmin = max(tmin, tzmin), tmax = min(tmax, tzmax);
            if (tmin > 0)
            {
                if (tmin < ray.t) ray.t = tmin, ray.objIdx = objIdx;
            }
            else if (tmax > 0)
            {
                if (tmax < ray.t) ray.t = tmax, ray.objIdx = objIdx;
            }
        }
        float3 GetNormal(const float3 I) const
        {
            // transform intersection point to object space
            float3 objI = TransformPosition(I, invM);
            // determine normal in object space
            float3 N = float3(-1, 0, 0);
            float d0 = fabs(objI.x - b[0].x), d1 = fabs(objI.x - b[1].x);
            float d2 = fabs(objI.y - b[0].y), d3 = fabs(objI.y - b[1].y);
            float d4 = fabs(objI.z - b[0].z), d5 = fabs(objI.z - b[1].z);
            float minDist = d0;
            if (d1 < minDist) minDist = d1, N.x = 1;
            if (d2 < minDist) minDist = d2, N = float3(0, -1, 0);
            if (d3 < minDist) minDist = d3, N = float3(0, 1, 0);
            if (d4 < minDist) minDist = d4, N = float3(0, 0, -1);
            if (d5 < minDist) minDist = d5, N = float3(0, 0, 1);
            // return normal in world space
            return TransformVector(N, M);
        }
        float3 GetAlbedo(const float3 I) const
        {
            return float3(1, 1, 1);
        }
        float3 b[2];
        mat4 M, invM;
        int objIdx = -1;
        Material material;
    };

    // -----------------------------------------------------------
    // Quad primitive
    // Oriented quad, intended to be used as a light source.
    // -----------------------------------------------------------
    class Quad
    {
    public:
        Quad() = default;
        Quad(int idx, float s, Material mat, mat4 transform = mat4::Identity()) : s(s), material(mat)
        {
            objIdx = idx;
            size = s * 0.5f;
            T = transform, invT = transform.FastInvertedTransformNoScale();
            N = GetNormal(float3(0));
            c1 = TransformPosition(float3(-size, 0, -size), T);
            c2 = TransformPosition(float3(size, 0, -size), T);
            c3 = TransformPosition(float3(size, 0, size), T);
            A = sqrf(s);
        }
        void Intersect(Ray& ray) const
        {
            const float3 O = TransformPosition(ray.O, invT);
            const float3 D = TransformVector(ray.D, invT);
            const float t = O.y / -D.y;
            if (t < ray.t && t > 0)
            {
                float3 I = O + t * D;
                if (I.x > -size && I.x < size && I.z > -size && I.z < size)
                    ray.t = t, ray.objIdx = objIdx;
            }
        }
        float3 GetNormal(const float3 I) const
        {
            // TransformVector( float3( 0, -1, 0 ), T )
            return float3(-T.cell[1], -T.cell[5], -T.cell[9]);
        }
        float3 GetAlbedo(const float3 I) const
        {
            return float3(10);
        }
        float size, s;
        mat4 T, invT;
        int objIdx = -1;
        Material material;

        // DELETE LATER
        float3 N; 

        float3 c1, c2, c3;
        float A;
    };
}