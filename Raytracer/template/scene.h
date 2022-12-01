#pragma once
#include "material.h"

// -----------------------------------------------------------
// scene.h
// Simple test scene for ray tracing experiments. Goals:
// - Super-fast scene intersection
// - Easy interface: scene.FindNearest / IsOccluded
// - With normals and albedo: GetNormal / GetAlbedo
// - Area light source (animated), for light transport
// - Primitives can be hit from inside - for dielectrics
// - Can be extended with other primitives and/or a BVH
// - Optionally animated - for temporal experiments
// - Not everything is axis aligned - for cache experiments
// - Can be evaluated at arbitrary time - for motion blur
// - Has some high-frequency details - for filtering
// Some speed tricks that severely affect maintainability
// are enclosed in #ifdef SPEEDTRIX / #endif. Mind these
// if you plan to alter the scene in any way.
// -----------------------------------------------------------

#define SPEEDTRIX

#define PLANE_X(o,i) {if((t=-(ray.O.x+o)*ray.rD.x)<ray.t)ray.t=t,ray.objIdx=i;}
#define PLANE_Y(o,i) {if((t=-(ray.O.y+o)*ray.rD.y)<ray.t)ray.t=t,ray.objIdx=i;}
#define PLANE_Z(o,i) {if((t=-(ray.O.z+o)*ray.rD.z)<ray.t)ray.t=t,ray.objIdx=i;}

namespace Tmpl8 {

__declspec(align(64)) class Ray
{
public:
    Ray() = default;
    Ray( float3 origin, float3 direction, float distance = 1e34f )
    {
        O = origin, D = direction, t = distance;
        // calculate reciprocal ray direction for triangles and AABBs
        rD = float3( 1 / D.x, 1 / D.y, 1 / D.z );
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
    D = dot(N, pos1);
  }

  void Intersect(Ray& ray) const
  {
    // based on https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution

    // No intersection if ray and plane are parallel
    float epsilon = 0.001f;
    if ( fabs(dot(N, ray.D)) < epsilon ) return;

    float t = -(dot(ray.O, N) + D) / (dot(ray.D, N));
    if (t < ray.t && t < 0) return; // Triangle is behind ray, so the ray should not see it.

    // Compute P
    float3 P = ray.O + t * ray.D;

    // Compute edges
    float3 crossProduct, C;
    // edge 1
    float3 edge1 = pos2 - pos1;
    C = P - pos1;
    crossProduct = cross(edge1, C);
    if (dot(N, crossProduct) <= 0) return; // P is outside triangle

    //edge 2
    float3 edge2 = pos3 - pos2;
    C = P - pos2;
    crossProduct = cross(edge2, C);
    if (dot(N, crossProduct) <= 0) return; // P is outside triangle

    //edge 3
    float3 edge3 = pos1 - pos3;
    C = P - pos3;
    crossProduct = cross(edge3, C);
    if (dot(N, crossProduct) <= 0) return; // P is outside triangle

    // If not returned by now, P is inside triangle
    ray.t = t, ray.objIdx = objIdx;
  }

  float3 GetNormal(const float3 intersection) const
  {
    float3 A = pos1 - pos3;
    float3 B = pos2 - pos3;
    return cross(A, B);
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
    Sphere(int idx, float3 p, float r, Material mat ) :
        pos( p ), r2( r* r ), invr( 1 / r ), objIdx( idx ), material (mat) {}
    void Intersect( Ray& ray ) const
    {
        float3 oc = ray.O - this->pos;
        float b = dot( oc, ray.D );
        float c = dot( oc, oc ) - this->r2;
        float t, d = b * b - c;
        if (d <= 0) return;
        d = sqrtf( d ), t = -b - d;
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
    float3 GetNormal( const float3 I ) const
    {
        return (I - this->pos) * invr;
    }
    float3 GetAlbedo( const float3 I ) const
    {
        return float3( 0.93f );
    }

    float3 pos = 0;
    float r2 = 0, invr = 0;
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
    Plane( int idx, float3 normal, float dist, Material mat ) : N( normal ), d( dist ), objIdx( idx ), material( mat ) {}
    void Intersect( Ray& ray ) const
    {
        float t = -(dot( ray.O, this->N ) + this->d) / (dot( ray.D, this->N ));
        if (t < ray.t && t > 0) ray.t = t, ray.objIdx = objIdx;
    }
    float3 GetNormal( const float3 I ) const
    {
        return N;
    }
    float3 GetAlbedo( const float3 I ) const
    {
        if (N.y == 1)
        {
            // floor albedo: checkerboard
            int ix = (int)(I.x * 2 + 96.01f);
            int iz = (int)(I.z * 2 + 96.01f);
            // add deliberate aliasing to two tile
            if (ix == 98 && iz == 98) ix = (int)(I.x * 32.01f), iz = (int)(I.z * 32.01f);
            if (ix == 94 && iz == 98) ix = (int)(I.x * 64.01f), iz = (int)(I.z * 64.01f);
            return float3( ((ix + iz) & 1) ? 1 : 0.3f );
        }
        else if (N.z == -1)
        {
            // back wall: logo
            static Surface logo( "assets/logo.png" );
            int ix = (int)((I.x + 4) * (128.0f / 8));
            int iy = (int)((2 - I.y) * (64.0f / 3));
            uint p = logo.pixels[(ix & 127) + (iy & 63) * 128];
            uint3 i3( (p >> 16) & 255, (p >> 8) & 255, p & 255 );
            return float3( i3 ) * (1.0f / 255.0f);
        }
        return float3( 0.93f );
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
    Cube( int idx, float3 pos, float3 size, Material mat, mat4 transform = mat4::Identity()) : material(mat)
    {
        objIdx = idx;
        b[0] = pos - 0.5f * size, b[1] = pos + 0.5f * size;
        M = transform, invM = transform.FastInvertedTransformNoScale();

    }
    void Intersect( Ray& ray ) const
    {
        // 'rotate' the cube by transforming the ray into object space
        // using the inverse of the cube transform.
        float3 O = TransformPosition( ray.O, invM );
        float3 D = TransformVector( ray.D, invM );
        float rDx = 1 / D.x, rDy = 1 / D.y, rDz = 1 / D.z;
        int signx = D.x < 0, signy = D.y < 0, signz = D.z < 0;
        float tmin = (b[signx].x - O.x) * rDx;
        float tmax = (b[1 - signx].x - O.x) * rDx;
        float tymin = (b[signy].y - O.y) * rDy;
        float tymax = (b[1 - signy].y - O.y) * rDy;
        if (tmin > tymax || tymin > tmax) return;
        tmin = max( tmin, tymin ), tmax = min( tmax, tymax );
        float tzmin = (b[signz].z - O.z) * rDz;
        float tzmax = (b[1 - signz].z - O.z) * rDz;
        if (tmin > tzmax || tzmin > tmax) return;
        tmin = max( tmin, tzmin ), tmax = min( tmax, tzmax );
        if (tmin > 0)
        {
            if (tmin < ray.t) ray.t = tmin, ray.objIdx = objIdx;
        }
        else if (tmax > 0)
        {
            if (tmax < ray.t) ray.t = tmax, ray.objIdx = objIdx;
        }
    }
    float3 GetNormal( const float3 I ) const
    {
        // transform intersection point to object space
        float3 objI = TransformPosition( I, invM );
        // determine normal in object space
        float3 N = float3( -1, 0, 0 );
        float d0 = fabs( objI.x - b[0].x ), d1 = fabs( objI.x - b[1].x );
        float d2 = fabs( objI.y - b[0].y ), d3 = fabs( objI.y - b[1].y );
        float d4 = fabs( objI.z - b[0].z ), d5 = fabs( objI.z - b[1].z );
        float minDist = d0;
        if (d1 < minDist) minDist = d1, N.x = 1;
        if (d2 < minDist) minDist = d2, N = float3( 0, -1, 0 );
        if (d3 < minDist) minDist = d3, N = float3( 0, 1, 0 );
        if (d4 < minDist) minDist = d4, N = float3( 0, 0, -1 );
        if (d5 < minDist) minDist = d5, N = float3( 0, 0, 1 );
        // return normal in world space
        return TransformVector( N, M );
    }
    float3 GetAlbedo( const float3 I ) const
    {
        return float3( 1, 1, 1 );
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
    Quad( int idx, float s, Material mat, mat4 transform = mat4::Identity()): material( mat )
    {
        objIdx = idx;
        size = s * 0.5f;
        T = transform, invT = transform.FastInvertedTransformNoScale();
    }
    void Intersect( Ray& ray ) const
    {
        const float3 O = TransformPosition( ray.O, invT );
        const float3 D = TransformVector( ray.D, invT );
        const float t = O.y / -D.y;
        if (t < ray.t && t > 0)
        {
            float3 I = O + t * D;
            if (I.x > -size && I.x < size && I.z > -size && I.z < size)
                ray.t = t, ray.objIdx = objIdx;
        }
    }
    float3 GetNormal( const float3 I ) const
    {
        // TransformVector( float3( 0, -1, 0 ), T )
        return float3( -T.cell[1], -T.cell[5], -T.cell[9] );
    }
    float3 GetAlbedo( const float3 I ) const
    {
        return float3( 10 );
    }
    float size;
    mat4 T, invT;
    int objIdx = -1;
    Material material;
};

// -----------------------------------------------------------
// Scene class
// We intersect this. The query is internally forwarded to the
// list of primitives, so that the nearest hit can be returned.
// For this hit (distance, obj id), we can query the normal and
// albedo.
// -----------------------------------------------------------
class Scene
{
public:
    Scene()
    {
        LoadObjects();
        def_mat = Material(Material::MaterialType::DIFFUSE, float3(1), 0);
        mirror_mat = Material(Material::MaterialType::MIRROR, float3(1), 0);
        glass_mat = Material(Material::MaterialType::GLASS, float3(1), 0, 1.52, float3(8.0f, 2.0f, 0.1f));
        light_mat = Material(Material::MaterialType::LIGHT, float3(1), NULL, NULL, float3(NULL), float3(10.0f));

        // we store all primitives in one continuous buffer
        quad = Quad( 0, 1, light_mat );																	// 0: light source
        sphere = Sphere( 1, float3( 0 ), 0.5f, glass_mat);												// 1: bouncing ball
        sphere2 = Sphere( 2, float3( 0, 2.5f, -3.07f ), 8, def_mat);									// 2: rounded corners
        cube = Cube( 3, float3( 0 ), float3( 1.15f ), def_mat);											// 3: cube
        plane[0] = Plane( 4, float3( 1, 0, 0 ), 3, mirror_mat);											// 4: left wall
        plane[1] = Plane( 5, float3( -1, 0, 0 ), 2.99f, def_mat);										// 5: right wall
        plane[2] = Plane( 6, float3( 0, 1, 0 ), 1, def_mat);											// 6: floor
        plane[3] = Plane( 7, float3( 0, -1, 0 ), 2, def_mat);											// 7: ceiling
        plane[4] = Plane( 8, float3( 0, 0, 1 ), 3, def_mat);											// 8: front wall
        plane[5] = Plane( 9, float3( 0, 0, -1 ), 3.99f, def_mat);										// 9: back wall
        triangle = Triangle(10, float3(1, 0, 1), float3(0, 1, -1), float3(0, 0, 1), def_mat);			//10: triangle
        SetTime( 0 );
        // Note: once we have triangle support we should get rid of the class
        // hierarchy: virtuals reduce performance somewhat.
    }
    void LoadObjects()
    {
        std::string inputfile = "assets/pyramid.obj";
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./assets"; // Path to material files

        tinyobj::ObjReader objectReader;

        if (!objectReader.ParseFromFile(inputfile, reader_config)) {
            if (!objectReader.Error().empty()) {
                std::cerr << "TinyObjReader: " << objectReader.Error();
            }
            exit(1);
        }

        if (!objectReader.Warning().empty()) {
            std::cout << "TinyObjReader: " << objectReader.Warning();
        }

        //auto& attrib = objectReader.GetAttrib();
        auto& vertices = objectReader.GetAttrib().GetVertices();
        auto& shapes = objectReader.GetShapes();

        // foreach shape
        for (size_t s = 0; s < shapes.size(); s++)
        {
            for (size_t vf = 0; vf < shapes[s].mesh.indices.size(); vf++)
            {
                const float vx = vertices[3 * size_t(shapes[s].mesh.indices[vf].vertex_index) + 0]; 
                const float vy = vertices[3 * size_t(shapes[s].mesh.indices[vf].vertex_index) + 1]; 
                const float vz = vertices[3 * size_t(shapes[s].mesh.indices[vf].vertex_index) + 2]; 

                std::cout << vx << vy << vz << std::endl;

                // create triangle
                triangles.push_back(Triangle(10 + vf, vx, vy, vz, def_mat));
            }
            
        }
    }
    void SetTime( float t )
    {
        // default time for the scene is simply 0. Updating/ the time per frame
        // enables animation. Updating it per ray can be used for motion blur.
        animTime = 0;

        if (isDynamic) animTime = t;
        // light source animation: swing
        mat4 M1base = mat4::Translate( float3( 0, 2.6f, 2 ) );
        mat4 M1 = M1base * mat4::RotateZ( sinf( animTime * 0.6f ) * 0.1f ) * mat4::Translate( float3( 0, -0.9, 0 ) );
        quad.T = M1, quad.invT = M1.FastInvertedTransformNoScale();
        // cube animation: spin
        mat4 M2base = mat4::RotateX( PI / 4 ) * mat4::RotateZ( PI / 4 );
        mat4 M2 = mat4::Translate( float3( 1.4f, 0, 2 ) ) * mat4::RotateY( animTime * 0.5f ) * M2base;
        cube.M = M2, cube.invM = M2.FastInvertedTransformNoScale();
        // sphere animation: bounce
        float tm = 1 - sqrf( fmodf( animTime, 2.0f ) - 1 );
        sphere.pos = float3( -1.4f, -0.5f + tm, 2 );
        
    }
    float3 GetLightPos() const
    {
        // light point position is the middle of the swinging quad
        float3 corner1 = TransformPosition( float3( -0.5f, 0, -0.5f ), quad.T );
        float3 corner2 = TransformPosition( float3( 0.5f, 0, 0.5f ), quad.T );
        return (corner1 + corner2) * 0.5f - float3( 0, 0.01f, 0 );
    }
    float3 GetLightColor() const
    {
        return float3( 24, 24, 22 );
    }
    void FindNearest( Ray& ray ) const
    {
        // room walls - ugly shortcut for more speed
        float t;
        if (ray.D.x < 0) PLANE_X( 3, 4 ) else PLANE_X( -2.99f, 5 );
        if (ray.D.y < 0) PLANE_Y( 1, 6 ) else PLANE_Y( -2, 7 );
        if (ray.D.z < 0) PLANE_Z( 3, 8 ) else PLANE_Z( -3.99f, 9 );
        quad.Intersect( ray );
        sphere.Intersect( ray );
        sphere2.Intersect( ray );
        cube.Intersect( ray );
        triangle.Intersect( ray );
        for (size_t triangleIndex = 0; triangleIndex < triangles.size(); triangleIndex++)
        {
            triangles[triangleIndex].Intersect(ray);
        }

    }
    bool IsOccluded( Ray& ray ) const
    {
        float rayLength = ray.t;
        // skip planes: it is not possible for the walls to occlude anything
        quad.Intersect( ray );
        sphere.Intersect( ray );
        sphere2.Intersect( ray );
        cube.Intersect( ray );
        triangle.Intersect(ray);
        for (size_t triangleIndex = 0; triangleIndex < triangles.size(); triangleIndex++)
        {
            triangles[triangleIndex].Intersect(ray);
        }
        return ray.t < rayLength;
        // technically this is wasteful:
        // - we potentially search beyond rayLength
        // - we store objIdx and t when we just need a yes/no
        // - we don't 'early out' after the first occlusion
    }
    float3 GetNormal( int objIdx, float3 I, float3 wo ) const
    {
        // we get the normal after finding the nearest intersection:
        // this way we prevent calculating it multiple times.
        if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
        float3 N;
        if (objIdx == 0) N = quad.GetNormal(I);
        else if (objIdx == 1) N = sphere.GetNormal(I);
        else if (objIdx == 2) N = sphere2.GetNormal(I);
        else if (objIdx == 3) N = cube.GetNormal(I);
        else if (objIdx == 10) N = triangle.GetNormal(I);
        else if (objIdx >= 11) N = triangles[objIdx - 11].GetNormal(I);
        else
        {
            // faster to handle the 6 planes without a call to GetNormal
            N = float3( 0 );
            N[(objIdx - 4) / 2] = 1 - 2 * (float)(objIdx & 1);
        }
        if (dot( N, wo ) > 0) N = -N; // hit backside / inside
        return N;
    }
    float3 GetAlbedo( int objIdx, float3 I ) const
    {
        if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
        if (objIdx == 0) return quad.GetAlbedo( I );
        if (objIdx == 1) return sphere.GetAlbedo( I );
        if (objIdx == 2) return sphere2.GetAlbedo( I );
        if (objIdx == 3) return cube.GetAlbedo( I );
        if (objIdx == 10)return triangle.GetAlbedo(I);
        if (objIdx >= 11)return triangles[objIdx - 11].GetAlbedo(I);
        return plane[objIdx - 4].GetAlbedo( I );
        // once we have triangle support, we should pass objIdx and the bary-
        // centric coordinates of the hit, instead of the intersection location.
    }

    Material GetMaterial(int objIdx) const
    {
        if (objIdx == -1) return Material(); // or perhaps we should just crash
        if (objIdx == 0) return quad.material;
        if (objIdx == 1) return sphere.material;
        if (objIdx == 2) return sphere2.material;
        if (objIdx == 3) return cube.material;
        if (objIdx == 10)return triangle.material;
        if (objIdx >= 11)return triangles[objIdx - 11].material;
        return plane[objIdx - 4].material;
        // once we have triangle support, we should pass objIdx and the bary-
        // centric coordinates of the hit, instead of the intersection location.
    }
    float GetReflectivity( int objIdx, float3 I ) const
    {
        if (objIdx == 1 /* ball */) return 1;
        if (objIdx == 6 /* floor */) return 0.3f;
        return 0;
    }
    float GetRefractivity( int objIdx, float3 I ) const
    {
        return objIdx == 3 ? 1.0f : 0.0f;
    }

    float3 DirectIllumination(float3 intersection, float3 normal)
    {

        // Check if ray hits other objects
        float3 light_postion = GetLightPos();
        float3 light_color = float3(1.0f);
        float light_intensity = 10.0f * PI;

        float3 shadowRayDirection = light_postion - intersection;
        float3 shadowRayDirectionNorm = normalize(shadowRayDirection);
        float epsilonOffset = 0.001f;
        Ray shadowRay = Ray(intersection + shadowRayDirectionNorm * epsilonOffset, shadowRayDirectionNorm);
        float shadowRayMagnitude = magnitude(shadowRayDirection);
        shadowRay.t = shadowRayMagnitude - epsilonOffset*2 ;
        FindNearest(shadowRay);
        if (shadowRay.objIdx == -1)
        {
            float distanceEnergy = 1 / sqrf(shadowRay.t);
            float angularEnergy = max(dot(normal, shadowRayDirectionNorm), 0.0f);
            return light_color * light_intensity * distanceEnergy * angularEnergy;
        }
        else
        {
            return float3(0);
        }
    }

    float3 DiffuseReflection(float3 normal)
    {
      float3 randomDirection = float3(Rand(2.0f) - 1.0f, Rand(2.0f) - 1.0f, Rand(2.0f) - 1.0f);

      while (magnitude(randomDirection) > 1.0f)
      {
          randomDirection = float3(Rand(2.0f) - 1.0f, Rand(2.0f) - 1.0f, Rand(2.0f) - 1.0f);
      }

      randomDirection = normalize(randomDirection);
      if (dot(normal, randomDirection) < 0) randomDirection = -randomDirection;

      return randomDirection;
    }

    void SetIsDynamicScene(bool _isDynamic) { isDynamic = _isDynamic; }

    __declspec(align(64)) // start a new cacheline here
    float animTime = 0;
    bool isDynamic = false;
    Quad quad;
    Sphere sphere;
    Sphere sphere2;
    Cube cube;
    Plane plane[6];
    Triangle triangle;
    std::vector<Triangle> triangles;
    Material def_mat, mirror_mat, glass_mat, light_mat;
};

}
