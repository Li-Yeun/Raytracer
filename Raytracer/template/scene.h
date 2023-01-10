#pragma once
#include "material.h"
#include "bvh.h"

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
        def_mat = Material(Material::MaterialType::DIFFUSE, float3(1), 0);
        red_mat = Material(Material::MaterialType::DIFFUSE, float3(0.93f, 0.21f, 0.21f), 0);
        cyan_mat = Material(Material::MaterialType::DIFFUSE, float3(0.11f, 0.95f, 0.91f), 0);
        mirror_mat = Material(Material::MaterialType::MIRROR, float3(1), 0);
        absorb_all_but_blue_mat = Material(Material::MaterialType::GLASS, float3(1), 0, 1.52, float3(8.0f, 2.0f, 1.0f));
        glass_mat = Material(Material::MaterialType::GLASS, float3(1), 0, 1.52, float3(0));//float3(8.0f, 2.0f, 0.1f));
        light_mat = Material(Material::MaterialType::LIGHT, float3(1), NULL, NULL, float3(NULL), float3(2.0f));


        quads_size = 1;
        quads = new Quad[quads_size] { Quad(id++, 3, light_mat) };                              // 0: light source
                  
        spheres_size = 2;
        spheres = new Sphere[spheres_size]{ 
            Sphere(id++, float3(-1.4f, -0.5f, 2), 0.5f, absorb_all_but_blue_mat),                             // 1: bouncing ball
            Sphere(id++,float3(0, 2.5f, -3.07f), 0.5f, def_mat) };						        // 2: rounded corners		

        cubes_size = 1;
        cubes = new Cube[cubes_size] { Cube(id++, float3(0), float3(0.75f), def_mat) };         // 3: cube
               
        planes_size = 6;                                                                                                
        planes = new Plane[planes_size]{
            Plane(id++, float3(1, 0, 0), 3, mirror_mat),                                        // 4: left wall
            Plane(id++, float3(-1, 0, 0), 2.99f, def_mat),										// 5: right wall
            Plane(id++, float3(0, 1, 0), 1, def_mat),											// 6: floor
            Plane(id++, float3(0, -1, 0), 2, def_mat),											// 7: ceiling
            Plane(id++, float3(0, 0, 1), 3, def_mat),											// 8: front wall
            Plane(id++, float3(0, 0, -1), 3.99f, def_mat)										// 9: back wall
        };

        std::cout << "Loading objects ..." << std::endl;

        LoadObject("assets/monkey.obj", glass_mat, float3(0, 0, 1.5));
        LoadObject("assets/monkey.obj",  cyan_mat, float3(1.5, 0, 1.5));
        LoadObject("assets/monkey.obj",  red_mat, float3(-1.5, 0, 1.5));

        bvh = new BVH(spheres, spheres_size, planes, planes_size, triangles, triangles_size);
        SetTime( 0 );
        // Note: once we have triangle support we should get rid of the class
        // hierarchy: virtuals reduce performance somewhat.

        std::cout << std::endl << triangles_size << " Triangles loaded" << std::endl << std::endl << std::endl;
    }
    void LoadObject(std::string inputfile, Material material, float3 transform = float3(0))
    {
        std::cout << "Loading: " << inputfile << std::endl;
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
            int size = shapes[s].mesh.indices.size() / 3;
            Triangle* newTriangles = new Triangle[size];
            // foreach face
            for (size_t vf = 0; vf < size; vf++)
            {
                const float3 vx = float3(vertices[3 * size_t(shapes[s].mesh.indices[3*vf].vertex_index) + 0],
                                         vertices[3 * size_t(shapes[s].mesh.indices[3*vf].vertex_index) + 1],
                                         vertices[3 * size_t(shapes[s].mesh.indices[3*vf].vertex_index) + 2]);

                const float3 vy = float3(vertices[3 * size_t(shapes[s].mesh.indices[3*vf+1].vertex_index) + 0],
                                         vertices[3 * size_t(shapes[s].mesh.indices[3*vf+1].vertex_index) + 1],
                                         vertices[3 * size_t(shapes[s].mesh.indices[3*vf+1].vertex_index) + 2]);

                const float3 vz = float3(vertices[3 * size_t(shapes[s].mesh.indices[3*vf+2].vertex_index) + 0],
                                         vertices[3 * size_t(shapes[s].mesh.indices[3*vf+2].vertex_index) + 1],
                                         vertices[3 * size_t(shapes[s].mesh.indices[3*vf+2].vertex_index) + 2]);

                // create triangle
                Triangle newTriangle = Triangle(id + vf, vx + transform, vy + transform, vz + transform, material);
                newTriangles[vf] = newTriangle;
            }
            
            id += size;

            if (triangles_size == 0)
            {
                triangles = new Triangle[triangles_size + size];
                std::copy(triangles, triangles + triangles_size, triangles);
                std::copy(newTriangles, newTriangles + size, triangles + triangles_size);
                triangles_size = triangles_size + size;
            }
            else
            {
                Triangle* triangles_buffer = new Triangle[triangles_size];
                std::copy(triangles, triangles + triangles_size, triangles_buffer);

                triangles = new Triangle[triangles_size + size];
                std::copy(triangles_buffer, triangles_buffer + triangles_size, triangles);
                std::copy(newTriangles, newTriangles + size, triangles + triangles_size);
                triangles_size = triangles_size + size;
            }
        }
        std::cout << "loaded: " << triangles_size << " in total" << std::endl;
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
        quads[0].T = M1, quads[0].invT = M1.FastInvertedTransformNoScale();
        // cube animation: spin
        mat4 M2base = mat4::RotateX( PI / 4 ) * mat4::RotateZ( PI / 4 );
        mat4 M2 = mat4::Translate( float3( 1.4f, 0, 2 ) ) * mat4::RotateY( animTime * 0.5f ) * M2base;
        cubes[0].M = M2, cubes[0].invM = M2.FastInvertedTransformNoScale();
        // sphere animation: bounce
        float tm = 1 - sqrf( fmodf( animTime, 2.0f ) - 1 );
        //spheres[0].pos = float3( -1.4f, -0.5f + tm, 2 );
        
    }
    float3 * GetLightPos()
    {
        // light point position is the middle of the swinging quad
        float3 corner1 = TransformPosition( float3( -0.5f, 0, -0.5f ), quads[0].T );
        float3 corner2 = TransformPosition( float3( 0.5f, 0, 0.5f ), quads[0].T );
        float3 lights[] = { (corner1 + corner2) * 0.5f - float3(0, 0.01f, 0), lightPos };
        return lights;
    }
    float3 * GetLightColor()
    {
        return lightColors;
    }
    int GetNLights() const
    {
        return 2;
    }
    void FindNearest( Ray& ray) const
    {
        quads[0].Intersect( ray );
        cubes[0].Intersect( ray );

        if(useQBVH)
            bvh->IntersectQBVH( ray );
        else
            bvh->IntersectBVH(ray);
    }
    bool IsOccluded( Ray& ray) const
    {
        quads[0].Intersect(ray);
        cubes[0].Intersect(ray);

        if(useQBVH)
            return bvh->IntersectQBVHShadowRay(ray, ray.objIdx != -1);
        else
            return bvh->IntersectBVHShadowRay(ray, ray.objIdx != -1);

    }
    float3 GetNormal( int objIdx, float3 I, float3 wo ) const
    {
        // we get the normal after finding the nearest intersection:
        // this way we prevent calculating it multiple times.
        if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
        float3 N;
        int lowerLimit = 0;
        int upperLimit = quads_size;
        if (objIdx >= lowerLimit && objIdx < upperLimit)
            N = quads[objIdx].GetNormal(I);

        lowerLimit = upperLimit;
        upperLimit += spheres_size;
        if (objIdx >= lowerLimit && objIdx < upperLimit)
            N = spheres[objIdx - lowerLimit].GetNormal(I);

        lowerLimit = upperLimit;
        upperLimit += cubes_size;

        if (objIdx >= lowerLimit  && objIdx < upperLimit)
            N = cubes[objIdx - lowerLimit].GetNormal(I);

        lowerLimit = upperLimit;
        upperLimit += planes_size;

        if (objIdx >= lowerLimit && objIdx < upperLimit)
        {
            N = planes[objIdx - lowerLimit].GetNormal(I);
        }

        if (objIdx >= upperLimit) N = triangles[objIdx - upperLimit].GetNormal(I);

        if (dot( N, wo ) > 0) N = -N; // hit backside / inside
        return N;
    }
    float3 GetAlbedo( int objIdx, float3 I, Material mat ) const
    {

        if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash

        int lowerLimit = 0;
        int upperLimit = quads_size;
        if (objIdx >= lowerLimit && objIdx < upperLimit)
            return quads[objIdx].GetAlbedo(I);

        lowerLimit = upperLimit;
        upperLimit += spheres_size;
        if (objIdx >= lowerLimit && objIdx < upperLimit)
            return mat.color;

        lowerLimit = upperLimit;
        upperLimit += cubes_size;

        if (objIdx >= lowerLimit && objIdx < upperLimit)
            return mat.color;

        lowerLimit = upperLimit;
        upperLimit += planes_size;

        if (objIdx >= lowerLimit && objIdx < upperLimit)
        {
             return planes[objIdx - lowerLimit].GetAlbedo(I);
        }

        if (objIdx >= upperLimit) return mat.color;

    }

    Material GetMaterial(int objIdx) const
    {
        if (objIdx == -1) return Material(); // or perhaps we should just crash
        int lowerLimit = 0;
        int upperLimit = quads_size;
        if (objIdx >= lowerLimit && objIdx < upperLimit)
            return quads[objIdx].material;

        lowerLimit = upperLimit;
        upperLimit += spheres_size;
        if (objIdx >= lowerLimit && objIdx < upperLimit)
            return spheres[objIdx - lowerLimit].material;

        lowerLimit = upperLimit;
        upperLimit += cubes_size;

        if (objIdx >= lowerLimit && objIdx < upperLimit)
            return cubes[objIdx - lowerLimit].material;

        lowerLimit = upperLimit;
        upperLimit += planes_size;

        if (objIdx >= lowerLimit && objIdx < upperLimit)
        {
             return planes[objIdx - lowerLimit].material;
        }

        if (objIdx >= upperLimit) return triangles[objIdx - upperLimit].material;
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
        float3 * light_postion = GetLightPos();
        float3 * light_color = GetLightColor();

        float3 lightAccumulator = float3(0);

        for (size_t i = 0; i < GetNLights(); i++)
        {
            float3 shadowRayDirection = light_postion[i] - intersection;
            float3 shadowRayDirectionNorm = normalize(shadowRayDirection);
            float epsilonOffset = 0.001f;
            Ray shadowRay = Ray(intersection + shadowRayDirectionNorm * epsilonOffset, shadowRayDirectionNorm);
            float shadowRayMagnitude = magnitude(shadowRayDirection);
            shadowRay.t = shadowRayMagnitude - epsilonOffset*2 ;
            if (!IsOccluded(shadowRay))
            {
                float distanceEnergy = 1 / sqrf(shadowRay.t);
                float angularEnergy = max(dot(normal, shadowRayDirectionNorm), 0.0f);
                lightAccumulator += light_color[i] * intensity * distanceEnergy * angularEnergy;
            }
        }
        return lightAccumulator;
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

    void SetLightIntensity(float newIntensity) { intensity = newIntensity; }

    void SetLightPos(float3 newLightPos)
    {
        lightPos = newLightPos;
    }

    void SetLightColor(float3 newLightColor)
    {
        lightColors[1] = newLightColor;
    }

    void SetRoofLightColor(float3 newLightColor)
    {
        lightColors[0] = newLightColor;
    }

    void SetQBVH(bool mode)
    {
        useQBVH = mode;
    }

    __declspec(align(64)) // start a new cacheline here
    float animTime = 0;
    bool isDynamic = false;

    float intensity = 24;
    float3 lightPos = float3(-1, 0, 0);
    float3 lightColors[2] = {float3(1, 1, 1), float3(0, 0, 0)};

    // Primitives
    Quad* quads = 0;
    Sphere* spheres = 0;
    Cube* cubes = 0;
    Plane* planes = 0;
    Triangle* triangles = 0;

    // Primitives count
    int quads_size = 0;
    int spheres_size = 0;
    int cubes_size = 0;
    int planes_size = 0;
    int triangles_size = 0;

    Material def_mat, mirror_mat, glass_mat, light_mat, absorb_all_but_blue_mat, red_mat, cyan_mat;
    BVH* bvh = 0;
    int id = 0;

    // QBVH
    bool useQBVH = true;
};

}
