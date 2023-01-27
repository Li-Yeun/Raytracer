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


        mat4 M1base = mat4::Translate(float3(0, 2.6f, 2));
        mat4 M1 = M1base * mat4::RotateZ(sinf(0 * 0.6f) * 0.1f) * mat4::Translate(float3(0, -0.9, 0));

        mat4 M2base = mat4::RotateX(PI / 4) * mat4::RotateZ(PI / 4);
        mat4 M2 = mat4::Translate(float3(1.4f, 0, 2)) * mat4::RotateY(0 * 0.5f) * M2base;

        quads_size = 1;
        quads = new Quad[quads_size] { Quad(id++, 3, light_mat, M1) };                              // 0: light source

        spheres_size = 2;
        spheres = new Sphere[spheres_size]{ 
            Sphere(id++, float3(-1.4f, -0.5f, 2), 0.5f, absorb_all_but_blue_mat),                             // 1: bouncing ball
            Sphere(id++,float3(0, 2.5f, -3.07f), 0.5f, def_mat) };						        // 2: rounded corners		

        cubes_size = 1;
        cubes = new Cube[cubes_size] { Cube(id++, float3(0), float3(0.75f), def_mat, M2) };         // 3: cube
               
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

        LoadObject("assets/pyramid.obj", glass_mat, float3(0, 0, 1.5));
        //LoadObject("assets/monkey.obj",  cyan_mat, float3(1.5, 0, 1.5));
        //LoadObject("assets/monkey.obj",  red_mat, float3(-1.5, 0, 1.5));

        bvh = new BVH(spheres, spheres_size, planes, planes_size, triangles, triangles_size);
        SetTime( 0 );
        // Note: once we have triangle support we should get rid of the class
        // hierarchy: virtuals reduce performance somewhat.

        std::cout << std::endl << triangles_size << " Triangles loaded" << std::endl << std::endl << std::endl;

        int totalPrimitives = quads_size + spheres_size + cubes_size + planes_size + triangles_size;
        
        //  Intersections
        primMaterials = new int[totalPrimitives];
        quadMatrices = new mat4[quads_size];
        quadSizes = new float[quads_size];
        sphereInfos = new float4[spheres_size];
        triangleInfos1 = new float4[triangles_size];
        triangleInfos2 = new float4[triangles_size];
        triangleInfos3 = new float4[triangles_size];

        // Normals
        primitives = new float4[totalPrimitives];
        sphereInvrs = new float[spheres_size];
        albedos = new float4[totalPrimitives];

        for (int i = 0; i < totalPrimitives; i++)
        {
            int lowerLimit = 0;
            int upperLimit = quads_size;
            if (i >= lowerLimit && i < upperLimit)
            {
                primMaterials[i] = (int) quads[i].material.type;
                quadMatrices[i] = quads[i].invT;
                quadSizes[i] = quads[i].size;

                primitives[i] = float4(quads[i].N, 0);
                albedos[i] = float4(quads[i].material.color, 0);
            }

            lowerLimit = upperLimit;
            upperLimit += spheres_size;
            if (i >= lowerLimit && i < upperLimit)
            {
                primMaterials[i] = (int) spheres[i - lowerLimit].material.type;
                sphereInfos[i - lowerLimit] = float4(spheres[i - lowerLimit].pos, spheres[i - lowerLimit].r2);

                primitives[i] = float4(spheres[i - lowerLimit].N, 0);
                sphereInvrs[i - lowerLimit] = spheres[i - lowerLimit].invr;
                albedos[i] = float4(spheres[i - lowerLimit].material.color, 0);
            }

            lowerLimit = upperLimit;
            upperLimit += cubes_size;

            if (i >= lowerLimit && i < upperLimit)
            {
                primMaterials[i] = (int) cubes[i - lowerLimit].material.type;

                // TODO CHANGE AND DELETE LATER
                primitives[i] = float4(0);
                albedos[i] = float4(0);
            }

            lowerLimit = upperLimit;
            upperLimit += planes_size;

            if (i >= lowerLimit && i < upperLimit)
            {
                primMaterials[i] = (int)planes[i - lowerLimit].material.type;

                primitives[i] = float4(planes[i - lowerLimit].N, planes[i - lowerLimit].d);
                albedos[i] = float4(planes[i - lowerLimit].material.color, 0);
            }

            if (i >= upperLimit)
            {
                primMaterials[i] = (int)triangles[i - upperLimit].material.type;
                triangleInfos1[i - upperLimit] = float4(triangles[i - upperLimit].pos1, 0.0f);
                triangleInfos2[i - upperLimit] = float4(triangles[i - upperLimit].pos2, 0.0f);
                triangleInfos3[i - upperLimit] = float4(triangles[i - upperLimit].pos3, 0.0f);

                primitives[i] = float4(triangles[i - upperLimit].N, 0);
                albedos[i] = float4(triangles[i - upperLimit].material.color, 0);
            }

        }
        // Intersections
        primMaterialBuffer = new Buffer(totalPrimitives * sizeof(int), primMaterials, 0);
        quadMatrixBuffer = new Buffer(quads_size * sizeof(mat4), quadMatrices, 0);
        quadSizeBuffer = new Buffer(quads_size * sizeof(float), quadSizes, 0);
        sphereInfoBuffer = new Buffer(spheres_size * sizeof(float4), sphereInfos, 0);
        triangleInfo1Buffer = new Buffer(triangles_size * sizeof(float4), triangleInfos1, 0);
        triangleInfo2Buffer = new Buffer(triangles_size * sizeof(float4), triangleInfos2, 0);
        triangleInfo3Buffer = new Buffer(triangles_size * sizeof(float4), triangleInfos3, 0);

        // Normals
        primitiveBuffer = new Buffer(totalPrimitives * sizeof(float4), primitives , 0);
        sphereInvrBuffer = new Buffer(spheres_size * sizeof(float), sphereInvrs, 0);
        albedoBuffer = new Buffer(totalPrimitives * sizeof(float4), albedos, 0);//CL_MEM_READ_ONLY);

        float4* lights = new float4[3]{ float4(quads[0].c1,0), float4(quads[0].c2, 0), float4(quads[0].c3, 0) };
        lightBuffer = new Buffer(1 * 3 * sizeof(float4), lights, 0);


        static Surface logo("assets/logo.png");
        textureBuffer = new Buffer(logo.width * logo.height * sizeof(uint), logo.pixels, 0); // Todo set this as texture memory

        isInitialized = true;
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
        return;
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

    std::tuple<float, float3, float3, float3> RandomPointOnLight()
    {
        // Pick random light source
        Quad light_source = quads[0];
        
        // Pick random position
        float3 Nl = light_source.GetNormal(float3(0));
        float A = sqrf(light_source.s);

        float3 c1 = TransformPosition(float3(-light_source.size, 0, -light_source.size), light_source.T);
        float3 c2 = TransformPosition(float3(light_source.size, 0, -light_source.size), light_source.T);
        float3 c3 = TransformPosition(float3(light_source.size, 0, light_source.size), light_source.T);
        
        float3 c1c2 = normalize(c1 - c2);
        float randomLength = RandomFloat() * light_source.s;
        float3 u = c1c2 * randomLength;

        float3 c2c3 = normalize(c3 - c2);
        randomLength = RandomFloat() * light_source.s;
        float3 v = c2c3 * randomLength;

        float3 light_point = c2 + u + v - float3(0, 0.01f, 0);

        return std::make_tuple(A, Nl, light_point, light_source.material.emission);
    }

    float3 * GetLightPos()
    {
        // light point position is the middle of the swinging quad
        float3 corner1 = TransformPosition( float3( -quads[0].size, 0, -quads[0].size), quads[0].T );
        float3 corner2 = TransformPosition( float3( quads[0].size, 0, quads[0].size), quads[0].T );
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

      if (dot(normal, randomDirection) < 0)
          randomDirection = -randomDirection;

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

    // Primitive Buffers

    static inline Buffer* albedoBuffer;
    static inline Buffer* textureBuffer;
    static inline Buffer* lightBuffer;

    // Intersection
    int* primMaterials;
    mat4* quadMatrices;
    float* quadSizes;
    float4* sphereInfos;
    // TODO CUBES
    float4* triangleInfos1;
    float4* triangleInfos2;
    float4* triangleInfos3;
  
    static inline Buffer* primMaterialBuffer;
    static inline Buffer* quadMatrixBuffer;
    static inline Buffer* quadSizeBuffer;
    static inline Buffer* sphereInfoBuffer;
    static inline Buffer* triangleInfo1Buffer;
    static inline Buffer* triangleInfo2Buffer;
    static inline Buffer* triangleInfo3Buffer;
    // Normals
    float4* primitives;
    float* sphereInvrs;
    float4* albedos;

    static inline Buffer* primitiveBuffer;
    static inline Buffer* sphereInvrBuffer;



    // Check if scene has been fully initialzed
    bool isInitialized = false;
};

}
