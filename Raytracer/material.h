#pragma once

class Material
{
public:
   enum class MaterialType { DIFFUSE, MIRROR, GLASS, SUBSTANCE, LIGHT };

  Material() = default;
  Material(MaterialType type, float3 color, float specularity, float refractive_index = 1, float3 absorption = float3(0), float3 emission = float3(0)) :
    type(type),
    color(color),
    specularity(specularity),
    refractive_index(refractive_index),
    absorption(absorption),
    emission(emission)
  {}
  MaterialType type;
  float3 color;
  float specularity;
  float refractive_index;
  float3 absorption;
  float3 emission; 

};