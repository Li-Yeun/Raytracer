#pragma once

class Material
{
public:
   enum class MaterialType { DIFFUSE, MIRROR, GLASS, SUBSTANCE };

  Material() = default;
  Material(MaterialType type, float3 color, float specularity, float refractive_index = 1):
    type(type),
    color(color),
    specularity(specularity),
    refractive_index(refractive_index)
  {}
  MaterialType type;
  float3 color;
  float specularity;
  float refractive_index;

};