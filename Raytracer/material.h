enum MaterialType
{
  DIFFUSE,
  MIRROR,
  GLASS
};

class Material
{
public:
  MaterialType type;
  float3 color;
  float specularity;

  Material(MaterialType type, float3 color, float specularity)
  {
    type = type;
    color = color;
    specularity = specularity;
  }
};