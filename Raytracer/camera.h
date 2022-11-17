#pragma once

// default screen resolution
#define SCRWIDTH	1280
#define SCRHEIGHT	720
// #define FULLSCREEN
// #define DOUBLESIZE

namespace Tmpl8 {

class Camera
{
public:
	Camera()
	{
		// setup a basic view frustum
		camPos = float3( 0, 0, -2 );
		topLeft = float3( -aspect, 1, 0 );
		topRight = float3( aspect, 1, 0 );
		bottomLeft = float3( -aspect, -1, 0 );
	}
	Ray GetPrimaryRay( const int x, const int y )
	{
		// calculate pixel position on virtual screen plane
		const float u = (float)x * (1.0f / SCRWIDTH);
		const float v = (float)y * (1.0f / SCRHEIGHT);
		const float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);
		return Ray( camPos, normalize( P - camPos ) );
	}
	float aspect = (float)SCRWIDTH / (float)SCRHEIGHT;
	float3 camPos;
	float3 topLeft, topRight, bottomLeft;

	void Translate(float3 translate)
	{
		camPos += translate;
		topLeft += translate;
		topRight += translate;
		bottomLeft += translate;
	}

	float3 Direction()
	{
		float u = SCRWIDTH / 2.0f * (1.0f / SCRWIDTH);
		float v = SCRHEIGHT / 2.0f * (1.0f / SCRHEIGHT);
		float3 C = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);
		return normalize(C - camPos);
	}

	void MoveHorizontal(float speed)
	{
		float3 direction = normalize(topRight - topLeft);
		Translate(direction * speed);
	}

	void MoveVertical(float speed)
	{
		float3 direction = normalize(topLeft - bottomLeft);
		Translate(direction * speed);
	}

	void MoveDistal(float speed)
	{
		Translate(Direction() * speed);
	}
};

}