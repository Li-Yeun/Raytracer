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

	// Angle in Radians
	void Rotate(float3 direction, float angular_velocity)
	{
		mat4 rotationMatrix = mat4::Rotate(direction, angular_velocity);
		topLeft = (float3) (rotationMatrix * (topLeft - camPos)) + camPos;
		topRight = (float3) (rotationMatrix * (topRight - camPos)) + camPos;
		bottomLeft = (float3) (rotationMatrix * (bottomLeft - camPos)) + camPos;
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

	void RotateHorizontal(float angular_velocity)
	{
		float3 direction = normalize(topLeft - bottomLeft);
		Rotate(direction, angular_velocity);
	}

	void RotateVertical(float angular_velocity)
	{
		float3 direction = normalize(topRight - topLeft);
		Rotate(direction, angular_velocity);
	}
};

}