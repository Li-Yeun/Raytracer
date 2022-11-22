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
	enum Direction { Left, Right, Up, Down, In, Out };

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

	void Translate(float3 translate, int direction)
	{
		translation += translate;
		subTranslation[direction] = translate;
	}

	// Undo Translation
	void Translate(int direction)
	{
		translation -= subTranslation[direction];
		subTranslation[direction] = float3(0, 0, 0);
	}

	// Angle in Radians
	void Rotate(float3 rotate, int direction)
	{ 
		rotation += rotate;
		subRotation[direction] = rotate;
	}

	// Undo Rotation
	void Rotate( int direction)
	{
		rotation -= subRotation[direction];
		subRotation[direction] = float3(0, 0, 0);
	}
	float3 Direction()
	{
		float u = SCRWIDTH / 2.0f * (1.0f / SCRWIDTH);
		float v = SCRHEIGHT / 2.0f * (1.0f / SCRHEIGHT);
		float3 C = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);
		return normalize(C - camPos);
	}

	void MoveHorizontal(float speed, int direction)
	{
		float3 normDirection = normalize(topRight - topLeft);
		Translate(normDirection * speed, direction);
	}


	void MoveVertical(float speed, int direction)
	{
		float3 normDirection = normalize(topLeft - bottomLeft);
		Translate(normDirection * speed, direction);
	}

	void MoveDistal(float speed, int direction)
	{
		Translate(Direction() * speed, direction);
	}

	void RotateHorizontal(float angular_velocity, int direction)
	{
		float3 normDirection = normalize(topLeft - bottomLeft);
		Rotate(normDirection * angular_velocity, direction);
	}

	void RotateVertical(float angular_velocity, int direction)
	{
		float3 normDirection = normalize(topRight - topLeft);
		Rotate(normDirection * angular_velocity, direction);
	}

	void Update()
	{
		if (rotation.x + rotation.y + rotation.z != 0)
		{
			mat4 transformationMatrix = mat4::Rotate(normalize(rotation), magnitude(rotation));
			topLeft = (float3)(transformationMatrix * (topLeft - camPos)) + camPos;
			topRight = (float3)(transformationMatrix * (topRight - camPos)) + camPos;
			bottomLeft = (float3)(transformationMatrix * (bottomLeft - camPos)) + camPos;
		}

		if (translation.x + translation.y + translation.z != 0)
		{
			camPos += translation;
			topLeft += translation;
			topRight += translation;
			bottomLeft += translation;
		}

	}
private:
	float3 translation = float3(0, 0, 0);
	float3 rotation = float3(0, 0, 0);

	float3 subTranslation[6] = { float3(0,0,0) };
	float3 subRotation[4] = { float3(0,0,0) };
};

}