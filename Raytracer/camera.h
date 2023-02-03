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
	float angular_velocity = PI / 180;

	Camera()
	{
		// setup a basic view frustum
		camPos = float3( 0, 0, -2 );
		topLeft = float3( -aspect, 1, 0 );
		topRight = float3( aspect, 1, 0 );
		bottomLeft = float3( -aspect, -1, 0 );

		gpuCamProp = new float4[4]{ float4(camPos, 0) , float4(topLeft, 0), float4(topRight, 0), float4(bottomLeft, 0) };
	}

	Ray GetPrimaryRay( const int x, const int y )
	{
		// calculate pixel position on virtual screen plane
		const float u = (float)x * (1.0f / SCRWIDTH);
		const float v = (float)y * (1.0f / SCRHEIGHT);
		const float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);
		return Ray( camPos, normalize( P - camPos ));
	}
	// Anti-Aliasing Primary Rays
	Ray GetPrimaryRay(const float x, const float y)
	{
		// calculate pixel position on virtual screen plane
		const float u = x * (1.0f / SCRWIDTH);
		const float v = y * (1.0f / SCRHEIGHT);
		const float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);
		return Ray(camPos, normalize(P - camPos));
	}

	float aspect = (float)SCRWIDTH / (float)SCRHEIGHT;
	float3 camPos;
	float3 topLeft, topRight, bottomLeft;

	float4* gpuCamProp;

	void Translate(float3 translate, int direction)
	{
		translation += translate;
		subTranslation[direction] = translate;
	}

	float3 Center()
	{
		float u = SCRWIDTH / 2.0f * (1.0f / SCRWIDTH);
		float v = SCRHEIGHT / 2.0f * (1.0f / SCRHEIGHT);
		return topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);
	}

	float3 Direction()
	{
		return normalize(Center() - camPos);
	}

	// Undo Translation
	void Translate(int direction)
	{
		translation -= subTranslation[direction];
		subTranslation[direction] = float3(0, 0, 0);
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


	bool Update()
	{
		float3 rotation = float3(0);
		float3 horizontalDirection = normalize(topLeft - bottomLeft);
		float3 verticalDirection = normalize(topRight - topLeft);
		if (rotateDirections[0])
			rotation += horizontalDirection * angular_velocity;
		else if (rotateDirections[1])
			rotation -= horizontalDirection * angular_velocity;
		else if (rotateDirections[2])
			rotation += verticalDirection * angular_velocity;
		else if (rotateDirections[3])
			rotation -= verticalDirection * angular_velocity;

		if (rotateDirections[0] || rotateDirections[1] || rotateDirections[2] || rotateDirections[3])
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

		gpuCamProp[0] = float4(camPos, 0);
		gpuCamProp[1] = float4(topLeft, 0);
		gpuCamProp[2] = float4(topRight, 0);
		gpuCamProp[3] = float4(bottomLeft, 0);

		if (magnitude(translation) != 0 || magnitude(rotation) != 0)
			return true;
		
		return false;
	}

	void SetFov(int fov)
	{
		float size = magnitude(topRight - topLeft);
		float distance = size / (2.0f * tanf((float)fov * PI / (2.0f * 180.0f)));
		camPos = Center() - Direction() * distance;
	}

	void SetAspectRatio(float aspect)
	{
		float3 midPoint_width = (topLeft + topRight) / 2;

		float3 direction_width = normalize(topRight - topLeft);
		float3 direction_height = normalize(topLeft - bottomLeft);

		topLeft = midPoint_width - direction_width * aspect;
		topRight = midPoint_width + direction_width * aspect;
		bottomLeft = topLeft - direction_height * 2;
	}

	bool rotateDirections[4] = { false };

private:
	float3 translation = float3(0, 0, 0);

	float3 subTranslation[6] = { float3(0,0,0) };
};

}