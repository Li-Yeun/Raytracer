#include "precomp.h"

// -----------------------------------------------------------
// Initialize the renderer
// -----------------------------------------------------------
void Renderer::Init()
{
	// create fp32 rgb pixel buffer to render to
	accumulator = (float4*)MALLOC64( SCRWIDTH * SCRHEIGHT * 16 );
	memset( accumulator, 0, SCRWIDTH * SCRHEIGHT * 16 );
}

// -----------------------------------------------------------
// Evaluate light transport
// -----------------------------------------------------------
float3 Renderer::Trace( Ray& ray )
{
	scene.FindNearest( ray );
	if (ray.objIdx == -1) return 0; // or a fancy sky color
	float3 intersection = ray.O + ray.t * ray.D;
	float3 normal = scene.GetNormal( ray.objIdx, intersection, ray.D );
	float3 albedo = scene.GetAlbedo( ray.objIdx, intersection);

	/* visualize normal */ // return (N + 1) * 0.5f;
	/* visualize distance */ // return 0.1f * float3( ray.t, ray.t, ray.t );
	/* visualize albedo */  return albedo * scene.DirectIllumination(intersection, normal);
}

void Renderer::KeyDown(int key) {
	float velocity = .05f;
	float angular_velocity = PI / 180;
	switch (key) {
		case GLFW_KEY_A:
			camera->MoveHorizontal(-velocity, camera->Left);
			break;
		case GLFW_KEY_D:
			camera->MoveHorizontal(velocity, camera->Right);
			break;
		case GLFW_KEY_W:
			camera->MoveVertical(velocity, camera->Up);
			break;
		case GLFW_KEY_S:
			camera->MoveVertical(-velocity, camera->Down);
			break;
		case GLFW_KEY_E:
			camera->MoveDistal(velocity, camera->In);
			break;
		case GLFW_KEY_Q:
			camera->MoveDistal(-velocity, camera->Out);
			break;
		case GLFW_KEY_UP:
			camera->RotateVertical(angular_velocity, camera->Up);
		    break;
		case GLFW_KEY_DOWN:
			camera->RotateVertical(-angular_velocity, camera->Down);
			break;
		case GLFW_KEY_LEFT:
			camera->RotateHorizontal(angular_velocity, camera->Left);
			break;
		case GLFW_KEY_RIGHT:
			camera->RotateHorizontal(-angular_velocity, camera->Right);
			break;
		default:
			break;
	}
}

void Renderer::KeyUp(int key) {
	switch (key) {
	case GLFW_KEY_A:
		camera->Translate(camera->Left);
		break;
	case GLFW_KEY_D:
		camera->Translate(camera->Right);
		break;
	case GLFW_KEY_W:
		camera->Translate(camera->Up);
		break;
	case GLFW_KEY_S:
		camera->Translate(camera->Down);
		break;
	case GLFW_KEY_E:
		camera->Translate(camera->In);
		break;
	case GLFW_KEY_Q:
		camera->Translate(camera->Out);
		break;
	case GLFW_KEY_UP:
		camera->Rotate(camera->Up);
		break;
	case GLFW_KEY_DOWN:
		camera->Rotate(camera->Down);
		break;
	case GLFW_KEY_LEFT:
		camera->Rotate(camera->Left);
		break;
	case GLFW_KEY_RIGHT:
		camera->Rotate(camera->Right);
		break;
	default:
		break;
	}
}
// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick( float deltaTime )
{
	// animation
	static float animTime = 0;
	scene.SetTime( animTime += deltaTime * 0.002f );
	// pixel loop
	Timer t;
	// lines are executed as OpenMP parallel tasks (disabled in DEBUG)
	#pragma omp parallel for schedule(dynamic)
	for (int y = 0; y < SCRHEIGHT; y++)
	{
		// trace a primary ray for each pixel on the line
		for (int x = 0; x < SCRWIDTH; x++)
			accumulator[x + y * SCRWIDTH] =
				float4( Trace( camera->GetPrimaryRay( x, y ) ), 0 );
		// translate accumulator contents to rgb32 pixels
		for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
			screen->pixels[dest + x] = 
				RGBF32_to_RGB8( &accumulator[x + y * SCRWIDTH] );
	}

	camera->Update();
	// performance report - running average - ms, MRays/s
	static float avg = 10, alpha = 1;
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.5f;
	float fps = 1000 / avg, rps = (SCRWIDTH * SCRHEIGHT) * fps;
	//std::cout << camera.Direction().x << ", " << camera.Direction().y << ", " << camera.Direction().z << std::endl;
	printf( "%5.2fms (%.1fps) - %.1fMrays/s\n", avg, fps, rps / 1000000 );
}