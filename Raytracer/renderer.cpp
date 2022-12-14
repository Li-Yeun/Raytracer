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

/// <summary>
/// Function that returns the pixelcolor when primary ray intersects with glass.
/// </summary>
/// <param name="intersection">intersection point</param>
/// <param name="normal">normal of the intersection object</param>
/// <param name="albedo">albedo of the intersection object</param>
/// <param name="material">material of the intersection object</param>
/// <param name="ray">the ray that has cast</param>
float3 Renderer::ComputePixelColorGlass(float3 intersection, float3 normal, float3 albedo, Material& material, Ray& ray, int recursion_depth)
{
  // Compute reflection
  float3 reflect_direction = ray.D - 2.0f * (dot(ray.D, normal)) * normal;
  Ray reflectRay = Ray(intersection + reflect_direction * 0.001f, reflect_direction);

  // Compute Refraction & Absoption
  float air_refractive_index = 1.0003f;
  float n1, n2, refraction_ratio;

  float3 absorption = float3(1.0f);
  if (ray.inside)
  {
	absorption = float3(exp(-material.absorption.x * ray.t), exp(-material.absorption.y * ray.t), exp(-material.absorption.z * ray.t));
	n1 = material.refractive_index;
	n2 = air_refractive_index;
	reflectRay.inside = true;
  }
  else
  {
	n1 = air_refractive_index;
	n2 = material.refractive_index;
  }

  refraction_ratio = n1 / n2;

  float3 reflectionColor = albedo * absorption * Trace(reflectRay, recursion_depth + 1);

  float incoming_angle = dot(normal, -ray.D);
  float k = 1.0f - sqrf(refraction_ratio) * (1.0f - sqrf(incoming_angle));

  if (k < 0) return reflectionColor;

  float3 refraction_direction = refraction_ratio * ray.D + normal * (refraction_ratio * incoming_angle - sqrt(k));

  Ray refractRay = Ray(intersection + refraction_direction * 0.001f, refraction_direction);
  refractRay.inside = !ray.inside;

  float3 refractionColor = albedo * absorption * Trace(refractRay, recursion_depth + 1);

  // Compute Freshnel 
  float outcoming_angle = dot(-normal, refraction_direction);
  double leftFracture = sqrf((n1 * incoming_angle - material.refractive_index * outcoming_angle) / (n1 * incoming_angle + n2 * outcoming_angle));
  double rightFracture = sqrf((n1 * outcoming_angle - material.refractive_index * incoming_angle) / (n1 * outcoming_angle + n2 * incoming_angle));

  float Fr = 0.5f * (leftFracture + rightFracture);

  return Fr * reflectionColor + (1 - Fr) * refractionColor;
}

// -----------------------------------------------------------
// Evaluate light transport
// -----------------------------------------------------------
float3 Renderer::Trace( Ray& ray, int recursion_depth)
{
	if (recursion_depth >= recursionDepth)
		return float3(0);

	scene.FindNearest( ray );

	if (ray.objIdx == -1) return 0; // or a fancy sky color
	float3 intersection = ray.O + ray.t * ray.D;

	Material material = scene.GetMaterial(ray.objIdx);
	float3 normal = scene.GetNormal( ray.objIdx, intersection, ray.D );
	float3 albedo = scene.GetAlbedo( ray.objIdx, intersection, material); // TODO no longer needed, change to material.color (but then we need to make materials for the different objects

	switch (visualizationMode) {
		case RayTracing:
		  if (material.type == Material::MaterialType::DIFFUSE || material.type == Material::MaterialType::LIGHT) // (dirty fix for light materials in whitted renderer)
		  {
			return albedo/PI * scene.DirectIllumination(intersection, normal);

		  }
		  else if (material.type == Material::MaterialType::MIRROR)
		  {
			float3 reflect_direction = ray.D - 2 * (dot(ray.D, normal)) * normal;
			return material.color * Trace(Ray(intersection + reflect_direction * 0.001f, reflect_direction), recursion_depth + 1);
		  }
		  else if (material.type == Material::MaterialType::GLASS)
		  {
			return ComputePixelColorGlass(intersection, normal, albedo, material, ray, recursion_depth);
		  }
		  break;
		case PathTracing:
			  if (material.type == Material::MaterialType::LIGHT)
			  {
				return material.emission;
			  }
			  else if (material.type == Material::MaterialType::DIFFUSE)
			  {
				float3 newDirection = scene.DiffuseReflection(normal);
				float3 BRDF = albedo / PI;
				float3 Ei = Trace(Ray(intersection + newDirection * 0.001f, newDirection), recursion_depth + 1) * dot(normal, newDirection);
				return PI * 2.0f * BRDF * Ei;
			  }
			  else if (material.type == Material::MaterialType::MIRROR)
			  {
				float3 reflect_direction = ray.D - 2 * (dot(ray.D, normal)) * normal;
				return material.color * Trace(Ray(intersection + reflect_direction * 0.001f, reflect_direction), recursion_depth + 1);
			  }
			  else if (material.type == Material::MaterialType::GLASS)
			  {

				  // Compute Refraction & Absoption
				  float air_refractive_index = 1.0003f;
				  float n1, n2, refraction_ratio;

				  float3 absorption = float3(1.0f);
				  if (ray.inside)
				  {
					  absorption = float3(exp(-material.absorption.x * ray.t), exp(-material.absorption.y * ray.t), exp(-material.absorption.z * ray.t));
					  n1 = material.refractive_index;
					  n2 = air_refractive_index;
				  }
				  else
				  {
					  n1 = air_refractive_index;
					  n2 = material.refractive_index;
				  }

				  refraction_ratio = n1 / n2;

				  float incoming_angle = dot(normal, -ray.D);
				  float k = 1.0f - sqrf(refraction_ratio) * (1.0f - sqrf(incoming_angle));

				  // Compute Freshnel 
				  float3 refraction_direction = refraction_ratio * ray.D + normal * (refraction_ratio * incoming_angle - sqrt(k));

				  float outcoming_angle = dot(-normal, refraction_direction);
				  double leftFracture = sqrf((n1 * incoming_angle - material.refractive_index * outcoming_angle) / (n1 * incoming_angle + n2 * outcoming_angle));
				  double rightFracture = sqrf((n1 * outcoming_angle - material.refractive_index * incoming_angle) / (n1 * outcoming_angle + n2 * incoming_angle));

				  float Fr = 0.5f * (leftFracture + rightFracture);

				  if (k < 0 || RandomFloat() <= Fr)
				  {
					  // Compute reflection
					  float3 reflect_direction = ray.D - 2.0f * (dot(ray.D, normal)) * normal;
					  Ray reflectRay = Ray(intersection + reflect_direction * 0.001f, reflect_direction);
					  reflectRay.inside = ray.inside;
					  return albedo * absorption * Trace(reflectRay, recursion_depth + 1);

				  }
				  else
				  {
					  // Compute refraction
					  Ray refractRay = Ray(intersection + refraction_direction * 0.001f, refraction_direction);
					  refractRay.inside = !ray.inside;
					  return albedo * absorption * Trace(refractRay, recursion_depth + 1);

				  }
			  };
		  break;
		case Albedo:
		  return albedo;
		  break;
		case Normal:
			return (normal + 1) * 0.5f;
			break;
		case Distance:
			return 0.1f * float3(ray.t, ray.t, ray.t);
			break;
	}
}

void Renderer::KeyDown(int key) {
	float velocity = .05f;
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
			camera->rotateDirections[2] = true;
		    break;
		case GLFW_KEY_DOWN:
			camera->rotateDirections[3] = true;
			break;
		case GLFW_KEY_LEFT:
			camera->rotateDirections[0] = true;
			break;
		case GLFW_KEY_RIGHT:
			camera->rotateDirections[1] = true;
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
		camera->rotateDirections[2] = false;
		break;
	case GLFW_KEY_DOWN:
		camera->rotateDirections[3] = false;
		break;
	case GLFW_KEY_LEFT:
		camera->rotateDirections[0] = false;
		break;
	case GLFW_KEY_RIGHT:
		camera->rotateDirections[1] = false;
		break;
	default:
		break;
	}
}
// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------

void Renderer::Tick(float deltaTime)
{
	// animation
	static float animTime = 0;
	scene.SetTime(animTime += deltaTime * 0.002f);
	// pixel loop
	Timer t;

	if (visualizationMode == PathTracing)
	{
		accumulatedFrames += 1;
		// lines are executed as OpenMP parallel tasks (disabled in DEBUG)
		#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < SCRHEIGHT; y++)
		{
			// trace a primary ray for each pixel on the line
			for (int x = 0; x < SCRWIDTH; x++)
				accumulator[x + y * SCRWIDTH] +=
				float4(Trace(camera->GetPrimaryRay(x, y), 0), 0);
			// translate accumulator contents to rgb32 pixels
			for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
				screen->pixels[dest + x] =
				RGBF32_to_RGB8(&accumulator[x + y * SCRWIDTH], accumulatedFrames);
		}
	}
	else {

		if (anti_aliasing == false)
		{
			// lines are executed as OpenMP parallel tasks (disabled in DEBUG)
			#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < SCRHEIGHT; y++)
			{
				// trace a primary ray for each pixel on the line
				for (int x = 0; x < SCRWIDTH; x++)
					accumulator[x + y * SCRWIDTH] =
					float4(Trace(camera->GetPrimaryRay(x, y), 0), 0);
				// translate accumulator contents to rgb32 pixels
				for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
					screen->pixels[dest + x] =
					RGBF32_to_RGB8(&accumulator[x + y * SCRWIDTH]);
			}
		}
		else
		{
			// lines are executed as OpenMP parallel tasks (disabled in DEBUG)
		#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < SCRHEIGHT; y++)
			{
				// trace a primary ray for each pixel on the line
				for (int x = 0; x < SCRWIDTH; x++)
				{
					accumulator[x + y * SCRWIDTH] = float4(0);

					for (int sample = 0; sample < 4; ++sample)
					{
						float sample_x = x + samplePattern[2 * sample];
						float sample_y = y + samplePattern[2 * sample + 1];
						accumulator[x + y * SCRWIDTH] += float4(Trace(camera->GetPrimaryRay(sample_x, sample_y), 0), 0);
					}
					accumulator[x + y * SCRWIDTH] /= 4.0f;
				}
				// translate accumulator contents to rgb32 pixels
				for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
					screen->pixels[dest + x] =
					RGBF32_to_RGB8(&accumulator[x + y * SCRWIDTH]);
			}
		}

		camera->Update();
	}
	// performance report - running average - ms, MRays/s
	static float avg = 10, alpha = 1;
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.5f;
	float fps = 1000 / avg, rps = (SCRWIDTH * SCRHEIGHT) * fps;
	//std::cout << camera.Direction().x << ", " << camera.Direction().y << ", " << camera.Direction().z << std::endl;
	printf("%5.2fms (%.1fps) - %.1fMrays/s\n", avg, fps, rps / 1000000);
}