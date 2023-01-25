#include "precomp.h"

// -----------------------------------------------------------
// Initialize the renderer
// -----------------------------------------------------------
void Renderer::Init()
{
	std::cout << sizeof(mat4) << std::endl;
	// create fp32 rgb pixel buffer to render to
	accumulator = (float4*)MALLOC64( SCRWIDTH * SCRHEIGHT * 16 );
	memset( accumulator, 0, SCRWIDTH * SCRHEIGHT * 16 );

	// Create Kernels
	generateInitialPrimaryRaysKernel = new Kernel("Kernels/generatePrimaryRays.cl", "GenerateInitialPrimaryRays");
	generatePrimaryRaysKernel = new Kernel("Kernels/generatePrimaryRays.cl", "GeneratePrimaryRays");
	extendKernel = new Kernel("Kernels/extend.cl", "Extend");
	shadeKernel = new Kernel("Kernels/shade.cl", "Shade");
	connectKernel = new Kernel("Kernels/connect.cl", "Connect");
	finalizeKernel = new Kernel("Kernels/finalize.cl", "Finalize");

	// Create Buffers
	deviceBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(uint), screen->pixels, 0);
	accumulatorBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4), accumulator, 0);

	// DELETE LATER
	rayCounter = new int[1]{ 0 };
	pixelIdxs = new int[SCRWIDTH * SCRHEIGHT];
	origins = new float4[SCRWIDTH * SCRHEIGHT];
	directions = new float4[SCRWIDTH * SCRHEIGHT];
	distances = new float[SCRWIDTH * SCRHEIGHT];
	primIdxs = new int[SCRWIDTH * SCRHEIGHT];

	// Primary Ray Buffers
	rayCounterBuffer = new Buffer(1 * sizeof(int), rayCounter, 0);
	pixelIdxBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(int), pixelIdxs, 0);
	originBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4), origins, 0);
	directionBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4), directions, 0);
	distanceBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float), distances, 0);
	primIdxBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(int), primIdxs, 0);

	// DELETE LATER
	energies = new float4[SCRWIDTH * SCRHEIGHT];
	transmissions = new float4[SCRWIDTH * SCRHEIGHT];
	// E & T Buffers
	energyBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4), energies, 0);
	transmissionBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4), transmissions, 0);

	// DELETE LATER
	shadowCounter = new int[1]{ 0 };
	shadowPixelIdxs = new int[SCRWIDTH * SCRHEIGHT];
	shadowOrigins = new float4[SCRWIDTH * SCRHEIGHT];
	shadowDirections = new float4[SCRWIDTH * SCRHEIGHT];
	shadowDistances = new float[SCRWIDTH * SCRHEIGHT];

	// Shadow Ray Buffers
	shadowCounterBuffer = new Buffer(1 * sizeof(int), shadowCounter, 0);
	shadowPixelIdxBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(int), shadowPixelIdxs, 0);
	shadowOriginBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4), shadowOrigins, 0);
	shadowDirectionBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4), shadowDirections, 0);
	shadowDistanceBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float), shadowDistances, 0);

	// DELETE LATER
	bounceCounter = new int[1]{ 0 };

	// Bounce Ray Buffers
	bounceCounterBuffer = new Buffer(1 * sizeof(int), bounceCounter, 0);
	bouncePixelIdxBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(int));  // Check
	bounceOriginBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4)); // Check
	bounceDirectionBuffer = new Buffer(SCRWIDTH * SCRHEIGHT * sizeof(float4)); // Check

	// Set Kernel Arguments
	generateInitialPrimaryRaysKernel->SetArguments(pixelIdxBuffer, originBuffer, directionBuffer, distanceBuffer, primIdxBuffer, // Primary Rays
	energyBuffer, transmissionBuffer);																							 // E & T

	// DELETE LATER
	pixelIdxBuffer->CopyToDevice(false);
	originBuffer->CopyToDevice(false);
	directionBuffer->CopyToDevice(false);
	distanceBuffer->CopyToDevice(false);	
	primIdxBuffer->CopyToDevice(false);
	energyBuffer->CopyToDevice(false);
	transmissionBuffer->CopyToDevice(false);

	generatePrimaryRaysKernel->SetArguments(rayCounterBuffer, pixelIdxBuffer, originBuffer, directionBuffer, distanceBuffer, primIdxBuffer, // Primary Rays
		shadowCounterBuffer,																												// Shadow Rays	
		bounceCounterBuffer, bouncePixelIdxBuffer, bounceOriginBuffer, bounceDirectionBuffer);											    // Bounce Rays  
	// DELETE LATER
	rayCounterBuffer->CopyToDevice(false);
	shadowCounterBuffer->CopyToDevice(false);
	bounceCounterBuffer->CopyToDevice(false);

	/*
	bouncePixelIdxBuffer->CopyToDevice(false);
	bounceOriginBuffer->CopyToDevice(false);
	bounceDirectionBuffer->CopyToDevice(false);
	*/
	finalizeKernel->SetArguments(deviceBuffer, accumulatorBuffer);

	accumulatorBuffer->CopyToDevice(false);
	deviceBuffer->CopyToDevice(true);

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

float3 Renderer::Sample(Ray& ray)
{
	float3 T = float3(1.0f, 1.0f, 1.0f), E = (0.0f, 0.0f, 0.0f);
	bool lastSpecular = true;

	while (1)
	{
		scene.FindNearest(ray);
		if (ray.objIdx == -1) break;

		Material material = scene.GetMaterial(ray.objIdx);

		if (material.type == Material::MaterialType::LIGHT)
		{
			if (lastSpecular)
				return  material.emission;

			break;
		}

		float3 intersection = ray.O + ray.t * ray.D;
		float3 normal = scene.GetNormal(ray.objIdx, intersection, ray.D);
		float3 albedo = scene.GetAlbedo(ray.objIdx, intersection, material);
		float3 BRDF = albedo / PI;

		// sample a random light source
		std::tuple<float, float3, float3, float3> result = scene.RandomPointOnLight();

		float A = std::get<0>(result);
		float3 Nl = std::get<1>(result);
		float3 light_point = std::get<2>(result);
		float3 L = light_point - intersection;
		float dist = magnitude(L);
		L = normalize(L);

		Ray lr = Ray(intersection + L * 0.001f, L, dist - 2.0f * 0.001f);

		if (dot(normal, L) > 0 && dot(Nl, -L) > 0 && !scene.IsOccluded(lr))
		{
			float solidAngle = (dot(Nl, -L) * A) / sqrf(dist);
			float lightPDF = 1.0f / solidAngle;
			E += T * (dot(normal, L) / lightPDF) * BRDF * std::get<3>(result);
		}

		// Russian Roulette
		float p = clamp(max(albedo.z, max(albedo.x, albedo.y)), 0.0f, 1.0f);

		if (p < RandomFloat()) 
			break; 

		T *= 1.0f / p;

		// continue random walk
		float3 R = scene.DiffuseReflection(normal);
		float hemiPDF = 1.0f / (PI * 2.0f);
		ray = Ray(intersection + R * 0.001f, R);
		T *= (dot(normal, R) / hemiPDF) * BRDF;

		lastSpecular = false;
	}
	return E;

}
/*	FULL SAMPLE CODE WITH GLASS AND MIRROR
float3 Renderer::Sample(Ray& ray)
{
	float3 T = float3(1.0f, 1.0f, 1.0f), E = (0.0f, 0.0f, 0.0f);

	bool lastSpecular = true;
	float3 lastAbsorption = float3(1.0f);
	while (1)
	{
		scene.FindNearest(ray);
		if (ray.objIdx == -1) break;

		Material material = scene.GetMaterial(ray.objIdx);
		float3 intersection = ray.O + ray.t * ray.D;
		float3 normal = scene.GetNormal(ray.objIdx, intersection, ray.D);
		float3 albedo = scene.GetAlbedo(ray.objIdx, intersection, material);
		float3 BRDF = albedo / PI;

		if (material.type == Material::MaterialType::LIGHT)
		{
			if (lastSpecular)
				return lastAbsorption * material.emission;

			break;
		}
		else if (material.type == Material::MaterialType::MIRROR)
		{
			float3 reflect_direction = ray.D - 2 * (dot(ray.D, normal)) * normal;
			ray = Ray(intersection + reflect_direction * 0.001f, reflect_direction);
			lastSpecular = true;
		}
		else if (material.type == Material::MaterialType::DIFFUSE)
		{
			// sample a random light source
			std::tuple<float, float3, float3, float3> result = scene.RandomPointOnLight();

			float A = std::get<0>(result);
			float3 Nl = std::get<1>(result);
			float3 light_point = std::get<2>(result);
			float3 L = light_point - intersection;
			float dist = magnitude(L);
			L = normalize(L);

			Ray lr = Ray(intersection + L * 0.001f, L, dist - 2.0f * 0.001f);

			if (dot(normal, L) > 0 && dot(Nl, -L) > 0 && !scene.IsOccluded(lr))
			{
				float solidAngle = (dot(Nl, -L) * A) / sqrf(dist);
				float lightPDF = 1.0f / solidAngle;
				E += T * (dot(normal, L) / lightPDF) * BRDF * lastAbsorption * std::get<3>(result);
			}

			// Russian Roulette
			float p = clamp(max(albedo.z, max(albedo.x, albedo.y)), 0.0f, 1.0f);
			if (p < RandomFloat()) break; else T *= 1.0f / p;

			// continue random walk
			float3 R = scene.DiffuseReflection(normal);
			float hemiPDF = 1.0f / (PI * 2.0f);
			ray = Ray(intersection + R * 0.001f, R);
			T *= (dot(normal, R) / hemiPDF) * BRDF * lastAbsorption;

			lastSpecular = false;
			lastAbsorption = float3(1.0f);
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
				ray = Ray(intersection + reflect_direction * 0.001f, reflect_direction);
				//reflectRay.inside = ray.inside;
				lastAbsorption *= albedo * absorption;

				//return albedo * absorption * Trace(reflectRay, recursion_depth + 1);

			}
			else
			{
				// Compute refraction
				ray = Ray(intersection + refraction_direction * 0.001f, refraction_direction);
				ray.inside = !ray.inside;
				lastAbsorption *= albedo * absorption;
				//return albedo * absorption * Trace(refractRay, recursion_depth + 1);

			}

			lastSpecular = false;

		}
	}
	return E;

}
*/

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
	if (!scene.isInitialized)
		return;

	/*
	std::cout << sizeof(float3) * scene.triangles_size << std::endl;
	std::cout << CL_DEVICE_MAX_MEM_ALLOC_SIZE << std::endl;
	std::cout << CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE << std::endl;
	std::cout << CL_DEVICE_GLOBAL_MEM_SIZE << std::endl;
	*/

	static bool firstTick = true;

	if (firstTick)
	{
		int planeStartIdx = scene.quads_size + scene.spheres_size + scene.cubes_size;
		shadeKernel->SetArguments(rayCounterBuffer, pixelIdxBuffer, originBuffer, directionBuffer, distanceBuffer, primIdxBuffer, // Primary Rays
			scene.albedoBuffer, scene.primitiveBuffer, scene.sphereInvrBuffer, float4(scene.quads_size, planeStartIdx, 0, 0), float4(scene.spheres_size, scene.planes_size, 0, 0), scene.textureBuffer,		  // Primitives
			scene.lightBuffer, scene.quads[0].A, scene.quads[0].s, float4(scene.quads[0].material.emission, 0),							  // TODO REMOVE A CAN BE CALCULATED FROM s   // Light Source(s)
			energyBuffer, transmissionBuffer,																					  // E & T
			shadowCounterBuffer, shadowPixelIdxBuffer, shadowOriginBuffer, shadowDirectionBuffer, shadowDistanceBuffer,			  // Shadow Rays
			bounceCounterBuffer, bouncePixelIdxBuffer, bounceOriginBuffer, bounceDirectionBuffer
		);

		scene.albedoBuffer->CopyToDevice(false);
		scene.primitiveBuffer->CopyToDevice(false);
		scene.sphereInvrBuffer->CopyToDevice(false);
		scene.textureBuffer->CopyToDevice(false);

		scene.lightBuffer->CopyToDevice(false);
		shadowPixelIdxBuffer->CopyToDevice(false);
		shadowOriginBuffer->CopyToDevice(false);
		shadowDirectionBuffer->CopyToDevice(false);
		shadowDistanceBuffer->CopyToDevice(true);

		firstTick = false;
	}
	// animation
	static float animTime = 0;
	scene.SetTime(animTime += deltaTime * 0.002f);
	// pixel loop
	Timer t;

	if (visualizationMode == PathTracing)
	{
		if (useGPU)
		{
			accumulatedFrames += 1;

			generateInitialPrimaryRaysKernel->S(7, camera->aspect);
			generateInitialPrimaryRaysKernel->S(8, float4(camera->camPos, 0));

			generateInitialPrimaryRaysKernel->Run(SCRWIDTH * SCRHEIGHT);

			pixelIdxBuffer->CopyFromDevice(false);
			originBuffer->CopyFromDevice(false);
			directionBuffer->CopyFromDevice(false);
			distanceBuffer->CopyFromDevice(false);
			primIdxBuffer->CopyFromDevice(true);
			clFinish(Kernel::GetQueue());
			int counter = 0;
			for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++)
			{
				float3 origin = float3(origins[i].x, origins[i].y, origins[i].z);
				float3 direction = float3(directions[i].x, directions[i].y, directions[i].z);
				Ray ray = Ray(origin, direction);

				scene.quads[0].Intersect(ray);
				//scene.cubes[0].Intersect(ray);

				if (scene.useQBVH)
					scene.bvh->IntersectQBVH(ray);
				else
					scene.bvh->IntersectBVH(ray);

				if (ray.objIdx == -1)
					continue;

				/* TODO IMPLEMNT THIS CORRECTLY FOR GPU
				if (material.type == Material::MaterialType::LIGHT)
				{
					if (lastSpecular)
						return  material.emission;

					break;
				}*/

				Material material = scene.GetMaterial(ray.objIdx);
				if (material.type == Material::MaterialType::LIGHT)
				{
					continue;
				}

				distances[counter] = ray.t;
				primIdxs[counter] = ray.objIdx;

				origins[counter] = origins[i];
				directions[counter] = directions[i];
				pixelIdxs[counter] = pixelIdxs[i];


				counter += 1;
			}

			if (counter == 0)
				return;

			rayCounter[0] = counter;

			distanceBuffer->CopyToDevice(false);
			primIdxBuffer->CopyToDevice(false);
			originBuffer->CopyToDevice(false);
			directionBuffer->CopyToDevice(false);
			pixelIdxBuffer->CopyToDevice(false);

			rayCounterBuffer->CopyToDevice(true);

			shadeKernel->S(27, (int) RandomUInt()); // Give a random seed to the GPU

			shadeKernel->Run(counter);

			bounceCounterBuffer->CopyFromDevice(false);

			shadowCounterBuffer->CopyFromDevice(false);
			shadowPixelIdxBuffer->CopyFromDevice(false);
			shadowOriginBuffer->CopyFromDevice(false);
			shadowDirectionBuffer->CopyFromDevice(false);
			shadowDistanceBuffer->CopyFromDevice(false);

			energyBuffer->CopyFromDevice(true);

			clFinish(Kernel::GetQueue());
			std::cout <<  "InitialShadowCounter:" << shadowCounter[0] << std::endl;
			for (int i = 0; i < shadowCounter[0]; i++)
			{
				float3 shadowOrigin = float3(shadowOrigins[i].x, shadowOrigins[i].y, shadowOrigins[i].z);
				float3 shadowDirection = float3(shadowDirections[i].x, shadowDirections[i].y, shadowDirections[i].z);

				Ray ray = Ray(shadowOrigin, shadowDirection, shadowDistances[i]);

				scene.quads[0].Intersect(ray);
				//scene.cubes[0].Intersect(ray);

				bool isOccluded;
				if (scene.useQBVH)
					isOccluded = scene.bvh->IntersectQBVHShadowRay(ray, ray.objIdx != -1);
				else
					isOccluded = scene.bvh->IntersectBVHShadowRay(ray, ray.objIdx != -1);

				if (!isOccluded)
				{
					accumulator[shadowPixelIdxs[i]] += energies[i];
				}
			}

			accumulatorBuffer->CopyToDevice(true);
			std::cout << "InitialbounceCounter: " << bounceCounter[0] << std::endl;

			int loop = 0;
			while (bounceCounter[0] > 0)
			{
				loop++;
				//std::cout << "bounceCounter: " << bounceCounter[0] << std::endl;

				generatePrimaryRaysKernel->Run(bounceCounter[0]);

				pixelIdxBuffer->CopyFromDevice(false);
				originBuffer->CopyFromDevice(false);
				directionBuffer->CopyFromDevice(false);
				distanceBuffer->CopyFromDevice(false);
				primIdxBuffer->CopyFromDevice(false);

				rayCounterBuffer->CopyFromDevice(true);
				clFinish(Kernel::GetQueue());
				int counter = 0;
				for (int i = 0; i < rayCounter[0]; i++)
				{
					float3 origin = float3(origins[i].x, origins[i].y, origins[i].z);
					float3 direction = float3(directions[i].x, directions[i].y, directions[i].z);
					Ray ray = Ray(origin, direction);

					scene.quads[0].Intersect(ray);
					//scene.cubes[0].Intersect(ray);

					if (scene.useQBVH)
						scene.bvh->IntersectQBVH(ray);
					else
						scene.bvh->IntersectBVH(ray);

					if (ray.objIdx == -1)
						continue;

					/* TODO IMPLEMNT THIS CORRECTLY FOR GPU
					if (material.type == Material::MaterialType::LIGHT)
					{
						if (lastSpecular)
							return  material.emission;

						break;
					}*/
					Material material = scene.GetMaterial(ray.objIdx);
					if (material.type == Material::MaterialType::LIGHT)
					{
						continue;
					}

					distances[counter] = ray.t;
					primIdxs[counter] = ray.objIdx;

					origins[counter] = origins[i];
					directions[counter] = directions[i];
					pixelIdxs[counter] = pixelIdxs[i];


					counter++;
				}
				if (counter == 0)
					break;

				rayCounter[0] = counter;

				distanceBuffer->CopyToDevice(false);
				primIdxBuffer->CopyToDevice(false);
				originBuffer->CopyToDevice(false);
				directionBuffer->CopyToDevice(false);
				pixelIdxBuffer->CopyToDevice(false);

				rayCounterBuffer->CopyToDevice(true);
				
				shadeKernel->S(27, (int)RandomUInt()); // Give a random seed to the GPU
				
				shadeKernel->Run(counter);
				
				bounceCounterBuffer->CopyFromDevice(false);
				shadowCounterBuffer->CopyFromDevice(false);
				shadowPixelIdxBuffer->CopyFromDevice(false);
				shadowOriginBuffer->CopyFromDevice(false);
				shadowDirectionBuffer->CopyFromDevice(false);
				shadowDistanceBuffer->CopyFromDevice(false);

				energyBuffer->CopyFromDevice(true);
				clFinish(Kernel::GetQueue());

				for (int i = 0; i < shadowCounter[0]; i++)
				{
					float3 shadowOrigin = float3(shadowOrigins[i].x, shadowOrigins[i].y, shadowOrigins[i].z);
					float3 shadowDirection = float3(shadowDirections[i].x, shadowDirections[i].y, shadowDirections[i].z);

					Ray ray = Ray(shadowOrigin, shadowDirection, shadowDistances[i]);

					scene.quads[0].Intersect(ray);
					//scene.cubes[0].Intersect(ray);

					bool isOccluded;
					if (scene.useQBVH)
						isOccluded = scene.bvh->IntersectQBVHShadowRay(ray, ray.objIdx != -1);
					else
						isOccluded = scene.bvh->IntersectBVHShadowRay(ray, ray.objIdx != -1);

					if (!isOccluded)
					{
						accumulator[shadowPixelIdxs[i]] += energies[i];
					}
				}

				accumulatorBuffer->CopyToDevice(true);
				
			}
			std::cout << "loop count:" << loop << std::endl;

			/*
			int co = 0;
			for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++)
			{
				if (magnitude(accumulator[i]) == 0)
				{
					co++;
					std::cout << "pixel: " << i << std::endl;
				}
			}
			std::cout << "totalpixel: " << co << std::endl;
			*/
			finalizeKernel->S(2, (int) accumulatedFrames);
			finalizeKernel->Run(SCRWIDTH * SCRHEIGHT);

			deviceBuffer->CopyFromDevice();
			clFinish(Kernel::GetQueue());
		}
		else {
			accumulatedFrames += 1;
			// lines are executed as OpenMP parallel tasks (disabled in DEBUG)
#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < SCRHEIGHT; y++)
			{
				// trace a primary ray for each pixel on the line
				for (int x = 0; x < SCRWIDTH; x++)
					accumulator[x + y * SCRWIDTH] +=
					float4(Sample(camera->GetPrimaryRay(x, y)), 0);
				//float4(Trace(camera->GetPrimaryRay(x, y), 0), 0);
			// translate accumulator contents to rgb32 pixels
				for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
					screen->pixels[dest + x] =
					RGBF32_to_RGB8(&accumulator[x + y * SCRWIDTH], accumulatedFrames);
			}
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