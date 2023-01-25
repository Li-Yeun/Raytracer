#pragma once

namespace Tmpl8
{

class Renderer : public TheApp
{
public:
	enum VisualizationMode { Normal, Distance, Albedo, RayTracing, PathTracing };
	// game flow methods
	void Init();
	float3 ComputePixelColorGlass(float3 intersection, float3 normal, float3 albedo, Material& material, Ray& ray, int recursion_depth);
	float3 Trace( Ray& ray, int recursion_depth);
	float3 Sample(Ray& ray);
	void Tick( float deltaTime );
	void Shutdown() { /* implement if you want to do something on exit */ }
	// input handling
	void MouseUp( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseDown( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseMove( int x, int y ) { mousePos.x = x, mousePos.y = y; }
	void MouseWheel( float y ) { /* implement if you want to handle the mouse wheel */ }
	void KeyUp(int key); //{ /* implement if you want to handle keys */ }
	void KeyDown(int key);// { /* implement if you want to handle keys */ }
	// data members
	int2 mousePos;
	float4* accumulator;
	Scene scene;
	Camera* camera = new Camera();
	bool anti_aliasing = false;
	// Microsoft MSAA 4x Standard Sample Pattern 
	float samplePattern[4 * 2] = {
		-1.0f / 8.0f,  3.0f / 8.0f,
		 3.0f / 8.0f,  1.0f / 8.0f,
		-3.0f / 8.0f, -1.0f / 8.0f,
		 1.0f / 8.0f, -3.0f / 8.0f,
	};

	VisualizationMode visualizationMode = RayTracing;
	int recursionDepth = 5;
	uint accumulatedFrames = 0;
	void SetAntiAliasing(bool AA) { anti_aliasing = AA; }

	void SetVisualizationMode(VisualizationMode visual) 
	{ 
		visualizationMode = visual; 
		if (visual == VisualizationMode::PathTracing)
		{
			accumulatedFrames = 0;
			//Clear accumulator
			#pragma omp parallel for schedule(dynamic)
			for (int y = 0; y < SCRHEIGHT; y++)
			{
				// trace a primary ray for each pixel on the line
				for (int x = 0; x < SCRWIDTH; x++)
					accumulator[x + y * SCRWIDTH] =
					float4(0);
			}
		}
	}
	void SetRecusionDepth(int newDepth) { recursionDepth = newDepth; }

	// GPU Setting
	bool useGPU = true;

	void SetGPU(bool mode) { 
		useGPU = mode; 

		// TODO Clearing frame is maybe unnecessary when GPU fully works

		accumulatedFrames = 0;
		//Clear accumulator
#pragma omp parallel for schedule(dynamic)
		for (int y = 0; y < SCRHEIGHT; y++)
		{
			// trace a primary ray for each pixel on the line
			for (int x = 0; x < SCRWIDTH; x++)
				accumulator[x + y * SCRWIDTH] =
				float4(0);
		}
		accumulatorBuffer->CopyToDevice();
	}

	//Kernels
	static inline Kernel* generateInitialPrimaryRaysKernel;
	static inline Kernel* generatePrimaryRaysKernel;
	static inline Kernel* extendKernel;
	static inline Kernel* shadeKernel;
	static inline Kernel* connectKernel;
	static inline Kernel* finalizeKernel;

	//Buffer

	// Screen Buffers
	static inline Buffer* deviceBuffer; // Buffer that stores and display the final pixel values
	static inline Buffer* accumulatorBuffer;

	// DELETE LATER
	float4* origins, *directions;
	float* distances;
	int* primIdxs, *rayCounter, *pixelIdxs;

	// Ray Buffers
	static inline Buffer* rayCounterBuffer;
	static inline Buffer* pixelIdxBuffer;
	static inline Buffer* originBuffer;
	static inline Buffer* directionBuffer;
	static inline Buffer* distanceBuffer;
	static inline Buffer* primIdxBuffer;

	// DELETE LATER
	float4* energies, *transmissions;

	// E & T
	static inline Buffer* energyBuffer;
	static inline Buffer* transmissionBuffer;

	// DELETE LATER
	float4* shadowOrigins, *shadowDirections;
	float* shadowDistances;
	int* shadowCounter, *shadowPixelIdxs;

	// Shadow Ray Buffers
	static inline Buffer* shadowCounterBuffer;
	static inline Buffer* shadowPixelIdxBuffer;
	static inline Buffer* shadowOriginBuffer;
	static inline Buffer* shadowDirectionBuffer;
	static inline Buffer* shadowDistanceBuffer;

	// DELETE LATER
	int* bounceCounter;

	// Bounce Ray Buffers
	static inline Buffer* bounceCounterBuffer;
	static inline Buffer* bouncePixelIdxBuffer;
	static inline Buffer* bounceOriginBuffer;
	static inline Buffer* bounceDirectionBuffer;
};

} // namespace Tmpl8