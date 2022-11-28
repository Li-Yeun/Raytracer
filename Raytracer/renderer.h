#pragma once

namespace Tmpl8
{

class Renderer : public TheApp
{
public:
	enum VisualizationMode { Normal, Distance, Albedo };
	// game flow methods
	void Init();
	float3 Trace( Ray& ray, int recursion_depth);
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

	VisualizationMode visualizationMode = Albedo;

	void SetAntiAliasing(bool AA) { anti_aliasing = AA; }
	void SetVisualizationMode(VisualizationMode visual) { visualizationMode = visual; }


};

} // namespace Tmpl8