#pragma once
class PostProcessing
{
public:
    PostProcessing() = default;
    PostProcessing(int screen_width, int screen_height, uint* pixels) :
        screen_width(screen_width),
		screen_height(screen_height),
		pixels(pixels),
        center(float2((float) screen_width / 2.0f, (float)screen_height / 2.0f)),
		max_distance(magnitude(center))
    {
		uint* newBuffer = new uint[screen_width * screen_height];
		buffer = newBuffer;
	}
    void Vignette(float vignette_intensity)
    {
		float2 cord;
		uint pixelColor, r, g, b;
		float distance, dimming;

		for (int y = 0; y < screen_height; y++)
			for (int x = 0; x < screen_width; x++)
			{
				cord = float2(x, y);
				distance = magnitude(center - cord);

				// TODO precalc this
				dimming = max(1 - (distance * vignette_intensity / max_distance), 0.0f);

				pixelColor = pixels[x + y *screen_width];
				r = ((pixelColor & 0xFF0000) >> 16) * dimming;
				g = ((pixelColor & 0x00FF00) >> 8) * dimming;
				b = (pixelColor & 0x0000FF) * dimming;

				pixelColor = (r << 16) + (g << 8) + b;
				pixels[x + y * screen_width] = pixelColor;
			}
    }

	void ChromaticAberration(float chroma_intensity)
	{
		float2 cord, cord2, dir;
		int2 newCord;
		uint pixelColor, r, g, b;
		float distance, distance_ratio;
		float g_chroma_offset = chroma_intensity;
		float b_chroma_offset = chroma_intensity * 2;

		for (int y = 0; y < screen_height; y++)
			for (int x = 0; x < screen_width; x++)
			{
				cord = float2(x, y);
				distance = magnitude(center - cord);
				distance_ratio = distance / max_distance;

				pixelColor = pixels[x + y * screen_width];
				r = (pixelColor & 0xFF0000) >> 16;

				dir = normalize(cord - center);
				cord2.x = cord.x + (int)(dir.x * g_chroma_offset * distance_ratio);
				cord2.y = cord.y + (int)(dir.y * g_chroma_offset * distance_ratio);
				newCord = int2(max(0, min((int)cord2.x, screen_width - 1)), max(0, min((int)cord2.y, screen_height - 1)));
				g = (pixels[newCord.x + newCord.y * screen_width] & 0x00FF00) >> 8;

				cord2.x = cord.x + (int)(dir.x * b_chroma_offset * distance_ratio);
				cord2.y = cord.y + (int)(dir.y * b_chroma_offset * distance_ratio);
				newCord = int2(max(0, min((int)cord2.x, screen_width - 1)), max(0, min((int)cord2.y, screen_height - 1)));
				b = (pixels[newCord.x + newCord.y * screen_width] & 0x0000FF);

				pixelColor = (r << 16) + (g << 8) + b;
				buffer[x + y * screen_width] = pixelColor;
			}

		memcpy(pixels, buffer, sizeof(uint) * screen_width * screen_height);
	}

	void GammaCorrection(float gamma)
	{
		float gamma_correction = 1.0f / gamma;
		uint pixelColor, r, g, b;

		for (int y = 0; y < screen_height; y++)
			for (int x = 0; x < screen_width; x++)
			{
				pixelColor = pixels[x + y * screen_width];
				r = (pixelColor & 0xFF0000) >> 16;
				r = 255.0f * pow(((float)r / 255.0f), gamma_correction);

				g = (pixelColor & 0x00FF00) >> 8;
				g = 255.0f * pow(((float)g / 255.0f), gamma_correction);

				b = pixelColor & 0x0000FF;
				b = 255.0f * pow(((float)b / 255.0f), gamma_correction);
				
				pixelColor = (r << 16) + (g << 8) + b;
				pixels[x + y * screen_width] = pixelColor;
			}
	}

private:
    int screen_width, screen_height;
	uint* pixels;
	uint* buffer;
    float2 center;
	float max_distance;
};