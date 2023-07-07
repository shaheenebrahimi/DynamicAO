#include "Image.h"
#include "Camera.h"
#include "Scene.h"
#include "Ray.h"
#include "Ellipsoid.h"
#include "BlinnPhong.h"
#include "Image.h"
#include "Material.h"
#include "Plane.h"
#include "Mesh.h"
#include "Sphere.h"


#include <glm/glm.hpp>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <random>

using namespace std;

const string RES_DIR = "../resources/";

Scene scn;
shared_ptr<Image> img;
int resolution = 1024;

Mesh* target;

void genOcclusionHemisphere(int samples, float radius, vector<glm::vec3>& kernel, vector<glm::vec3>& noise) {
	uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0]
	default_random_engine generator;

	for (int i = 0; i < samples; ++i) {
		// sample random vectors in unit hemisphere
		glm::vec3 pointSample(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) // ignore bottom half since not sphere
		);
		pointSample = glm::normalize(pointSample);
		// float scale = (float)i / samples; 
		// scale = lerp(0.1f, 1.0f, scale * scale);
		// pointSample *= scale;
		kernel.push_back(pointSample);

		// generate noise
		glm::vec3 noiseSample(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) * 2.0 - 1.0,
			0.0f // rotate along z
		);
		noise.push_back(noiseSample);
	}
}

void genOclusionTexture(int samples, float radius, Mesh* target, int textureResolution) {
	vector<glm::vec3> kernel; // ao
	vector<glm::vec3> noise; // ao noise
	genOcclusionHemisphere(samples, radius, kernel, noise);

	// float texelStep = 1.0f / (float) textureResolution;

	// optimization: iterate through bounding box of each triangle in mesh

	for (int ty = 0; ty < textureResolution; ++ty) { // texels 0 0 bottom left
		// cout << ty << endl;
		for (int tx = 0; tx < textureResolution; ++tx) {
			glm::vec2 texel (tx, ty);
			glm::vec2 texCoord = texel / (float) textureResolution;
			img->setPixel(tx, ty, 255, 255, 255); // default white
			
			for (Triangle* tri : target->triangles) { // does this texel intersect any triangles? 
				glm::vec3 bary = tri->computeBarycentric(texCoord); // x = a, y = b, z = c
				if (bary.x >= 0 && bary.x <= 1 && bary.y >= 0 && bary.y <= 1 && bary.z >= 0 && bary.z <= 1) {
					glm::vec3 pos = (bary.x * tri->vert0 + bary.y * tri->vert1 + bary.z * tri->vert2);
					glm::vec3 nor = (bary.x * tri->nor0 + bary.y * tri->nor1 + bary.z * tri->nor2);
					glm::vec3 worldPos = glm::vec3(glm::vec4(pos, 1.0f) * target->transform);
					glm::vec3 worldNor = glm::vec3(glm::vec4(nor, 0.0f) * inverse(transpose(target->transform)));
					float ao = scn.computePointAmbientOcclusion(worldPos, worldNor, kernel, noise, radius);
					img->setPixel(tx, ty, 255*ao, 255*ao, 255*ao); // bottom left to top right image
					break;
				}
			}
		}
	}
}

void renderOcclusion(int samples, float radius) {
	vector<glm::vec3> kernel; // ao kernel
	vector<glm::vec3> noise; // ao noise
	genOcclusionHemisphere(samples, radius, kernel, noise);

	// compute ao for every vertex
	for (int r = 0; r < resolution; r++) { // iterate through pixels
		for (int c = 0; c < resolution; c++) {
			Ray ray = scn.cam.getRay(r, c);
			float ao = scn.computeRayAmbientOcclusion(ray, kernel, noise, radius);
			img->setPixel(c, r, 255*ao, 255*ao, 255*ao);
		}
	}
}

void raytrace() {
    scn.setCamResolution(resolution);
	for (int r = 0; r < resolution; r++) { // iterate through pixels
		for (int c = 0; c < resolution; c++) {
			// cout << r << " " << c << endl;
			Ray ray = scn.cam.getRay(r, c);
			glm::vec3 fragColor = 255.0f * scn.computeColor(ray);

			float red = (fragColor[0] < 0) ? 0 : fragColor[0];
			float green = (fragColor[1] < 0) ? 0 : fragColor[1];
			float blue = (fragColor[2] < 0) ? 0 : fragColor[2];
			red = (red > 255) ? 255 : red;
			green = (green > 255) ? 255 : green;
			blue = (blue > 255) ? 255 : blue;

			img->setPixel(c, r, (unsigned char)red, (unsigned char)green, (unsigned char)blue);
		}
	}
}

void init() {

	img = make_shared<Image>(resolution, resolution);

	// Create Scene
	scn.cam.setResolution(resolution);

	// Lights
	scn.addLight(new Light(glm::vec3(1.0f, 2.0f, 2.0f), 0.5f));
	scn.addLight(new Light(glm::vec3(-1.0f, 2.0f, -1.0f), 0.5f));

	// Objects
	target = new Mesh(
		RES_DIR + "models/sphere2.obj",
		glm::vec3(0.0f, 0.0f, 0.0f), // position
		glm::vec4(0, 1.0f, 0.0f, 0.0f), // rotation
		glm::vec3(1.0f, 1.0f, 1.0f), // scale
		new BlinnPhong(
			glm::vec3(0.0f,0.0f,1.0f), // kd
			glm::vec3(1.0f,1.0f,0.5f), // ks
			glm::vec3(0.1f,0.1f,0.1f), // ka
			100.0f
		)
	);
	// scn.addShape(new Sphere(
	// 	glm::vec3(0.0f, 0.0f, 0.0f), // position
	// 	glm::vec3(1.0f, 1.0f, 1.0f), // scale
	// 	1.0f,
	// 	new BlinnPhong(
	// 		glm::vec3(0.0f,0.0f,1.0f), // kd
	// 		glm::vec3(1.0f,1.0f,0.5f), // ks
	// 		glm::vec3(0.1f,0.1f,0.1f), // ka
	// 		100.0f
	// 	)
	// ));
    scn.addShape(
		target
	);
}

int main(int argc, char **argv)
{
	string filename = (argc >= 2) ? argv[1] : "aoTexture.png"; // optional name of output file
	resolution = (argc >= 3) ? atoi(argv[2]) : 1024; // optional size of output file

	init();
	raytrace();
	// renderOcclusion(500, 0.25f);
	// genOclusionTexture(500, 1.5f, target, resolution);

	img->writeToFile(filename);
	
	return 0;
}
