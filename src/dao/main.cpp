#include "Image.h"
#include "Scene.h"
#include "Ray.h"
#include "Material.h"
#include "Mesh.h"
#include "Rasterizer.h"
#include "Object.h"
#include "Evaluator.cuh"
#include "Batch.cuh"

#include <glm/glm.hpp>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>

#define DEG_TO_RAD 3.1415/180.0

using namespace std;

const std::string RES_DIR =
	#ifdef _WIN32
	// on windows, visual studio creates _two_ levels of build dir
	"../../../resources/"
	#else
	// on linux, common practice is to have ONE level of build dir
	"../../resources/"
	#endif
;

Scene createScene(const string &name) {
	Scene scn;

	// Lights
	std::shared_ptr<Light> l1 = make_shared<Light>(glm::vec3(1.0f, 2.0f, 2.0f), 0.5f);
	scn.addLight(l1);

	// Objects
	shared_ptr<Object> obj = make_shared<Object>();
	obj->addMesh(RES_DIR + "models/", name);
	obj->addTexture(RES_DIR + "textures/" + name + ".png");
	obj->addEvaluator(RES_DIR + "evaluators/" + name + ".txt");
	obj->setMaterial(glm::vec3(0.2, 0.2, 0.2), glm::vec3(0.1, 0.1, 0.1), glm::vec3(0.1, 0.1, 0.1), 100);
	scn.addObject(obj); // Add to scene

	return scn;
}

void generateMeshes(int genCount) {
	Mesh mesh;
	mesh.loader(RES_DIR + "models/", "arm");

	// TODO: make work for all theta values
	for (int i = 0; i < genCount; ++i) {
		// random angle between 0 and 100 degrees
		float theta = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 100));
		theta = theta * DEG_TO_RAD;

		vector<float> thetas(mesh.getBoneCount(), 0);
		int boneInd = mesh.getBoneIndex("mixamorig:RightForeArm");
		thetas[boneInd] = theta;

		mesh.setBoneAngles(thetas);
		vector<string> header = {
			"The next comment says how many theta values and the following one has their values",
			to_string(1),
			to_string(theta)
		};
		mesh.dumpMesh(RES_DIR + "data/arm" + to_string(i) + ".obj", header);
	}
}

int main(int argc, char **argv) {
	// TODO: fix so that doesn't need texture to run
	///* Initializations */
	string name = "arm";
	//string name = "sphere2";
	Scene scn = createScene(name);

	// viewer
	Rasterizer raster; // Initialize Rasterizer
	raster.setScene(scn);
	raster.init();
	raster.run();

	//generateMeshes(50);
	
	return 0;
}
