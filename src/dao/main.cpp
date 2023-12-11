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
	// std::shared_ptr<Light> l2 = make_shared<Light>(glm::vec3(-1.0f, 2.0f, -1.0f), 0.5f);

	scn.addLight(l1);
	// scn.addLight(l2);

	// Objects
	shared_ptr<Object> obj = make_shared<Object>(RES_DIR + "models/" + name + ".obj");
	obj->setMaterial(glm::vec3(0,0,1), glm::vec3(0.1,0.1,0.1), glm::vec3(0.1,0.1,0.1), 100);
	obj->addTexture(RES_DIR + "textures/" + name + ".png");
	obj->addEvaluator(RES_DIR + "evaluators/" + name + ".txt");
	scn.addObject(obj); // Add to scene

	return scn;
}

int main(int argc, char **argv) {
	///* Initializations */
	string name = "sphere2";
	Scene scn = createScene(name);

	// viewer
	Rasterizer raster; // Initialize Rasterizer
	raster.setScene(scn);
	raster.init();
	raster.run();
	
	return 0;
}
