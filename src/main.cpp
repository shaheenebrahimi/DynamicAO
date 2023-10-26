#include "Image.h"
#include "Camera.h"
#include "Scene.h"
#include "Ray.h"
#include "Material.h"
#include "Mesh.h"
#include "Raytracer.h"
#include "Occluder.h"
#include "Rasterizer.h"
#include "Object.h"
#include "Sampler.h"
#include "NeuralNetwork.h"
#include "Evaluator.cuh"


#include <glm/glm.hpp>
//#include <getopt.h>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

const string RES_DIR = "../resources/";
shared_ptr<Object> target;

Scene createScene() {
	Scene scn;

	// Lights
	std::shared_ptr<Light> l1 = make_shared<Light>(glm::vec3(1.0f, 2.0f, 2.0f), 0.5f);
	// std::shared_ptr<Light> l2 = make_shared<Light>(glm::vec3(-1.0f, 2.0f, -1.0f), 0.5f);

	scn.addLight(l1);
	// scn.addLight(l2);

	// Objects
	shared_ptr<Object> sphere = make_shared<Object>(RES_DIR + "models/sphere2.obj");
	sphere->setMaterial(glm::vec3(0,0,1), glm::vec3(0.1,0.1,0.1), glm::vec3(0.1,0.1,0.1), 100);
	sphere->addTexture(RES_DIR + "/textures/sphereTex.png");
	
	std::shared_ptr<Object> floor = make_shared<Object>(RES_DIR + "models/square.obj");
	floor->setMaterial(glm::vec3(1,0,0), glm::vec3(0.1,0.1,0.1), glm::vec3(0.1,0.1,0.1), 100);
	floor->setPosition(glm::vec3(0.0f, -1.0f, 0.0f));
	//floor->setRotation(glm::vec4(-M_PI_2, 1.0f, 0.0f, 0.0f));
	floor->setRotation(glm::vec4(-1.57, 1.0f, 0.0f, 0.0f));
	floor->setScale(glm::vec3(3.0f, 3.0f, 3.0f));
	floor->addTexture(RES_DIR + "/textures/floorTex.png");


	// Add to scene
	scn.addObject(sphere);
	scn.addObject(floor);

	target = sphere;

	return scn;
}

int main(int argc, char **argv) {
	/* Get Arguments */
	//int opt;
	//string filename = "out.png"; // optional name of output file
	//int resolution = 512; // optional size of output file
	//bool generate = false;

	///*while((opt = getopt(argc, argv, ":gf:r:")) != -1) { 
	//	switch(opt) {
	//		case 'g': generate = true; break;
	//		case 'f': filename = optarg; break;
	//		case 'r': resolution = atoi(optarg); break;
	//	}
	//}*/

	///* Initializations */
	//Scene scn = createScene();

	//if (generate) { // generator
	//	chrono::time_point<chrono::system_clock> start, end; // Declare timers

	//	Occluder occluder (filename, resolution); // Initialize Occluder
	//	occluder.setScene(scn);
	//	occluder.init();

	//	start = chrono::system_clock::now();
	//	occluder.renderTexture(target);
	//	end = chrono::system_clock::now();

	//	chrono::duration<double> elapsed = end - start;
	//	cout << "Elapsed time: " << elapsed.count() << "s" << endl;

	//}
	//else { // viewer
	//	Rasterizer raster; // Initialize Rasterizer
	//	raster.setScene(scn);
	//	raster.init();
	//	raster.run();
	//}
	// Initialize arrays A, B, and C.

	// Sampler s (sphere);
	// s.sample(50); // samples per triangle
	

	Matrix i0(1, 2);
	float buf0[2] = { 0.5, 0.5 };
	i0.allocateMemory();
	i0.setBuf(buf0);

	Matrix i1(1, 2);
	float buf1[2] = { 0.75, 0.7 };
	i1.allocateMemory();
	i1.setBuf(buf1);

	Matrix i2(1, 2);
	float buf2[2] = { 0.1, 0.05 };
	i2.allocateMemory();
	i2.setBuf(buf2);

	Matrix i3(1, 2);
	float buf3[2] = { 0.8, 0.9 };
	i3.allocateMemory();
	i3.setBuf(buf3);

	Batch input({ i0, i1, i2, i3 });

	Evaluator ev;
	ev.loadEvaluator(RES_DIR + "evaluators/model.txt");
	auto res = ev.evaluateBatch(input);
	cout << "Batched EV: ";
	for (int i = 0; i < res.size(); ++i) {
		cout << res[i] << " ";
	}
	
	return 0;
}
