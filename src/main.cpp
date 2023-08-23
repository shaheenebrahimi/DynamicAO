#include "Image.h"
#include "Camera.h"
#include "Scene.h"
#include "Ray.h"
// #include "BlinnPhong.h"
#include "Material.h"
#include "Mesh.h"
#include "Raytracer.h"
#include "Occluder.h"
#include "Rasterizer.h"
#include "Object.h"

#include <glm/glm.hpp>
#include <getopt.h>
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
	floor->setRotation(glm::vec4(-M_PI_2, 1.0f, 0.0f, 0.0f));
	floor->setScale(glm::vec3(3.0f, 3.0f, 3.0f));
	floor->addTexture(RES_DIR + "/textures/floorTex.png");


	// Add to scene
	scn.addObject(sphere);
	scn.addObject(floor);

	target = floor;

	return scn;
}

int main(int argc, char **argv) {
	/* Get Arguments */
	int opt;
	string filename = "out.png"; // optional name of output file
	int resolution = 512; // optional size of output file
	bool generate = false;

	while((opt = getopt(argc, argv, ":gf:r:")) != -1) { 
		switch(opt) {
			case 'g': generate = true; break;
			case 'f': filename = optarg; break;
			case 'r': resolution = atoi(optarg); break;
		}
	}

	/* Initializations */
	Scene scn = createScene();

	if (generate) { // generator
		chrono::time_point<chrono::system_clock> start, end; // Declare timers

		Occluder occluder (filename, resolution); // Initialize Occluder
		occluder.setScene(scn);
		occluder.init();

		start = chrono::system_clock::now();
		occluder.renderTexture(target);
		end = chrono::system_clock::now();

		chrono::duration<double> elapsed = end - start;
		cout << "Elapsed time: " << elapsed.count() << "s" << endl;

	}
	else { // viewer
		Rasterizer raster; // Initialize Rasterizer
		raster.setScene(scn);
		raster.init();
		raster.run();
	}

	/*
		TODOS:
		- Fix ao texture gen speed (benchmark: 1:40)
			- optimization bugs: pass by reference etc
			- pthreads on outerloop
		- Make textures better (noisy)
			- bluenoise sampling
	*/
	
	return 0;
}
