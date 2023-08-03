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
	target = make_shared<Object>(RES_DIR + "models/sphere2.obj");
	target->setMaterial(glm::vec3(0,0,1), glm::vec3(0.1,0.1,0.1), glm::vec3(0.1,0.1,0.1), 100);
	target->addTexture(RES_DIR + "/textures/aoTexture.png");

	std::shared_ptr<Object> floor = make_shared<Object>(RES_DIR + "models/square.obj");
	floor->setMaterial(glm::vec3(1,0,0), glm::vec3(0.1,0.1,0.1), glm::vec3(0.1,0.1,0.1), 100);
	floor->setPosition(glm::vec3(0.0f, -1.0f, 0.0f));
	floor->setRotation(glm::vec4(-M_PI_2, 1.0f, 0.0f, 0.0f));
	floor->setScale(glm::vec3(3.0f, 3.0f, 3.0f));

	// Add to scene
	scn.addObject(target);
	scn.addObject(floor);

	return scn;
}


int main(int argc, char **argv) {
	/* Get Arguments */
	int opt;
	string filename = "out.png"; // optional name of output file
	int resolution = 512; // optional size of output file
	// int renderer = 0; // renderer type (rasterizer, raytracer, occluder)

	while((opt = getopt(argc, argv, "f:r:")) != -1) { 
		switch(opt) {
			case 'f': filename = optarg; break;
			case 'r': resolution = atoi(optarg); break;
		}
	}

	/* Initializations */
	Scene scn = createScene();

	// Initialize Raytracer
	Raytracer tracer (filename, resolution);
	tracer.setScene(scn);
	tracer.render();

	// Initialize Occluder
	// Occluder occluder (filename, resolution);
	// occluder.setScene(scn);
	// occluder.renderTexture(target);

	// Initialize Rasterizer
	// Rasterizer raster;
	// raster.setScene(scn);
	
	return 0;
}
