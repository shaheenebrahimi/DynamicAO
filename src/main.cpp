#include "Image.h"
#include "Camera.h"
#include "Scene.h"
#include "Ray.h"
// #include "BlinnPhong.h"
#include "Material.h"
#include "Mesh.h"
#include "Raytracer.h"
#include "Occluder.h"
#include "Object.h"

#include <glm/glm.hpp>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <random>

using namespace std;

const string RES_DIR = "../resources/";

shared_ptr<Object> target;


void createScene(Scene& scn) {
	// Lights
	std::shared_ptr<Light> l1 = make_shared<Light>(glm::vec3(1.0f, 2.0f, 2.0f), 0.5f);
	std::shared_ptr<Light> l2 = make_shared<Light>(glm::vec3(-1.0f, 2.0f, -1.0f), 0.5f);

	scn.addLight(l1);
	scn.addLight(l2);

	// Objects
	target = make_shared<Object>(RES_DIR + "models/sphere2.obj");
	o1->setMaterial(glm::vec3(0,0,1), glm::vec3(0.1,0.1,0.1), glm::vec3(0.1,0.1,0.1), 100);

	std::shared_ptr<Object> o2 = make_shared<Object>(RES_DIR + "models/sphere2.obj");
	o2->setMaterial(glm::vec3(1,0,0), glm::vec3(0.1,0.1,0.1), glm::vec3(0.1,0.1,0.1), 100);
	o2->setPosition(glm::vec3(0.0f, -1.0f, 0.0f));

	// Add to scene
	scn.addObject(target);
	scn.addObject(o2);
}


int main(int argc, char **argv) {
	string filename = (argc >= 2) ? argv[1] : "out.png"; // optional name of output file
	int resolution = (argc >= 3) ? atoi(argv[2]) : 512; // optional size of output file

	// Create Scene
	Scene scn;
	createScene(scn);

	// Initialize Raytracer
	// Raytracer tracer (filename, resolution);
	// tracer.setScene(scn);
	// tracer.render();

	// Initialize Occluder
	Occluder occluder (filename, resolution);
	occluder.setScene(scn);
	occluder.render();
	// occluder.renderTexture();
	return 0;
}
