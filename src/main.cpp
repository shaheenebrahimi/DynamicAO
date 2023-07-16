#include "Image.h"
#include "Camera.h"
#include "Scene.h"
#include "Ray.h"
#include "BlinnPhong.h"
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

Object target;


void createScene(Scene& scn) {
	// Lights
	scn.addLight(Light(glm::vec3(1.0f, 2.0f, 2.0f), 0.5f));
	scn.addLight(Light(glm::vec3(-1.0f, 2.0f, -1.0f), 0.5f));

	// Objects
	target = Object(RES_DIR + "models/sphere2.obj");
	Object floor (RES_DIR + "models/square.obj");
	floor.move(glm::vec3(0.0f, -1.0f, 0.0f));

	// Add to scene
    scn.addShape(target);
	scn.addShape(floor);
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
