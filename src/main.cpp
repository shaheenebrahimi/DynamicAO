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
#include "Raytracer.h"
#include "Occluder.h"


#include <glm/glm.hpp>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <random>

using namespace std;

const string RES_DIR = "../resources/";

Mesh* target;

void createScene(Scene& scn) {
	// Lights
	scn.addLight(new Light(glm::vec3(1.0f, 2.0f, 2.0f), 0.5f));
	scn.addLight(new Light(glm::vec3(-1.0f, 2.0f, -1.0f), 0.5f));

	// Objects
	target = new Mesh(RES_DIR + "models/sphere2.obj", new BlinnPhong(glm::vec3(0,0,1),glm::vec3(0.1,0.1,0.1),glm::vec3(0.1,0.1,0.1),100));
	// Mesh* floor = new Mesh(RES_DIR + "models/square.obj");
	// floor.position = glm::vec3(0, -1)

	// Add to scene
    scn.addShape(target);
	// scn.addShape(floor);
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