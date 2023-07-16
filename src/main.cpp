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

int main(int argc, char **argv)
{
	string filename = (argc >= 2) ? argv[1] : "aoTexture.png"; // optional name of output file
	int resolution = (argc >= 3) ? atoi(argv[2]) : 1024; // optional size of output file
	
	return 0;
}
