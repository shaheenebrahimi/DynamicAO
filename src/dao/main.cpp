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

std::uniform_real_distribution<float> randomFloats(-1.0, 1.0); // random floats between [0.0, 1.0]
std::default_random_engine generator;

/* HELPERS */

glm::quat eulerToQuaternion(float roll, float pitch, float yaw) { // roll (x), pitch (y), yaw (z), angles are in radians
	float cr = cos(roll * 0.5);
	float sr = sin(roll * 0.5);
	float cp = cos(pitch * 0.5);
	float sp = sin(pitch * 0.5);
	float cy = cos(yaw * 0.5);
	float sy = sin(yaw * 0.5);

	glm::quat q;
	q.w = cr * cp * cy + sr * sp * sy;
	q.x = sr * cp * cy - cr * sp * sy;
	q.y = cr * sp * cy + sr * cp * sy;
	q.z = cr * cp * sy - sr * sp * cy;

	return q;
}

glm::quat getRandomQuaternion() {
	double x, y, z, u, v, w, s;
	do { x = randomFloats(generator); y = randomFloats(generator); z = x * x + y * y; } while (z > 1);
	do { u = randomFloats(generator); v = randomFloats(generator); w = u * u + v * v; } while (w > 1);
	s = sqrt((1 - z) / w);
	return glm::quat(x, y, s * u, s * v);
}

/* INITIALIZER */

Scene createScene(const string &name) {
	Scene scn;

	// Lights
	std::shared_ptr<Light> l1 = make_shared<Light>(glm::vec3(1.0f, 2.0f, 2.0f), 0.5f);
	scn.addLight(l1);

	// Objects
	shared_ptr<Object> obj = make_shared<Object>();
	obj->addMesh(RES_DIR + "models/", name);
	obj->addTexture(RES_DIR + "textures/" + name + ".png");
	//obj->addEvaluator(RES_DIR + "evaluators/" + name + ".txt");
	obj->setMaterial(glm::vec3(0.2, 0.2, 0.2), glm::vec3(0.1, 0.1, 0.1), glm::vec3(0.1, 0.1, 0.1), 100);
	scn.addObject(obj); // Add to scene

	return scn;
}

/* DATA GENERATORS */

void generateTrainingMeshes(const string &meshname) {
	Mesh mesh;
	mesh.loader(RES_DIR + "models/", meshname);

	// iterate through all bones
	float lowerBound = -50.0, upperBound = 50.0, increment = 20.0; // in degrees

	// samples = boneCount * ((upperBound - lowerBound) / increment) ^ 3
	long counter = 0;
	for (int bone = 1; bone < mesh.getBoneCount(); ++bone) { // ignore first bone since root
		for (float yaw = lowerBound; yaw < upperBound; yaw += increment) {
			for (float roll = lowerBound; roll < upperBound; roll += increment) {
				for (float pitch = lowerBound; pitch < upperBound; pitch += increment) {

					// get quaternion and set orientation
					glm::quat orientation = eulerToQuaternion(yaw * DEG_TO_RAD, roll * DEG_TO_RAD, pitch * DEG_TO_RAD);
					//mesh.setBone(bone, orientation);

					// convert to printable string
					vector<float> buffer = mesh.getFlattenedRotations();
					string values = "";
					for (int i = 0; i < buffer.size(); ++i) {
						values += to_string(buffer[i]) + " ";
					}

					vector<string> header = {
						"The next comment says how many bones and their orientations",
						to_string(mesh.getBoneCount()),
						values
					};
					mesh.dumpMesh(RES_DIR + "data/" + meshname + to_string(counter++) + ".obj", header);
				}
			}
		}
	}

}

void generateAnimatedMeshes(const string& meshname, int step=1) {

	// TODO: get orientations from animation files
	Mesh mesh;
	mesh.loader(RES_DIR + "models/", meshname);
	
	string animPath = RES_DIR + "models/" + meshname + "/animations/";
	int animCount = 11;

	// iterate through animations
	int sample = 0;
	for (int i = 0; i < animCount; ++i) {
		mesh.setAnimation(animPath + "anim_" + to_string(i) + ".txt");

		//glm::vec3 b = mesh.getRotationData(11, 0); // 11 is arm
		//cout << b.x << " " << b.y << " " << b.z << endl;

		// iterate through frames
		for (int j = 0; j < mesh.getFrameCount(); j+=step) {
			// get comment header
			vector<float> rotations = mesh.getFlattenedRotations();
			string values = "";
			for (int r = 0; r < rotations.size(); ++r) {
				values += to_string(rotations[r]) + " ";
			}
			vector<string> header = {
				"The next comment says how many bones and their orientations in Euler angles (rx, ry, rz)",
				to_string(mesh.getBoneCount()),
				values
			};

			// dump mesh
			mesh.dumpMesh(RES_DIR + "data/" + meshname + to_string(sample) + ".obj", header);
			mesh.setFrame(j);
			sample++; // since j != sample if step != 1
		}
	}
	
}

/* MAIN */

int main(int argc, char **argv) {
	// TODO: fix so that doesn't need texture to run
	///* Initializations */
	string name = "warrior";
	Scene scn = createScene(name);

	// viewer
	//Rasterizer raster; // Initialize Rasterizer
	//raster.setScene(scn);
	//raster.init();
	//raster.run();

	int step = 2;
	generateAnimatedMeshes(name, step); // go to 100 degrees with 1 degree increments
	
	return 0;
}
