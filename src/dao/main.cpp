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

std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0)
std::default_random_engine generator;

/* HELPERS */

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
	obj->addEvaluator(RES_DIR + "evaluators/" + name + ".txt");
	obj->setMaterial(glm::vec3(0.2, 0.2, 0.2), glm::vec3(0.1, 0.1, 0.1), glm::vec3(0.1, 0.1, 0.1), 100);
	scn.addObject(obj); // Add to scene

	return scn;
}

//#define _ALL_BONES_

/* DATA GENERATORS */
void generateDatasetMeshes(const string& meshname, const int numSamples) {
	Mesh mesh;
	mesh.loader(RES_DIR + "models/", meshname);

	// iterate through all bones
	float lowerBound = -80.0 * DEG_TO_RAD, upperBound = 80.0 * DEG_TO_RAD;
	long train_counter = 246;
	long test_counter = 54;

	
#ifdef _ALL_BONES
	for (bone = 1; bone < mesh.getBoneCount(); ++bone) { // ignore first bone since root
#else
	int bone = 12;
#endif
		for (int sample = 0; sample < numSamples; ++sample) {

			bool is_train = (randomFloats(generator) <= 0.8); // 80% train, 20% test
			float x = lowerBound + randomFloats(generator) * (upperBound - lowerBound);
			float y = lowerBound + randomFloats(generator) * (upperBound - lowerBound);
			float z = lowerBound + randomFloats(generator) * (upperBound - lowerBound);
			mesh.setBone(bone, glm::vec3(x, y, z)); // relative euler angle x, y, z

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

			mesh.dumpMesh(RES_DIR + "data/_" + meshname + (is_train ? "_train_" + to_string(train_counter++) : "_test_" + to_string(test_counter++)) + ".obj", header);
		}
//#ifdef _ALL_BONES
//	for (bone = 1; bone < mesh.getBoneCount(); ++bone) { // ignore first bone since root
//#endif
//		for (float x = lowerBound; x < upperBound; x += increment) {
//			for (float y = lowerBound; y < upperBound; y += increment) {
//				for (float z = lowerBound; z < upperBound; z += increment) {
//					bool is_train = (randomFloats(generator) <= 0.8); // 80% train, 20% test
//
//					mesh.setBone(bone, glm::vec3(x, y, z)); // relative euler angle x, y, z
//
//					// convert to printable string
//					vector<float> buffer = mesh.getFlattenedRotations();
//					string values = "";
//					for (int i = 0; i < buffer.size(); ++i) {
//						values += to_string(buffer[i]) + " ";
//					}
//					vector<string> header = {
//						"The next comment says how many bones and their orientations",
//						to_string(mesh.getBoneCount()),
//						values
//					};
//
//					mesh.dumpMesh(RES_DIR + "data/_" + meshname + (is_train ? "_train_" + to_string(train_counter++) : "_test_" + to_string(test_counter++)) + ".obj", header);
//				}
//			}
//		}
#ifdef _ALL_BONES
	}
#endif

}

void generateAnimatedMeshes(const string& meshname, int step=1) {

	// TODO: get orientations from animation files
	Mesh mesh;
	mesh.loader(RES_DIR + "models/", meshname);
	
	const string animPath = RES_DIR + "models/" + meshname + "/animations/";
	const bool is_train = false;
	int animCount = 7;

	// iterate through animations
	int counter = 0;
	for (int i = 0; i < animCount; ++i) {

		mesh.setAnimation(animPath + "anim" + (is_train ? "_train_" : "_test_") + to_string(i) + ".txt");

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
			mesh.dumpMesh(RES_DIR + "data/" + meshname + (is_train ? "_train_" : "_test_") + to_string(counter++) + ".obj", header);
			mesh.setFrame(j);
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
	Rasterizer raster; // Initialize Rasterizer
	raster.setScene(scn);
	raster.init();
	raster.run();

	//int step = 2;
	//generateAnimatedMeshes(name, step); // go to 100 degrees with 1 degree increments
	//generateDatasetMeshes(name, 900);

	return 0;
}

// TODO:
// Reduce to 1 joint space
// Rotate each joint to create augmented data
// Fix visualizer
// ML stuff: tensorboard, adam optimizer, adjust learning rate