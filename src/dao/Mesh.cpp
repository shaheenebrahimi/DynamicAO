#include "Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#define DEG_TO_RAD 3.1415/180.0

#include "tiny_obj_loader.h"
#include <cuda_gl_interop.h>
#include <glm/gtx/euler_angles.hpp>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

/* ---------------------------------------- PUBLIC ---------------------------------------- */

Mesh::Mesh() {
	this->elemBufID = 0;
	this->posBufID = 0;
	this->norBufID = 0;
	this->texBufID = 0;
	this->occBufID = 0;
	this->boneCount = 0;
	this->influences = 0;
	this->currFrame = 0;
	this->cudaOccResource = nullptr;
}

Mesh::Mesh(const std::string name) {
	this->elemBufID = 0;
	this->posBufID = 0;
	this->norBufID = 0;
	this->texBufID = 0;
	this->occBufID = 0;
	this->boneCount = 0;
	this->influences = 0;
	this->currFrame = 0;
	this->cudaOccResource = nullptr;
}

Mesh::~Mesh() { }

// MESH HANDLERS

void Mesh::updateMesh()
{
	// update frame and pose
	computeOcclusion();
}

void Mesh::dumpMesh(const std::string &filename, const std::vector<std::string> &header)
{
	ofstream out;
	out.open(filename);
	if (!out.good()) {
		cout << "Cannot open " << filename << endl;
		return;
	}
	for (int i = 0; i < header.size(); ++i) {
		out << "# " << header[i] << "\n";
	}
	for (int i = 0; i < skPosBuf.size() / 3; ++i) {
		out << "v " << skPosBuf[3 * i] << " " << skPosBuf[3 * i + 1] << " " << skPosBuf[3 * i + 2] << "\n";
	}
	for (int i = 0; i < skNorBuf.size() / 3; ++i) {
		out << "vn " << skNorBuf[3 * i] << " " << skNorBuf[3 * i + 1] << " " << skNorBuf[3 * i + 2] << "\n";
	}
	for (int i = 0; i < texBuf.size() / 2; ++i) {
		out << "vt " << texBuf[2 * i] << " " << texBuf[2 * i + 1] << "\n";
	}
	for (int i = 0; i < elemBuf.size() / 3; ++i) {
		int v1 = elemBuf[3 * i] + 1, v2 = elemBuf[3 * i + 1] + 1, v3 = elemBuf[3 * i + 2] + 1;
		out << "f " << v1 << "/" << v1 << "/" << v1 << " "
					<< v2 << "/" << v2 << "/" << v2 << " "
					<< v3 << "/" << v3 << "/" << v3 << "\n";
	}
	out.close();
}

void Mesh::drawMesh(std::shared_ptr<Program> prog) {
    // Bind position buffer
	int h_pos = prog->getAttribute("aPos");
	glEnableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	
	// Bind normal buffer
	int h_nor = prog->getAttribute("aNor");
	if(h_nor != -1 && norBufID != 0) {
		glEnableVertexAttribArray(h_nor);
		glBindBuffer(GL_ARRAY_BUFFER, norBufID);
		glVertexAttribPointer(h_nor, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	}
	
	// Bind texcoords buffer
	int h_tex = prog->getAttribute("aTex");
	if(h_tex != -1 && texBufID != 0) {
		glEnableVertexAttribArray(h_tex);
		glBindBuffer(GL_ARRAY_BUFFER, texBufID);
		glVertexAttribPointer(h_tex, 2, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	}

	// Bind occlusion buffer
	int h_occ = prog->getAttribute("aOcc");
	if (h_occ != -1 && occBufID != 0) {
		glEnableVertexAttribArray(h_occ);
		glBindBuffer(GL_ARRAY_BUFFER, occBufID);
		glVertexAttribPointer(h_occ, 1, GL_FLOAT, GL_FALSE, 0, (const void*)0);
	}

	// Draw
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemBufID);
	glDrawElements(GL_TRIANGLES, (int)elemBuf.size(), GL_UNSIGNED_INT, (const void *)0);
	
	// Disable and unbind
	if (h_pos != -1) {
		glDisableVertexAttribArray(h_pos);
	}
	if(h_nor != -1) {
		glDisableVertexAttribArray(h_nor);
	}
	if (h_tex != -1) {
		glDisableVertexAttribArray(h_tex);
	}
	if (h_occ != -1) {
		glDisableVertexAttribArray(h_occ);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	
	GLSL::checkError(GET_FILE_LINE);
}

std::vector<float> Mesh::getFlattenedRotations()
{
	std::vector<float> flattened;
	for (const auto& r : relativeRotations) {
		//flattened.push_back(r.w);
		flattened.push_back(r.x);
		flattened.push_back(r.y);
		flattened.push_back(r.z);
	}
	return flattened;
}


// LOADERS

void Mesh::loader(const std::string& dir, const std::string& name)
{
	// load all data and set bind pose as initial pose
	loadMesh(dir + name + "/" + name + ".obj");
	loadSkeleton(dir + name + "/" + name + "_skeleton.txt");
	loadHierarchy(dir + name + "/" + name + "_hierarchy.txt");
	loadLocalTransforms(dir + name + "/" + name + "_static_transforms.txt");
	loadLocalSkeleton(dir + name + "/" + name + "_bind_pose.txt");
	loadSkinWeights(dir + name + "/" + name + "_skin.txt");

	//traverseHierarchy();
	computeBoneTransforms();
	applySkinning();
}

void Mesh::loadMesh(const string& meshName) {
	// Load geometry
	// This works only if the OBJ file has the same indices for v/n/t.
	// In other words, the 'f' lines must look like:
	// f 70/70/70 41/41/41 67/67/67
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;
	string errStr;
	bool rc = tinyobj::LoadObj(&attrib, &shapes, &materials, &errStr, meshName.c_str());
	if (!rc) {
		cerr << errStr << endl;
	}
	else {
		posBuf = attrib.vertices;
		norBuf = attrib.normals;
		texBuf = attrib.texcoords;

		// Loop over shapes
		for (size_t s = 0; s < shapes.size(); s++) {
			// Loop over faces (polygons)

			size_t index_offset = 0;
			for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
				size_t fv = shapes[s].mesh.num_face_vertices[f];
				// Loop over vertices in the face.
				for (size_t v = 0; v < fv; v++) {
					// access to vertex
					tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
					elemBuf.push_back(idx.vertex_index);
				}
				index_offset += fv;
			}
		}
		skPosBuf.resize(posBuf.size());
		skNorBuf.resize(norBuf.size());
	}
}

void Mesh::loadSkeleton(const std::string& filename)
{
	ifstream in;
	in.open(filename);
	if (!in.good()) {
		cout << "Cannot read " << filename << endl;
		return;
	}
	// cout << "Loading " << filename << endl;

	string line;
	stringstream ss;

	// Ignore comment header
	while (line.empty() || line.at(0) == '#') {
		getline(in, line);
	}

	// Get meta data
	int frameCount;
	ss = stringstream(line);
	ss >> frameCount; ss >> boneCount;
	jointPositions.resize(boneCount);

	getline(in, line);
	ss = stringstream(line);
	for (int bone = 0; bone < boneCount; ++bone) {
		float qx, qy, qz, qw, px, py, pz;
		ss >> qx; ss >> qy; ss >> qz; ss >> qw; ss >> px; ss >> py; ss >> pz;
		jointPositions[bone] = glm::vec3(px, py, pz);
	}
	in.close();
}

void Mesh::loadLocalTransforms(const std::string& filename)
{
	// Each matrix is 7 numbers: 4 for quaternion(x, y, z, w) and 3 for position(x, y, z), so that each line has 7 * 8 = 56 numbers.
	ifstream in;
	in.open(filename);
	if (!in.good()) {
		cout << "Cannot read " << filename << endl;
		return;
	}
	// cout << "Loading " << filename << endl;

	string line;
	stringstream ss;

	// Ignore comment header
	while (line.empty() || line.at(0) == '#') {
		getline(in, line);
	}

	// Get meta data
	ss = stringstream(line);
	ss >> this->boneCount;

	// WorldTransform = ParentWorldTransform * T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1
	// 8 matrices: T Roff Rp Rpre Rpost Soff Sp S
	boneTransforms.resize(boneCount, std::vector<glm::mat4>(8));
	for (int bone = 0; bone < boneCount; ++bone) {
		getline(in, line);
		ss = stringstream(line);
		for (int mat = 0; mat < 8; ++mat) {
			float qx, qy, qz, qw;
			ss >> qx; ss >> qy; ss >> qz; ss >> qw;
			glm::quat q (qw, qx, qy, qz);

			float px, py, pz;
			ss >> px; ss >> py; ss >> pz;
			glm::vec3 p (px, py, pz);

			boneTransforms[bone][mat] = glm::translate(glm::mat4(1.0), p) * glm::mat4_cast(q);
		}
	}

	in.close();
}


void Mesh::loadLocalSkeleton(const std::string& filename)
{
	ifstream in;
	in.open(filename);
	if (!in.good()) {
		cout << "Cannot read " << filename << endl;
		return;
	}
	// cout << "Loading " << filename << endl;

	string line;
	stringstream ss;

	// Ignore comment header
	while (line.empty() || line.at(0) == '#') {
		getline(in, line);
	}

	// Get meta data
	ss = stringstream(line);
	ss >> this->frameCount; ss >> this->boneCount;
	bindPose.resize(boneCount);

	getline(in, line);
	ss = stringstream(line);

	float px, py, pz;
	std::vector<glm::vec3> rotations(boneCount);
	for (int bone = 0; bone < boneCount; ++bone) { // ignore bind pose
		float rx, ry, rz;
		ss >> rx; ss >> ry; ss >> rz;
		rotations[bone] = glm::vec3(rx, ry, rz);
		if (bone == 0) {
			ss >> px; ss >> py; ss >> pz;
			boneTransforms[bone][0] = glm::translate(glm::mat4(1.0), glm::vec3(px, py, pz));
		}
		//cout << bone << ": " << rx << " " << ry << " " << rz << " -> " << glm::to_string(eulerToQuaternion(rx, ry, rz)) << endl;
	}

	bindPose = computeAbsolutePose(rotations);
	in.close();
}

void Mesh::loadSkinWeights(const std::string& filename)
{
	ifstream in;
	in.open(filename);
	if (!in.good()) {
		cout << "Cannot read " << filename << endl;
		return;
	}
	// cout << "Loading " << filename << endl;

	string line;
	stringstream ss;

	// Ignore comment header
	while (line.empty() || line.at(0) == '#') {
		getline(in, line);
	}

	// Get meta data
	int verts, bones;
	ss = stringstream(line);
	ss >> verts; ss >> bones; ss >> influences;
	skBoneInds.resize(verts * influences, 0);
	skWeights.resize(verts * influences, 0);

	// Get bone weights per vertex
	for (unsigned int vert = 0; vert < verts; ++vert) {
		getline(in, line);
		ss = stringstream(line);
		int vertInfluences;
		ss >> vertInfluences;
		for (int i = 0; i < vertInfluences; ++i) {
			ss >> skBoneInds[vert * influences + i];
			ss >> skWeights[vert * influences + i];
		}
	}
	in.close();
}

void Mesh::loadHierarchy(const std::string& filename)
{
	ifstream in;
	in.open(filename);
	if (!in.good()) {
		cout << "Cannot read " << filename << endl;
		return;
	}
	// cout << "Loading " << filename << endl;

	string line;
	stringstream ss;

	// Ignore comment header
	while (line.empty() || line.at(0) == '#') {
		getline(in, line);
	}

	// Get meta data
	ss = stringstream(line);
	ss >> boneCount;

	// Get hierarchy data per line
	boneHierarchy.resize(boneCount);
	for (int i = 0; i < boneCount; ++i) {
		unsigned int jointInd, parentInd; string rotOrder, jointName;
		getline(in, line);
		ss = stringstream(line);
		ss >> jointInd; ss >> parentInd; ss >> rotOrder; ss >> jointName;
		boneHierarchy[jointInd] = parentInd;
		boneMap[jointName] = jointInd;
	}

	// populate hierarchy
	relativeTranslations.resize(boneCount);
	relativeRotations.resize(boneCount);
	for (int i = 0; i < boneCount; ++i) { // iter through bone index
		relativeTranslations[i] = (i == 0) ? jointPositions[i] : jointPositions[i] - jointPositions[boneHierarchy[i]]; // abs pos = rel pos if root
		//relativeRotations[i] = glm::quat(1.0, 0.0, 0.0, 0.0);
		relativeRotations[i] = glm::vec3(0.0, 0.0, 0.0);
	}
	in.close();
}

void Mesh::loadEvaluator(const string& modelName) {
	eval = std::make_shared<Evaluator>(modelName);
}

void Mesh::loadBuffers() {

	// Send the position array to the GPU
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_STATIC_DRAW);

	// Send the normal array to the GPU
	if (!norBuf.empty()) {
		glGenBuffers(1, &norBufID);
		glBindBuffer(GL_ARRAY_BUFFER, norBufID);
		glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_STATIC_DRAW);
	}

	// Send the texture array to the GPU
	if (!texBuf.empty()) {
		glGenBuffers(1, &texBufID);
		glBindBuffer(GL_ARRAY_BUFFER, texBufID);
		glBufferData(GL_ARRAY_BUFFER, texBuf.size() * sizeof(float), &texBuf[0], GL_STATIC_DRAW);
	}

	// Send the element array to the GPU
	glGenBuffers(1, &elemBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, elemBuf.size() * sizeof(unsigned int), &elemBuf[0], GL_STATIC_DRAW);

	// Send occlusion array to GPU
	createCudaVBO(&occBufID, &cudaOccResource, cudaGraphicsMapFlagsWriteDiscard, posBuf.size() / 3);

	// Unbind the arrays
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	GLSL::checkError(GET_FILE_LINE);
}


// SKINNING

void Mesh::setAnimation(const std::string& filename) {
	ifstream in;
	in.open(filename);
	if (!in.good()) {
		cout << "Cannot read " << filename << endl;
		return;
	}
	// cout << "Loading " << filename << endl;

	string line;
	stringstream ss;

	// Ignore comment header
	while (line.empty() || line.at(0) == '#') {
		getline(in, line);
	}

	// Get meta data
	ss = stringstream(line);
	ss >> this->frameCount; ss >> this->boneCount;
	bindPose.resize(boneCount);
	this->frameData.resize(frameCount, vector<glm::vec3>(boneCount));

	for (int frame = 0; frame < frameCount; ++frame) {
		getline(in, line);
		ss = stringstream(line);
		float px, py, pz;
		for (int bone = 0; bone < boneCount; ++bone) { // ignore bind pose
			float rx, ry, rz;
			ss >> rx; ss >> ry; ss >> rz;
			frameData[frame][bone] = glm::vec3(rx, ry, rz); // relative euler angle per frame
			if (bone == 0) {
				ss >> px; ss >> py; ss >> pz; // TODO: add support for movement of mesh (set pos of root to x, y, z)
				boneTransforms[bone][0] = glm::translate(glm::mat4(1.0), glm::vec3(px, py, pz));
			}
		}
	}

	this->currFrame = 0;
	setPose(frameData[this->currFrame]);

	in.close();
}


void Mesh::setPose(const std::vector<glm::vec3>& relOrientations) {
	assert(relOrientations.size() == boneCount);


	relativeRotations = relOrientations;

	// set matrix representation of pose
	//traverseHierarchy();
	computeBoneTransforms();

	// deform mesh with skinning
	applySkinning();
}

void Mesh::setBone(const int boneInd, glm::vec3 relOrientation) {
	assert(boneInd >= 0 && boneInd < boneCount);
	relativeRotations[boneInd] = relOrientation;

	// set matrix representation of pose
	//traverseHierarchy();
	computeBoneTransforms();


	// deform mesh with skinning
	applySkinning();
}

void Mesh::traverseHierarchy()
{
	pose.resize(boneCount);
	for (int i = 0; i < boneCount; ++i) {
		glm::mat4 M(1);
		int ind = i;
		do {
			//glm::mat4 Mi = glm::translate(glm::mat4(1), relativeTranslations[ind]) * glm::mat4_cast(relativeRotations[ind]); // T * R
			glm::mat4 T = glm::translate(glm::mat4(1), relativeTranslations[ind]);
			glm::mat4 R = glm::eulerAngleZ(relativeRotations[ind].z) * glm::eulerAngleY(relativeRotations[ind].y) * glm::eulerAngleX(relativeRotations[ind].x);
			glm::mat4 Mi = T * R; // T * R
			M = Mi * M;
			ind = boneHierarchy[ind]; // go down hierarchy
		} while (ind != -1);
		pose[i] = M;
	}
}

std::vector<glm::mat4> Mesh::computeAbsolutePose(std::vector<glm::vec3>& relRotation)
{
	// WorldTransform = ParentWorldTransform * T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1
	// boneTransforms: T Roff Rp Rpre Rpost Soff Sp S
	std::vector<glm::mat4> absPose(boneCount);
	for (int i = 0; i < boneCount; ++i) {
		glm::mat4 M(1);
		int ind = i;
		do {
			glm::mat4 T = boneTransforms[ind][0];
			glm::mat4 Roff = boneTransforms[ind][1];
			glm::mat4 Rp = boneTransforms[ind][2];
			glm::mat4 Rpre = boneTransforms[ind][3];
			glm::mat4 Rpost = boneTransforms[ind][4];
			glm::mat4 Soff = boneTransforms[ind][5];
			glm::mat4 Sp = boneTransforms[ind][6];
			glm::mat4 S = boneTransforms[ind][7];
			glm::mat4 R = glm::eulerAngleZ(relativeRotations[ind].z) * glm::eulerAngleY(relativeRotations[ind].y) * glm::eulerAngleX(relativeRotations[ind].x);
			glm::mat4 Mi = T * Roff * Rp * Rpre * R * glm::inverse(Rpost) * glm::inverse(Rp) * Soff * Sp * S * glm::inverse(Sp);
			M = Mi * M;
			ind = boneHierarchy[ind]; // go down hierarchy
		} while (ind != -1);
		absPose[i] = M;
	}
	return absPose;
}

void Mesh::computeBoneTransforms()
{
	// WorldTransform = ParentWorldTransform * T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1
	// boneTransforms: T Roff Rp Rpre Rpost Soff Sp S
	pose.resize(boneCount);
	for (int i = 0; i < boneCount; ++i) {
		glm::mat4 M(1);
		int ind = i;
		do {
			glm::mat4 T = boneTransforms[ind][0];
			glm::mat4 Roff = boneTransforms[ind][1];
			glm::mat4 Rp = boneTransforms[ind][2];
			glm::mat4 Rpre = boneTransforms[ind][3];
			glm::mat4 Rpost = boneTransforms[ind][4];
			glm::mat4 Soff = boneTransforms[ind][5];
			glm::mat4 Sp = boneTransforms[ind][6];
			glm::mat4 S = boneTransforms[ind][7];
			glm::mat4 R = glm::eulerAngleZ(relativeRotations[ind].z) * glm::eulerAngleY(relativeRotations[ind].y) * glm::eulerAngleX(relativeRotations[ind].x);
			glm::mat4 Mi = T * Roff * Rp * Rpre * R * glm::inverse(Rpost) * glm::inverse(Rp) * Soff * Sp * S * glm::inverse(Sp);
			M = Mi * M;
			ind = boneHierarchy[ind]; // go down hierarchy
		} while (ind != -1);
		pose[i] = M;
	}
}

void Mesh::applySkinning()
{
	// Linear Blend Skinning
	for (int i = 0; i < posBuf.size() / 3; ++i) { // iterate through vertices
		glm::vec4 x0(posBuf[3 * i], posBuf[3 * i + 1], posBuf[3 * i + 2], 1); // world mesh pos
		glm::vec4 n0(norBuf[3 * i], norBuf[3 * i + 1], norBuf[3 * i + 2], 0); // world mesh nor
		glm::mat4 weightedTrans(0);
		for (int j = 0; j < influences; ++j) { // iterate through bone influences
			unsigned int boneInd = skBoneInds[influences * i + j];
			glm::mat4 M0 = bindPose[boneInd]; // bind pose bone space to world
			glm::mat4 Mk = pose[boneInd];
			float w = skWeights[influences * i + j]; // weight of bone on vert
			glm::mat4 T = Mk * glm::inverse(M0); // linear skinning eq
			weightedTrans += T * w;
		}
		glm::vec4 xk = weightedTrans * x0;
		glm::vec4 nk = glm::normalize(weightedTrans * n0);
		skPosBuf[3 * i] = xk.x; skPosBuf[3 * i + 1] = xk.y; skPosBuf[3 * i + 2] = xk.z; // set skinned mesh
		skNorBuf[3 * i] = nk.x; skNorBuf[3 * i + 1] = nk.y; skNorBuf[3 * i + 2] = nk.z;
	}
}

void Mesh::stepAnimation() {
	this->currFrame = (currFrame + 1) % frameCount;
	setPose(frameData[currFrame]);
}

/* ---------------------------------------- PRIVATE ---------------------------------------- */

// OCCLUSION

void Mesh::computeOcclusion()
{
	Batch inputs = getBatch();
	eval->sharedBatchCompute(inputs, &cudaOccResource);
}

void Mesh::createCudaVBO(GLuint* vbo, cudaGraphicsResource** vboRes, unsigned int vboResFlags, unsigned int size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(vboRes, *vbo, vboResFlags);
}

void Mesh::deleteCudaVBO(GLuint* vbo, cudaGraphicsResource* vboRes)
{
	// unregister this buffer object with CUDA
	//checkCudaErrors(cudaGraphicsUnregisterResource(vboRes));
	cudaGraphicsUnregisterResource(vboRes);

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

// HELPERS

Batch Mesh::getBatch()
{
	std::vector<float> rotations = getFlattenedRotations();
	Batch inputs (Shape(1, rotations.size()), { rotations });
	return inputs;
}