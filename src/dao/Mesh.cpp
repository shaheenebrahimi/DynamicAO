#include "Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

/* PUBLIC  */
Mesh::Mesh() {
    this->transform = glm::mat4(1);
}

Mesh::Mesh(const string &meshName) {
    this->transform = glm::mat4(1);
    loadMesh(meshName); // load obj and populate the triangles
}

Mesh::~Mesh() {
	//triangles.clear();
	//transformed.clear();
}

void Mesh::loader(const std::string& dir, const std::string& name)
{
	loadMesh(dir + name + "/" + name + ".obj");
	loadSkeleton(dir + name + "/" + name + "_skeleton.txt");
	loadHierarchy(dir + name + "/" + name + "_hierarchy.txt");
	loadSkinWeights(dir + name + "/" + name + "_skin.txt");
}

void Mesh::loadMesh(const string &meshName) {
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;
	string errStr;
	bool rc = tinyobj::LoadObj(&attrib, &shapes, &materials, &errStr, meshName.c_str());
	if(!rc) {
		cerr << errStr << endl;
	} else {
		posBuf = attrib.vertices;
		norBuf = attrib.normals;
		texBuf = attrib.texcoords;

		// Loop over shapes
		for(size_t s = 0; s < shapes.size(); s++) {
			// Loop over faces (polygons)
			size_t index_offset = 0;
			for(size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
				size_t fv = shapes[s].mesh.num_face_vertices[f];
				// Loop over vertices in the face.
				for(size_t v = 0; v < fv; v++) {
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
	bindPose.resize(boneCount);

	getline(in, line);
	ss = stringstream(line);
	for (int bone = 0; bone < boneCount; ++bone) {
		float qx, qy, qz, qw, px, py, pz;
		ss >> qx; ss >> qy; ss >> qz; ss >> qw; ss >> px; ss >> py; ss >> pz;
		bindPose[bone] = glm::vec3(px, py, pz);
	}
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
		relativeTranslations[i] = (i == 0) ? bindPose[i] : bindPose[i] - bindPose[boneHierarchy[i]]; // abs pos = rel pos if root
		relativeRotations[i] = glm::quat(1.0, 0.0, 0.0, 0.0);
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
	glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_STATIC_DRAW);

	// Send the normal array to the GPU
	if(!norBuf.empty()) {
		glGenBuffers(1, &norBufID);
		glBindBuffer(GL_ARRAY_BUFFER, norBufID);
		glBufferData(GL_ARRAY_BUFFER, norBuf.size()*sizeof(float), &norBuf[0], GL_STATIC_DRAW);
	}
	
	// Send the texture array to the GPU
	if(!texBuf.empty()) {
		glGenBuffers(1, &texBufID);
		glBindBuffer(GL_ARRAY_BUFFER, texBufID);
		glBufferData(GL_ARRAY_BUFFER, texBuf.size()*sizeof(float), &texBuf[0], GL_STATIC_DRAW);
	}

	// Send occlusion array to GPU
	createCudaVBO(&occBufID, &cudaOccResource, cudaGraphicsMapFlagsWriteDiscard, posBuf.size()/3);

	// Unbind the arrays
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	GLSL::checkError(GET_FILE_LINE);
}

void Mesh::setBoneAngles(const std::vector<float>& thetas)
{
	// convert to quaternions
	// for (int i = 0; i < relativeRotations.size(); ++i) {
	// 	relativeRotations[i] = glm::quat(cos(thetas[i]/2), 0, sin(thetas[i]/2), 0); // TOFIX: hard coded to rotate about y
	// }
	// assert(thetas.size() == boneCount);

	// convert to pose
	std::vector<glm::mat4> pose;
	traverseHierarchy(pose);
	for (int i = 0; i < boneCount; ++i) {
		cout << glm::to_string(pose[i]) << endl;
	}

	// apply pose
	applyPose(pose);
}

/* PRIVATE */

void Mesh::createCudaVBO(GLuint *vbo, cudaGraphicsResource **vboRes, unsigned int vboResFlags, unsigned int size)
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

void Mesh::deleteCudaVBO(GLuint *vbo, cudaGraphicsResource *vboRes)
{
	// unregister this buffer object with CUDA
	//checkCudaErrors(cudaGraphicsUnregisterResource(vboRes));
	cudaGraphicsUnregisterResource(vboRes);

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

void Mesh::computeOcclusion()
{
	Batch inputs = getInputs(); // TODO: fix to get proper inputs
	eval->sharedBatchCompute(inputs, &cudaOccResource);
}

void Mesh::dumpMesh(const std::string &filename)
{
	ofstream out;
	out.open(filename);
	if (!out.good()) {
		cout << "Cannot open " << filename << endl;
		return;
	}
	for (int i = 0; i < skPosBuf.size() / 3; ++i) {
		out << "v " << skPosBuf[3 * i] << " " << skPosBuf[3 * i + 1] << " " << skPosBuf[3 * i + 2] << "\n";
	}
	for (int i = 0; i < skNorBuf.size() / 3; ++i) {
		out << "vn " << skNorBuf[3 * i] << " " << skNorBuf[3 * i + 1] << " " << skNorBuf[3 * i + 2] << "\n";
	}
	//for (int i = 0; i < posBuf.size() / 3; ++i) {
	//	out << "v " << posBuf[3 * i] << " " << posBuf[3 * i + 1] << " " << posBuf[3 * i + 2] << "\n";
	//}
	//for (int i = 0; i < norBuf.size() / 3; ++i) {
	//	out << "vn " << norBuf[3 * i] << " " << norBuf[3 * i + 1] << " " << norBuf[3 * i + 2] << "\n";
	//}
	for (int i = 0; i < texBuf.size() / 2; ++i) {
		out << "vt " << texBuf[2 * i] << " " << texBuf[2 * i + 1] << "\n";
	}
	for (int i = 0; i < elemBuf.size() / 3; ++i)
		out << "f " << elemBuf[3 * i] + 1 << " " << elemBuf[3 * i + 1] + 1 << " " << elemBuf[3 * i + 2] + 1 << "\n";
	out.close();
}

void Mesh::updateMesh()
{
	// TODO: skinning
	computeOcclusion();
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

	int h_occ = prog->getAttribute("aOcc");
	if (h_occ != -1 && occBufID != 0) {
		glEnableVertexAttribArray(h_occ);
		glBindBuffer(GL_ARRAY_BUFFER, occBufID);
		glVertexAttribPointer(h_occ, 1, GL_FLOAT, GL_FALSE, 0, (const void*)0);
	}

	// Draw
	int count = posBuf.size()/3; // number of indices to be rendered
	glDrawArrays(GL_TRIANGLES, 0, count);
	
	// Disable and unbind
	if(h_tex != -1) {
		glDisableVertexAttribArray(h_tex);
	}
	if(h_nor != -1) {
		glDisableVertexAttribArray(h_nor);
	}
	glDisableVertexAttribArray(h_occ);
	glDisableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	GLSL::checkError(GET_FILE_LINE);
}

void Mesh::traverseHierarchy(std::vector<glm::mat4>& pose)
{
	pose.resize(boneCount);
	for (int i = 0; i < boneCount; ++i) {
		glm::mat4 M(1);
		int ind = i;
		do {
			glm::mat4 Mi = glm::translate(glm::mat4(1), relativeTranslations[ind]) * glm::mat4_cast(relativeRotations[ind]); // T * R
			M = Mi * M;
			ind = boneHierarchy[ind]; // go down hierarchy
		} while (ind != -1);
		pose[i] = M;
	}
}

void Mesh::applyPose(const std::vector<glm::mat4>& pose)
{
	// Linear Blend Skinning
	for (int i = 0; i < posBuf.size() / 3; ++i) { // iterate through vertices
		glm::vec4 x0(posBuf[3 * i], posBuf[3 * i + 1], posBuf[3 * i + 2], 1); // world mesh pos
		glm::vec4 n0(norBuf[3 * i], norBuf[3 * i + 1], norBuf[3 * i + 2], 0); // world mesh nor
		glm::mat4 weightedTrans(0);
		for (int j = 0; j < influences; ++j) { // iterate through bone influences
			unsigned int boneInd = skBoneInds[influences * i + j];
			glm::mat4 M0 = glm::translate(glm::mat4(1), bindPose[boneInd]); // bind pose bone space to world
			glm::mat4 Mk = pose[boneInd];
			float w = skWeights[influences * i + j]; // weight of bone on vert
			glm::mat4 T = Mk * glm::inverse(M0); // linear skinning eq
			weightedTrans += T * w;
		}
		glm::vec4 xk = weightedTrans * x0;
		glm::vec4 nk = glm::normalize(weightedTrans * n0);
		skPosBuf[3 * i] = xk.x; skPosBuf[3 * i + 1] = xk.y; skPosBuf[3 * i + 2] = xk.z;
		skNorBuf[3 * i] = nk.x; skNorBuf[3 * i + 1] = nk.y; skNorBuf[3 * i + 2] = nk.z;
	}
}

Batch Mesh::getInputs()
{
	std::vector<std::vector<float>> data;
	for (int i = 0; i < texBuf.size(); i += 2) {
		data.push_back({ texBuf[i], texBuf[i + 1] });
	}
	Batch inputs (Shape(1,2), data);
	return inputs;
}