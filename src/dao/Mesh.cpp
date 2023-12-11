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

void Mesh::loadMesh(const string &meshName) {
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;
	string errStr;
	bool rc = tinyobj::LoadObj(&attrib, &shapes, &materials, &errStr, meshName.c_str());
	if(!rc) {
		cerr << errStr << endl;
	} else {
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
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+0]);
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+1]);
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+2]);
					if(!attrib.normals.empty()) {
						norBuf.push_back(attrib.normals[3*idx.normal_index+0]);
						norBuf.push_back(attrib.normals[3*idx.normal_index+1]);
						norBuf.push_back(attrib.normals[3*idx.normal_index+2]);
					}
					if(!attrib.texcoords.empty()) {
						texBuf.push_back(attrib.texcoords[2*idx.texcoord_index+0]);
						texBuf.push_back(attrib.texcoords[2*idx.texcoord_index+1]);
					}
					else {
						cerr << "Error: model does not have texture coords" << endl;
					}
				}
				index_offset += fv;
			}
		}
	}
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

/* PRIVATE */

//void Mesh::bufToTriangles() {
//    for (int i = 0; i < posBuf.size()/9; i++) {
//        glm::vec3 pos0 (posBuf[9*i], posBuf[9*i+1], posBuf[9*i+2]);
//        glm::vec3 pos1 (posBuf[9*i+3], posBuf[9*i+4], posBuf[9*i+5]);
//        glm::vec3 pos2 (posBuf[9*i+6], posBuf[9*i+7], posBuf[9*i+8]);
//        glm::vec3 nor0 (norBuf[9*i], norBuf[9*i+1], norBuf[9*i+2]);
//        glm::vec3 nor1 (norBuf[9*i+3], norBuf[9*i+4], norBuf[9*i+5]);
//        glm::vec3 nor2 (norBuf[9*i+6], norBuf[9*i+7], norBuf[9*i+8]);
//		glm::vec2 tex0 (texBuf[6*i], texBuf[6*i+1]);
//        glm::vec2 tex1 (texBuf[6*i+2], texBuf[6*i+3]);
//        glm::vec2 tex2 (texBuf[6*i+4], texBuf[6*i+5]);
//        shared_ptr<Triangle> tri = std::make_shared<Triangle>(pos0, pos1, pos2, nor0, nor1, nor2, tex0, tex1, tex2);
//        triangles.push_back(tri);
//    }
//}

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

Batch Mesh::getInputs()
{
	std::vector<std::vector<float>> data;
	for (int i = 0; i < texBuf.size(); i += 2) {
		data.push_back({ texBuf[i], texBuf[i + 1] });
	}
	Batch inputs (Shape(1,2), data);
	return inputs;
}