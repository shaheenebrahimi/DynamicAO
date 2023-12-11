#pragma once
#ifndef MESH_H
#define MESH_H

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <optional>

#include "GLSL.h"
#include "Material.h"
#include "Program.h"
#include "Evaluator.cuh"

#define NO_ROTATION glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);

class Mesh {
public:
    glm::mat4 transform;

    Mesh();
    Mesh(const std::string &objPath);
    ~Mesh();
    void loadMesh(const std::string &meshName);
    void loadEvaluator(const std::string& modelName);

    void loadBuffers(); // only for rasterization
    void setTransform(glm::mat4 transform) { this->transform = transform; }
    //std::vector<std::shared_ptr<Triangle>> getTriangles() { return transformed; } // return transformed tris
    void updateMesh();
    void drawMesh(std::shared_ptr<Program> prog);
private:
    //std::vector<std::shared_ptr<Triangle>> triangles;
    //std::vector<std::shared_ptr<Triangle>> transformed;

    std::shared_ptr<Evaluator> eval;

    std::vector<float> posBuf;
    std::vector<float> norBuf;
    std::vector<float> texBuf;
    std::vector<float> occBuf;
    unsigned posBufID;
	unsigned norBufID;
    unsigned texBufID;
    unsigned occBufID;

    struct cudaGraphicsResource* cudaOccResource;
    
    Batch getInputs();
    void createCudaVBO(GLuint *vbo, struct cudaGraphicsResource **vboRes, unsigned int vboResFlags, unsigned int size);
    void deleteCudaVBO(GLuint *vbo, struct cudaGraphicsResource *vboRes);
    void computeOcclusion();
};

#endif