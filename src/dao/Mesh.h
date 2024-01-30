#pragma once
#ifndef MESH_H
#define MESH_H

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

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

    void loader(const std::string& dir, const std::string& name);
    void loadMesh(const std::string& fileName);
    void loadSkeleton(const std::string& fileName);
    void loadSkinWeights(const std::string& fileName);
    void loadHierarchy(const std::string& fileName);
    void loadEvaluator(const std::string& fileName);
    void loadBuffers(); // only for rasterization

    void setBoneAngles(const std::vector<float>& thetas);
    void setTransform(glm::mat4 transform) { this->transform = transform; }

    void dumpMesh(const std::string &filename, const std::vector<std::string>& header = {});
    void updateMesh();
    void drawMesh(std::shared_ptr<Program> prog);

    int getBoneCount() { return boneCount; }
    int getBoneIndex(const std::string& name) { return boneMap[name]; }
private:
    //std::vector<std::shared_ptr<Triangle>> triangles;
    //std::vector<std::shared_ptr<Triangle>> transformed;

    std::shared_ptr<Evaluator> eval;

    std::vector<unsigned int> elemBuf;
    std::vector<float> posBuf;
    std::vector<float> norBuf;
    std::vector<float> texBuf;
    std::vector<float> occBuf;
    std::vector<float> thetaBuf;                        // all of the thetas for bones 

    std::vector<float> skBoneInds;
    std::vector<float> skWeights;
    std::vector<float> skNumInfl;
    std::vector<float> skPosBuf;
    std::vector<float> skNorBuf;

    int boneCount, influences;
    std::vector<glm::vec3> bindPose;                    // absolute positions - size of (bones,)
    std::vector<int> boneHierarchy;                     // hierarchy of bones
    std::vector<glm::vec3> relativeTranslations;        // bone translation with respect to parent
    std::vector<glm::quat> relativeRotations;           // bone rotation with respect to parent
    std::unordered_map<std::string, int> boneMap;       // bone name to index

    unsigned elemBufID;
    unsigned posBufID;
	unsigned norBufID;
    unsigned texBufID;
    unsigned occBufID;

    struct cudaGraphicsResource* cudaOccResource;

    void traverseHierarchy(std::vector<glm::mat4>& pose);
    void applyPose(const std::vector<glm::mat4>& pose);
    
    Batch getInputs();
    void createCudaVBO(GLuint *vbo, struct cudaGraphicsResource **vboRes, unsigned int vboResFlags, unsigned int size);
    void deleteCudaVBO(GLuint *vbo, struct cudaGraphicsResource *vboRes);
    void computeOcclusion();
};

#endif