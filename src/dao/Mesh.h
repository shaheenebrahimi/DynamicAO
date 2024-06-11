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

#include "tensorflow/c/c_api.h"
#include "cppflow/ops.h"
#include "cppflow/model.h"

#define NO_ROTATION glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);

class Mesh {
public:

    Mesh();
    Mesh(const std::string name);
    ~Mesh();

    void loader(const std::string& dir, const std::string& name);
    void loadMesh(const std::string& fileName);
    void loadSkeleton(const std::string& fileName);
    void loadLocalSkeleton(const std::string& filename);
    void loadLocalTransforms(const std::string& filename);
    void loadSkinWeights(const std::string& fileName);
    void loadHierarchy(const std::string& fileName);
    void loadEvaluator(const std::string& fileName);
    void loadGenerator(const std::string& modelPath);
    void loadOcclusionBuffer(const std::string& fileName);
    void loadBuffers(); // only for rasterization
    void updateBuffers(); // only for rasterization

    void setAnimation(const std::string& fileName);
    void setPose(const std::vector<glm::vec3>& orientations);
    void setBone(const int boneInd, glm::vec3 orientation);
    std::vector<glm::mat4> computeAbsolutePose(std::vector<glm::vec3>& relRotations);
    void computeBoneTransforms();
    void traverseHierarchy();
    void applySkinning();
    void stepAnimation();
    void setFrame(int frame);

    void updateMesh();
    void dumpMesh(const std::string &filename, const std::vector<std::string>& header = {});
    void drawMesh(std::shared_ptr<Program> prog);

    int getBoneCount() { return boneCount; }
    int getFrameCount() { return frameCount; }
    glm::vec3 getRotationData(int frame, int boneInd) { return frameData[frame][boneInd]; }
    glm::vec3 getBoneRotation(int boneInd) { return relativeRotations[boneInd]; }
    std::vector<float> getFlattenedRotations();
    int getBoneIndex(const std::string& name) { return boneMap[name]; }

    void bindTexture(GLint handle);
    void unbindTexture(GLint handle);
private:
    std::shared_ptr<Evaluator> eval;
    std::shared_ptr<cppflow::model> model;

    std::vector<unsigned int> elemBuf;
    std::vector<float> posBuf;
    std::vector<float> norBuf;
    std::vector<float> texBuf;
    std::vector<float> occBuf;
    std::vector<float> genBuf;

    std::vector<float> skBoneInds;
    std::vector<float> skWeights;
    std::vector<float> skNumInfl;
    std::vector<float> skPosBuf;
    std::vector<float> skNorBuf;

    int vertexCount, boneCount, frameCount, influences, currFrame;
    std::vector<std::vector<glm::mat4>> boneTransforms;     // local bone transforms
    std::vector<std::vector<glm::vec3>> frameData;          // poses of each bone for each frame
    std::vector<glm::mat4> pose;                            // matrix transforms to get to pose
    std::vector<glm::mat4> bindPose;                        // position for each bone
    std::vector<glm::vec3> jointPositions;                  // absolute positions - size of (bones,)
    std::vector<glm::vec3> relativeTranslations;            // bone translation with respect to parent
    std::vector<glm::vec3> relativeRotations;               // bone rotation with respect to parent
    std::vector<int> boneHierarchy;                         // hierarchy of bones
    std::unordered_map<std::string, int> boneMap;           // bone name to index

    unsigned elemBufID;
    unsigned posBufID;
	unsigned norBufID;
    unsigned texBufID;
    unsigned occBufID;
    unsigned genBufID;

    struct cudaGraphicsResource* cudaOccResource;
    
    void computeOcclusion();
    void generateTexture();
    //void computeRelativeRotations(const std::vector<glm::quat>& orientations);

    Batch getBatch();
    void createCudaVBO(GLuint *vbo, struct cudaGraphicsResource **vboRes, unsigned int vboResFlags, unsigned int size);
    void deleteCudaVBO(GLuint *vbo, struct cudaGraphicsResource *vboRes);
};

#endif