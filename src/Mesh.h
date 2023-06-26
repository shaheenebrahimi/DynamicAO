#pragma once
#ifndef MESH_H
#define MESH_H

#include <string>
#include "Triangle.h"
#include "AABB.h"


class Mesh : public Shape {
public:
    AABB* box;
    std::vector<Triangle*> triangles;
    
    Mesh(const std::string& objPath, glm::vec3 position, glm::vec4 rotation, glm::vec3 scale, Material* mat);
    void loadMesh(const std::string& meshName);
    void bufToTriangles(std::vector<float>& posBuf, std::vector<float>& norBuf, std::vector<float>& texBuf);
    void computeBounds(std::vector<float>& posBuf);
    glm::vec3 getMinBound() { return minBound; }
    glm::vec3 getMaxBound() { return maxBound; }
    Hit* collider(Ray& ray);

private:
    glm::vec3 minBound;
    glm::vec3 maxBound;
};

#endif