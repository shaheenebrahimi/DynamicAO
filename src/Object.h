#pragma once
#ifndef OBJECT_H
#define OBJECT_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Mesh.h"
#include "Material.h"

class Object {
    // Transform
public:
    Mesh mesh; // all meshes should have a collider
    Material mat;

    Object();
    Object(std::string& objPath);
    Object(std::string& objPath, Material mat);
    Object(std::string& objPath, glm::vec3 position, glm::vec4 rotation, glm::vec3 scale, Material mat);

    void translate(glm::vec3 translation);
    void rotate(glm::vec4 rotation);
    void scale(glm::vec3 scale);

    void setPosition(glm::vec3 position);
    void setRotation(glm::vec4 rotation);
    void setScale(glm::vec3 scale);

private:
    glm::mat4 transform;
    glm::mat4 T;
    glm::mat4 R;
    glm::mat4 S;

    glm::mat4 computeTransform();
};

#endif