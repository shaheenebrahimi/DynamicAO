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
    std::shared_ptr<Mesh> mesh; // all meshes should have a collider
    std::shared_ptr<Material> mat;
    glm::mat4 transform;

    Object();
    Object(const std::string& objPath);
    Object(const std::string& objPath, std::shared_ptr<Material> mat);
    Object(const std::string& objPath, glm::vec3 position, glm::vec4 rotation, glm::vec3 scale,  std::shared_ptr<Material> mat);

    void translate(glm::vec3 translation);
    void rotate(glm::vec4 rotation);
    void scale(glm::vec3 scale);

    void setMaterial(glm::vec3 kd, glm::vec3 ks, glm::vec3 ka, float s);
    void setPosition(glm::vec3 position);
    void setRotation(glm::vec4 rotation);
    void setScale(glm::vec3 scale);

private:
    glm::mat4 T;
    glm::mat4 R;
    glm::mat4 S;

    glm::mat4 computeTransform();
};

#endif