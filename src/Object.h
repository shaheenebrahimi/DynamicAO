#pragma once
#ifndef OBJECT_H
#define OBJECT_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Mesh.h"
#include "Material.h"
#include "Texture.h"
#include "Program.h"
#include "MatrixStack.h"

class Object {
    // Transform
public:
    std::shared_ptr<Mesh> mesh; // all meshes should have a collider
    std::shared_ptr<Texture> tex; // all meshes should have a collider
    std::shared_ptr<Material> mat;
    glm::mat4 transform;

    Object();
    Object(const std::string &objPath);
    Object(const std::string &objPath, std::shared_ptr<Material> mat);
    Object(const std::string &objPath, const glm::vec3 &position, const glm::vec4 &rotation, const glm::vec3 &scale,  std::shared_ptr<Material> mat);

    void translate(const glm::vec3 &translation);
    void rotate(const glm::vec4 &rotation);
    void scale(const glm::vec3 &scale);

    void setMaterial(const glm::vec3 &kd, const glm::vec3 &ks, const glm::vec3 &ka, float s);
    void setPosition(const glm::vec3 &position);
    void setRotation(const glm::vec4 &rotation);
    void setScale(const glm::vec3 &scale);

    void addTexture(const std::string &texPath);

    void draw(std::shared_ptr<Program> prog, std::shared_ptr<MatrixStack> P, std::shared_ptr<MatrixStack> MV);

private:
    glm::mat4 T;
    glm::mat4 R;
    glm::mat4 S;

    glm::mat4 computeTransform();
};

#endif