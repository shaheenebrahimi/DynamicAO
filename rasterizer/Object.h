#pragma once
#ifndef OBJECT_H
#define OBJECT_H

#include "Shape.h"
#include "Program.h"
#include "MatrixStack.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>

class Object {
public:
    struct Material {
        glm::vec3 ka;
        glm::vec3 kd;
        glm::vec3 ks;
        float s;
        Material() : ka(glm::vec3(0.0f)), kd(glm::vec3(0.0f)), ks(glm::vec3(0.0f)), s(0.0f) { }
        Material(glm::vec3 ka, glm::vec3 kd, glm::vec3 ks, float s) { this->ka = ka; this->kd = kd; this->ks = ks; this->s = s; }
    };

    Object();
    Object(std::shared_ptr<Shape> mesh, glm::vec3 ka, glm::vec3 kd, glm::vec3 ks, float s);
    virtual ~Object();
    void draw(std::shared_ptr<Program> prog, std::shared_ptr<MatrixStack> P, std::shared_ptr<MatrixStack> MV);

    std::shared_ptr<Shape> getMesh() { return mesh; }
    Material getMat() { return mat; }
    
private:
    std::shared_ptr<Shape> mesh;
    Material mat;
};

#endif