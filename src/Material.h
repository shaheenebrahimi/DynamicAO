#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

#include <glm/glm.hpp>

#include <vector>
#include <memory>

#include "Light.h"

class Material {
public:
    glm::vec3 kd;
    glm::vec3 ks;
    glm::vec3 ka;
    float s;

    Material();
    Material(glm::vec3 kd, glm::vec3 ks, glm::vec3 ka, float s);
    glm::vec3 computeFrag(glm::vec3 ray, glm::vec3 pos, glm::vec3 nor, std::vector<std::shared_ptr<Light>>& lights);
};

#endif