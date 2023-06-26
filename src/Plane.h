#pragma once
#ifndef PLANE_H
#define PLANE_H

#include "Shape.h"
#include "Ray.h"
#include "Hit.h"

class Plane : public Shape {
public:
    glm::vec3 normal;
    Plane(glm::vec3 normal, glm::vec3 position, glm::vec4 rotation, Material* mat);
    Hit* collider(Ray& ray);
    glm::vec3 getMinBound() { return glm::vec3(0, 0, 0); }
    glm::vec3 getMaxBound() { return glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX); }
};

#endif