#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "Shape.h"

class Sphere : public Shape {
public:
    float radius;

    Sphere();
    Sphere(glm::vec3 position, glm::vec3 scale, float radius, Material* mat);
    Hit* collider(Ray& ray);
    glm::vec3 getMinBound() { return position - radius; }
    glm::vec3 getMaxBound() { return position + radius; }
};

#endif
