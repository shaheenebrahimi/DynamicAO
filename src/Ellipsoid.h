#pragma once
#ifndef ELLIPSOID_H
#define ELLIPSOID_H

#include "Shape.h"
#include "Ray.h"
#include "Hit.h"

#include <vector>


class Ellipsoid : public Shape {
public:
    Ellipsoid();
    Ellipsoid(glm::vec3 position, glm::vec4 rotation, glm::vec3 scale, Material* mat);
    Hit* collider(Ray& ray);
    glm::vec3 getMinBound() { return position - scale; }
    glm::vec3 getMaxBound() { return position + scale; }
    
};

#endif