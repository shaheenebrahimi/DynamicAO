#pragma once
#ifndef AABB_H
#define AABB_H

#include "Shape.h"

// class AABB : public Shape {
class AABB {
public:
    glm::vec3 minBound; // bottom left
    glm::vec3 maxBound; // bottom right

    AABB();
    AABB(glm::vec3 minBound, glm::vec3 maxBound);
    AABB(Shape* s);
    AABB(AABB* b1, AABB* b2);
    Hit* collider(Ray& ray);
};

#endif