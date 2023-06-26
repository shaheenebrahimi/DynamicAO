#pragma once
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Shape.h"

struct BoundingBox {
    float minX;
    float maxX;
    float minY;
    float maxY;

    BoundingBox() {
        this->minX = 0.0f;
        this->maxX = 0.0f;
        this->minY = 0.0f;
        this->maxY = 0.0f;
    }

    BoundingBox(glm::vec2 v0, glm::vec2 v1, glm::vec2 v2) {
        this->minX = std::min(std::min(v0.x, v1.x), v2.x);
        this->maxX = std::max(std::max(v0.x, v1.x), v2.x);
        this->minY = std::min(std::min(v0.y, v1.y), v2.y);
        this->maxY = std::max(std::max(v0.y, v1.y), v2.y);
    }
};

class Triangle : public Shape {
public:
    glm::vec3 vert0;
    glm::vec3 vert1;
    glm::vec3 vert2;
    glm::vec3 nor0;
    glm::vec3 nor1;
    glm::vec3 nor2;
    glm::vec2 tex0;
    glm::vec2 tex1;
    glm::vec2 tex2;
    float area;
    BoundingBox box;

    Triangle();
    Triangle(
        glm::vec3 vert0, glm::vec3 vert1, glm::vec3 vert2,
        glm::vec3 nor0, glm::vec3 nor1, glm::vec3 nor2,
        glm::vec2 tex0, glm::vec2 tex1, glm::vec2 tex2,
        Material* mat
    );
    glm::vec3 computeBarycentric(glm::vec2 tex);
    Hit* collider(Ray& ray);
    glm::vec3 getMinBound() { return glm::vec3(std::min({vert0.x, vert1.x, vert2.x}), std::min({vert0.y, vert1.y, vert2.y}), std::min({vert0.z, vert1.z, vert2.z})); }
    glm::vec3 getMaxBound() { return glm::vec3(std::max({vert0.x, vert1.x, vert2.x}), std::max({vert0.y, vert1.y, vert2.y}), std::max({vert0.z, vert1.z, vert2.z})); }

private:
    float computeArea(glm::vec2 vert0, glm::vec2 vert1, glm::vec2 vert2);
};

#endif