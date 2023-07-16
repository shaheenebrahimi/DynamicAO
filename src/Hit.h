#pragma once
#ifndef HIT_H
#define HIT_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Triangle.h"

class Hit {
public:
    float t; // distance along ray
    float w; // bary
    float u;
    float v;
    Triangle* intersected; // change to primitive later -> sphere triangle

    Hit();
    Hit(float t, Triangle* intersected);
    Hit(float t, float w, float u, float v, Triangle* intersected);

    glm::vec3 computePos();
    glm::vec3 computeNor();
    glm::vec2 computeTex();
};

#endif