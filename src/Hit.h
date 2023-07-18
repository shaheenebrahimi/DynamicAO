#pragma once
#ifndef HIT_H
#define HIT_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Triangle.h"

class Hit {
public:
    float t; // distance along ray
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 tex;

    Hit() {
        this->t = 0;
        this->pos = glm::vec3(0);
        this->nor = glm::vec3(0);
        this->tex = glm::vec2(0);
    }

    Hit(float t, glm::vec3 pos, glm::vec3 nor, glm::vec2 tex) {
        this->t = t;
        this->pos = pos;
        this->nor = nor;
        this->tex = tex;
    }
};

#endif