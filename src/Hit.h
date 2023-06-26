#pragma once
#ifndef HIT_H
#define HIT_H

#include <glm/glm.hpp>

class Hit {
public:
    glm::vec3 pos; // position
    glm::vec3 nor; // normal
    glm::vec2 tex; // normal
    float t;

    Hit(glm::vec3 pos, glm::vec3 nor, glm::vec2 tex, float t) {
        this->pos = pos; this->nor = nor; this->tex = tex; this->t = t;
    }
};

#endif