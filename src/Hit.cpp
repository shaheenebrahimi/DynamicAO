#include "Hit.h"

float t; // distance along ray
    float w; // bary
    float u;
    float v;
    Triangle* intersected;

    Hit::Hit() { 
        this->t = 0;
        this->w = 0;
        this->u = 0;
        this->v = 0;
        this->intersected = nullptr;
    }

    Hit::Hit(float t, Triangle* intersected) {
        this->t = t;
        this->w = 0;
        this->u = 0;
        this->v = 0;
        this->intersected = intersected;
    }

    Hit::Hit(float t, float w, float u, float v, Triangle* intersected) {
        this->t = t;
        this->w = w;
        this->u = u;
        this->v = v;
        this->intersected = intersected;
    }

    glm::vec3 Hit::computePos() {
        return w * this->intersected->pos0 + u * this->intersected->pos1 + v * this->intersected->pos2;
    }

    glm::vec3 Hit::computeNor() {
        return normalize(w * this->intersected->nor0 + u * this->intersected->nor1 + v * this->intersected->nor2);
    }

    glm::vec2 Hit::computeTex() {
        return w * this->intersected->tex0 + u * this->intersected->tex1 + v * this->intersected->tex2;
    }