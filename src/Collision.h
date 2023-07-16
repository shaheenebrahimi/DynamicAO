#pragma once
#ifndef COLLISION_H
#define COLLISION_H

#include "Hit.h"

class Collision {
public:
    Hit hit;
    Triangle* intersected;

    Collision(Hit hit, Triangle* intersected) {
        this->hit = hit;
        this->intersected = intersected;
    }
};

#endif