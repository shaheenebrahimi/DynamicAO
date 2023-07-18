#pragma once
#ifndef COLLISION_H
#define COLLISION_H

#include "Hit.h"
#include "Object.h"

class Collision {
public:
    Hit hit;
    std::shared_ptr<Object> obj;

    Collision(Hit& hit, std::shared_ptr<Object> obj) {
        this->hit = hit;
        this->obj = obj;
    }
};

#endif