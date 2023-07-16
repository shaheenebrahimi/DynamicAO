#pragma once
#ifndef SCENE_H
#define SCENE_H

#define GLM_FORCE_RADIANS
#define DEG_TO_RAD M_PI / 180.0f
#include <glm/glm.hpp>
#include <vector>

#include "Object.h"
#include "Light.h"
#include "Ray.h"
#include "Hit.h"
#include "Camera.h"


class Scene {
public:
    Camera cam;
    glm::vec3 bkgColor;
    std::vector<Object> shapes;
    std::vector<Light> lights;
    int maxBounces = 4;

    Scene() { this->bkgColor = glm::vec3(0.0f, 0.0f, 0.0f); }
    ~Scene() { shapes.clear(); lights.clear(); }
    // void loadScene(string filename);

    void setBkgColor(glm::vec3 bkgColor) { this->bkgColor = bkgColor; }
    void setMaxBounces(int bounces) { this->maxBounces = bounces; }

    void addShape(Object obj) { shapes.push_back(obj); }
    void addLight(Light l) { lights.push_back(l); }
    
};

#endif