#pragma once
#ifndef SCENE_H
#define SCENE_H

#define GLM_FORCE_RADIANS
#define DEG_TO_RAD M_PI / 180.0f

#include <glm/glm.hpp>
#include <vector>
#include <memory>

#include "Object.h"
#include "Light.h"
#include "Ray.h"
#include "Hit.h"
#include "Camera.h"


class Scene {
public:
    Camera cam;
    glm::vec3 bkgColor;
    std::vector<std::shared_ptr<Object>> objects;
    std::vector<std::shared_ptr<Light>> lights;
    int maxBounces = 4;

    Scene() { this->bkgColor = glm::vec3(0.0f, 0.0f, 0.0f); }
    ~Scene() { objects.clear(); lights.clear(); }
    // void loadScene(string filename);

    void setBkgColor(const glm::vec3 &bkgColor) { this->bkgColor = bkgColor; }
    void setMaxBounces(int bounces) { this->maxBounces = bounces; }

    void addObject(std::shared_ptr<Object> obj) { objects.push_back(obj); }
    void addLight(std::shared_ptr<Light> l) { lights.push_back(l); }
    
};

#endif