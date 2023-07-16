#pragma once
#ifndef SCENE_H
#define SCENE_H

#define GLM_FORCE_RADIANS
#define DEG_TO_RAD M_PI / 180.0f
#include <glm/glm.hpp>
#include <vector>

#include "Shape.h"
#include "Light.h"
#include "Ray.h"
#include "Collision.h"
#include "Camera.h"


class Scene {
public:
    Camera cam;
    glm::vec3 bkgColor;
    std::vector<Shape*> shapes;
    std::vector<Light*> lights;

    Scene() { this->bkgColor = glm::vec3(0.0f, 0.0f, 0.0f); }
    // void loadScene(string filename);

    void setBkgColor(glm::vec3 bkgColor) { this->bkgColor = bkgColor; }
    void setCamResolution(int resolution) { this->cam.resolution = resolution; }

    void addShape(Shape* obj) { shapes.push_back(obj); }
    void addLight(Light* l) { lights.push_back(l); }
    std::vector<Shape*> getShapes() { return shapes; }
    std::vector<Light*> getLights() { return lights; }
};

#endif