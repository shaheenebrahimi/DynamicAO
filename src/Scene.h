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
    glm::vec3 background;
    std::vector<Shape*> shapes;
    std::vector<Light*> lights;

    Scene();
    // void loadScene(string filename);

    void setBackground(glm::vec3 background) { this->background = background; }
    void setCamResolution(int resolution) { this->cam.resolution = resolution; }

    void addShape(Shape* obj) { shapes.push_back(obj); }
    void addLight(Light* l) { lights.push_back(l); }
    std::vector<Shape*> getShapes() { return shapes; }
    std::vector<Light*> getLights() { return lights; }

    Collision* shootRay(Ray& ray);
    glm::vec3 computeColor(Ray& ray, int depth = 0);
    float computeRayAmbientOcclusion(Ray& ray, std::vector<glm::vec3>& kernel, std::vector<glm::vec3>& noise, float radius);
    float computePointAmbientOcclusion(glm::vec3 pos, glm::vec3 nor, std::vector<glm::vec3>& kernel, std::vector<glm::vec3>& noise, float radius);

private:
    const std::string RESOURCE_DIR = "../resources/";
    const int MAX_BOUNCES = 4;
};

#endif