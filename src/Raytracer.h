#pragma once
#ifndef RAYTRACER_H
#define RAYTRACER_H

#define GLM_FORCE_RADIANS
#define DEG_TO_RAD M_PI / 180.0f

#include <glm/glm.hpp>
#include <vector>

#include "Object.h"
#include "Light.h"
#include "Ray.h"
#include "Camera.h"
#include "Scene.h"
#include "Image.h"
#include "Hit.h"
#include "Collision.h"

class Raytracer {
public:
    Raytracer();
    Raytracer(std::string& filename, int resolution);
    void setFilename(std::string& filename) { this->filename = filename; }
    void setResolution(int resolution) { this->resolution = resolution; init(); }
    void setScene(Scene& scn) { this->scn = scn; init(); }
    void init();
    void render();

private:
    Scene scn;
    int resolution;
    std::string filename;
    std::shared_ptr<Image> img;

    std::optional<Collision> shootRay(Ray& ray);
    glm::vec3 computeColor(Ray& ray);
};


#endif