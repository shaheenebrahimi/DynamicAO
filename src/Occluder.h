#pragma once
#ifndef OCCLUDER_H
#define OCCLUDER_H

#define GLM_FORCE_RADIANS
#define DEG_TO_RAD M_PI / 180.0f

#include <glm/glm.hpp>

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <random>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

using Scalar  = float;
using BBox2D  = bvh::v2::BBox<Scalar, 2>;
using Tri2D   = bvh::v2::Tri<Scalar, 2>;

#include "Object.h"
#include "Light.h"
#include "Hit.h"
#include "Ray.h"
#include "Camera.h"
#include "Scene.h"
#include "Image.h"
#include "Mesh.h"

class Occluder {
public:
    Occluder();
    Occluder(const std::string &filename, int resolution);
    ~Occluder();
    void setFilename(const std::string &filename) { this->filename = filename; }
    void setResolution(int resolution) { this->resolution = resolution; }
    void setScene(const Scene& scn) { this->scn = scn; }
    void setSamples(int samples) { this->samples = samples; }
    void setRadius(float radius) { this->radius = radius; }
    void init();
    void render();
    void renderTexture(std::shared_ptr<Object> target);
    void renderTextureLegacy(std::shared_ptr<Object> target);

private:
    Scene scn;
    int resolution;
    std::string filename;
    std::shared_ptr<Image> img;

    int samples;
    float radius;
    std::vector<glm::vec3> kernel; // ao kernel
	std::vector<glm::vec3> noise; // ao noise

    std::optional<Hit> shootRay(const Ray& ray);
    void genOcclusionHemisphere();
    float computeRayOcclusion(const Ray& ray);
    float computePointOcclusion(const glm::vec3 &pos, glm::vec3 &nor);
};


#endif