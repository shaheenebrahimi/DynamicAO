#pragma once
#ifndef TRIANGLE_H
#define TRIANGLE_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include "Material.h"


using Scalar  = float;
using Tri3D   = bvh::v2::Tri<Scalar, 3>;
using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

class Triangle {
public:
    glm::vec3 pos0;
    glm::vec3 pos1;
    glm::vec3 pos2;
    glm::vec3 nor0;
    glm::vec3 nor1;
    glm::vec3 nor2;
    glm::vec2 tex0;
    glm::vec2 tex1;
    glm::vec2 tex2;
    Material mat;
    float area;

    Triangle();
    Triangle(
        glm::vec3 pos0, glm::vec3 pos1, glm::vec3 pos2,
        glm::vec3 nor0, glm::vec3 nor1, glm::vec3 nor2,
        glm::vec2 tex0, glm::vec2 tex1, glm::vec2 tex2,
        Material mat
    );
    ~Triangle() { }
    glm::vec3 computeBarycentric(glm::vec2 tex);
    // Hit collider(Ray& ray);
    Triangle applyTransformation(glm::mat4 matrix);
    Tri3D convertPosToTri();
    Tri3D convertTexToTri();

private:
    float computeArea(glm::vec2 pos0, glm::vec2 pos1, glm::vec2 pos2);
};

#endif