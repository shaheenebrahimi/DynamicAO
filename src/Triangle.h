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

#include "Hit.h"
#include "Ray.h"


using Scalar  = float;
using Vec2D   = bvh::v2::Vec<Scalar, 2>;
using Vec   = bvh::v2::Vec<Scalar, 3>;
using Tri2D   = bvh::v2::Tri<Scalar, 2>;
using Tri   = bvh::v2::Tri<Scalar, 3>;
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
    float area;

    Triangle();
    Triangle(
        const glm::vec3 &pos0, const glm::vec3 &pos1, const glm::vec3 &pos2,
        const glm::vec3 &nor0, const glm::vec3 &nor1, const glm::vec3 &nor2,
        const glm::vec2 &tex0, const glm::vec2 &tex1, const glm::vec2 &tex2
    );
    ~Triangle() { }
    glm::vec3 computeBarycentric(const glm::vec2 &tex);
    glm::vec3 interpolatePos(float w, float u, float v);
    glm::vec3 interpolateNor(float w, float u, float v);
    glm::vec2 interpolateTex(float w, float u, float v);
    std::shared_ptr<Triangle> applyTransformation(const glm::mat4 &matrix);
    Tri convertPosToTri();
    Tri2D convertTexToTri();

private:
    float computeArea(const glm::vec2 &pos0, const glm::vec2 &pos1, const glm::vec2 &pos2);
};

#endif