#pragma once
#ifndef MESH_H
#define MESH_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "Triangle.h"
#include "Shape.h"
#include "Hit.h"
#include "Ray.h"
// #include "Hit.h"
#include "Material.h"

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

using Scalar  = float;
using Vec3D   = bvh::v2::Vec<Scalar, 3>;
using BBox2D  = bvh::v2::BBox<Scalar, 2>;
using BBox3D  = bvh::v2::BBox<Scalar, 3>;
using Tri3D   = bvh::v2::Tri<Scalar, 3>;
using Node3D  = bvh::v2::Node<Scalar, 3>;
using Bvh3D   = bvh::v2::Bvh<Node3D>;
using Ray3D   = bvh::v2::Ray<Scalar, 3>;
using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

#define NO_ROTATION glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);

class Mesh {
public:
    glm::mat4 transform;

    Mesh(const std::string& objPath);
    Mesh(const std::string& objPath, Material mat);
    ~Mesh();
    void loadMesh(const std::string& meshName);
    void constructBVH();
    void setTransform(glm::mat4 transform) { this->transform = transform; }
    Hit collider(Ray& ray);
private:
    static constexpr bool should_permute = true;
    static constexpr bool use_robust_traversal = false;
    static constexpr size_t stack_size = 64;
    static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();

    Bvh3D accel;
    std::vector<Triangle> triangles;
    std::vector<Triangle> transformed;
    std::vector<PrecomputedTri> precomputed;

    void bufToTriangles(std::vector<float>& posBuf, std::vector<float>& norBuf, std::vector<float>& texBuf);
};

#endif