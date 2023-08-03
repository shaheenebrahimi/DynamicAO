#pragma once
#ifndef MESH_H
#define MESH_H

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <optional>

#include "Triangle.h"
#include "Hit.h"
#include "Ray.h"
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
using Vec   = bvh::v2::Vec<Scalar, 3>;
using BBox  = bvh::v2::BBox<Scalar, 3>;
using Tri   = bvh::v2::Tri<Scalar, 3>;
using Node  = bvh::v2::Node<Scalar, 3>;
using Bvh   = bvh::v2::Bvh<Node>;
using BvhRay   = bvh::v2::Ray<Scalar, 3>;
using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

#define NO_ROTATION glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);

class Mesh {
public:
    glm::mat4 transform;

    Mesh();
    Mesh(const std::string& objPath);
    ~Mesh();
    void loadMesh(const std::string& meshName);
    void constructBVH();
    void setTransform(glm::mat4 transform) { this->transform = transform; constructBVH(); }
    std::vector<std::shared_ptr<Triangle>> getTriangles() { return transformed; } // return transformed tris
    std::optional<Hit> collider(Ray& ray);
private:
    static constexpr bool should_permute = true;
    static constexpr bool use_robust_traversal = false;
    static constexpr size_t stack_size = 64;
    static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();

    Bvh accel;
    std::vector<std::shared_ptr<Triangle>> triangles;
    std::vector<std::shared_ptr<Triangle>> transformed;
    std::vector<PrecomputedTri> precomputed;

    void bufToTriangles(std::vector<float>& posBuf, std::vector<float>& norBuf, std::vector<float>& texBuf);
};

#endif