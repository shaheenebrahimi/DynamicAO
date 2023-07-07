#pragma once
#ifndef MESH_H
#define MESH_H

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
// #include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include <iostream>

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using BVH     = bvh::v2::Bvh<Node>;
using Ray3    = bvh::v2::Ray<Scalar, 3>;
using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

#include <string>
#include "Triangle.h"
#include "AABB.h"


class Mesh : public Shape {
public:
    BVH bvh;
    AABB* box;
    std::vector<Triangle*> triangles;
    
    Mesh(const std::string& objPath, glm::vec3 position, glm::vec4 rotation, glm::vec3 scale, Material* mat);
    void loadMesh(const std::string& meshName);
    void bufToTriangles(std::vector<float>& posBuf, std::vector<float>& norBuf, std::vector<float>& texBuf);
    void computeBounds(std::vector<float>& posBuf);
    void initializeBVH();
    glm::vec3 getMinBound() { return minBound; }
    glm::vec3 getMaxBound() { return maxBound; }
    Hit* collider(Ray& ray);

private:
    glm::vec3 minBound;
    glm::vec3 maxBound;
    std::vector<PrecomputedTri> precomputed_tris;
};

#endif