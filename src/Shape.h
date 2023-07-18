// #pragma once
// #ifndef SHAPE_H
// #define SHAPE_H

// #define GLM_FORCE_RADIANS
// #include <glm/glm.hpp>
// #include <glm/gtc/matrix_transform.hpp>

// #include "Ray.h"
// // #include "Hit.h"
// #include "Material.h"

// #include <vector>
// #include <iostream>

// #include <bvh/v2/bvh.h>
// #include <bvh/v2/vec.h>
// #include <bvh/v2/node.h>
// #include <bvh/v2/default_builder.h>
// #include <bvh/v2/thread_pool.h>
// #include <bvh/v2/executor.h>
// #include <bvh/v2/stack.h>
// #include <bvh/v2/tri.h>

// using Scalar  = float;
// // using Vec2D   = bvh::v2::Vec<Scalar, 2>;
// using Vec3D   = bvh::v2::Vec<Scalar, 3>;
// using BBox2D  = bvh::v2::BBox<Scalar, 2>;
// using BBox3D  = bvh::v2::BBox<Scalar, 3>;
// // using Tri2D   = bvh::v2::Tri<Scalar, 2>;
// using Tri3D   = bvh::v2::Tri<Scalar, 3>;
// // using Node2D  = bvh::v2::Node<Scalar, 2>;
// using Node3D  = bvh::v2::Node<Scalar, 3>;
// // using Bvh2D   = bvh::v2::Bvh<Node2D>;
// using Bvh3D   = bvh::v2::Bvh<Node3D>;
// // using Ray2D   = bvh::v2::Ray<Scalar, 2>;
// using Ray3D   = bvh::v2::Ray<Scalar, 3>;
// using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

// #define NO_ROTATION glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);

// class Shape {
// public:
//     glm::mat4 transform;

//     glm::mat4 getTransformationMatrix() {
//         glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
//         glm::mat4 R = glm::rotate(glm::mat4(1.0f), rotation[0], glm::vec3(rotation[1], rotation[2], rotation[3]));
//         glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
//         return T * R * S;
//     }
    
// private:
//     glm::vec3 position;
//     glm::vec4 rotation;
//     glm::vec3 scale;
// };

// #endif

