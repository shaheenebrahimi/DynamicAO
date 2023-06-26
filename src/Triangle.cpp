#include "Triangle.h"

Triangle::Triangle() {
    this->vert0 = glm::vec3(0);
    this->vert1 = glm::vec3(0);
    this->vert2 = glm::vec3(0);
    this->nor0 = glm::vec3(0);
    this->nor1 = glm::vec3(0);
    this->nor2 = glm::vec3(0);
    this->tex0 = glm::vec2(0);
    this->tex1 = glm::vec2(0);
    this->tex2 = glm::vec2(0);
    this->mat = nullptr;
    this->area = 0.0f;
}

Triangle::Triangle(glm::vec3 vert0, glm::vec3 vert1, glm::vec3 vert2, glm::vec3 nor0, glm::vec3 nor1, glm::vec3 nor2, glm::vec2 tex0, glm::vec2 tex1, glm::vec2 tex2, Material* mat) {
    this->vert0 = vert0;
    this->vert1 = vert1;
    this->vert2 = vert2;
    this->nor0 = nor0;
    this->nor1 = nor1;
    this->nor2 = nor2;
    this->tex0 = tex0;
    this->tex1 = tex1;
    this->tex2 = tex2;
    this->mat = mat;
    this->area = computeArea(this->tex0, this->tex1, this->tex2);
    this->box = BoundingBox(this->tex0, this->tex1, this->tex2);
}

glm::vec3 Triangle::computeBarycentric(glm::vec2 pos) {
    // tex0 = A, tex1 = B, tex2 = C
    float a = computeArea(pos, tex1, tex2) / this->area;
    float b = computeArea(pos, tex2, tex0) / this->area;
    float c = computeArea(pos, tex0, tex1) / this->area;

    return glm::vec3(a, b, c);
}

Hit* Triangle::collider(Ray& ray) {
    float epsilon = 0.0001f;

    glm::vec3 edge1 = vert1 - vert0;
    glm::vec3 edge2 = vert2 - vert0;

    glm::vec3 p_vec = glm::cross(ray.v, edge2);

    float det = dot(edge1, p_vec);
    if (det < epsilon && det > -epsilon) // lies in triangle plane
        return nullptr;
    float inv_det = 1.0f / det;

    // calculate U parameter and test bounds
    glm::vec3 t_vec = ray.p - vert0;
    float u = dot(t_vec, p_vec) * inv_det;
    if (u < 0.0f || u > 1.0f) // barycentric
        return nullptr;

    glm::vec3 q_vec = glm::cross(t_vec, edge1);

    // calculate V parameter and test bounds
    float v = dot(ray.v, q_vec) * inv_det;
    if (v < 0.0f || u + v > 1.0)
        return nullptr;

    // calculate W parameter, ray intersects triangle
    float w = 1.0f - u - v;

    // calculate t'
    float t = dot(edge2, q_vec) * inv_det;
    if (t < epsilon)
        return nullptr;

    // compute barycentric
    glm::vec3 pos = w * vert0 + u * vert1 + v * vert2;
    glm::vec3 nor = normalize(w * nor0 + u * nor1 + v * nor2);
    glm::vec2 tex = w * tex0 + u * tex1 + v * tex2;

    return new Hit(pos, nor, tex, t);
}

float Triangle::computeArea(glm::vec2 v0, glm::vec2 v1, glm::vec2 v2) {
    return 0.5*((v1.x-v0.x)*(v2.y-v0.y)-(v2.x-v0.x)*(v1.y-v0.y));
}