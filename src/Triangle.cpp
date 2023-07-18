#include "Triangle.h"

/* PUBLIC */

Triangle::Triangle() {
    this->pos0 = glm::vec3(0);
    this->pos1 = glm::vec3(0);
    this->pos2 = glm::vec3(0);
    this->nor0 = glm::vec3(0);
    this->nor1 = glm::vec3(0);
    this->nor2 = glm::vec3(0);
    this->tex0 = glm::vec2(0);
    this->tex1 = glm::vec2(0);
    this->tex2 = glm::vec2(0);
    this->area = 0.0f;
}

Triangle::Triangle(glm::vec3 pos0, glm::vec3 pos1, glm::vec3 pos2, glm::vec3 nor0, glm::vec3 nor1, glm::vec3 nor2, glm::vec2 tex0, glm::vec2 tex1, glm::vec2 tex2) {
    this->pos0 = pos0;
    this->pos1 = pos1;
    this->pos2 = pos2;
    this->nor0 = nor0;
    this->nor1 = nor1;
    this->nor2 = nor2;
    this->tex0 = tex0;
    this->tex1 = tex1;
    this->tex2 = tex2;
    this->area = computeArea(this->tex0, this->tex1, this->tex2);
}

glm::vec3 Triangle::computeBarycentric(glm::vec2 pos) {
    // tex0 = A, tex1 = B, tex2 = C
    float a = computeArea(pos, tex1, tex2) / this->area;
    float b = computeArea(pos, tex2, tex0) / this->area;
    float c = computeArea(pos, tex0, tex1) / this->area;

    return glm::vec3(a, b, c);
}


Hit Triangle::collider(Ray& ray) {
    // float epsilon = 0.0001f;

    // glm::vec3 edge1 = pos1 - pos0;
    // glm::vec3 edge2 = pos2 - pos0;

    // glm::vec3 p_vec = glm::cross(ray.v, edge2);

    // float det = dot(edge1, p_vec);
    // if (det < epsilon && det > -epsilon) // lies in triangle plane
    //     return nullptr;
    // float inv_det = 1.0f / det;

    // // calculate U parameter and test bounds
    // glm::vec3 t_vec = ray.p - pos0;
    // float u = dot(t_vec, p_vec) * inv_det;
    // if (u < 0.0f || u > 1.0f) // barycentric
    //     return nullptr;

    // glm::vec3 q_vec = glm::cross(t_vec, edge1);

    // // calculate V parameter and test bounds
    // float v = dot(ray.v, q_vec) * inv_det;
    // if (v < 0.0f || u + v > 1.0)
    //     return nullptr;

    // // calculate W parameter, ray intersects triangle
    // float w = 1.0f - u - v;

    // // calculate t'
    // float t = dot(edge2, q_vec) * inv_det;
    // if (t < epsilon)
    //     return nullptr;

    // // compute barycentric
    // // glm::vec3 pos = w * pos0 + u * pos1 + v * pos2;
    // // glm::vec3 nor = normalize(w * nor0 + u * nor1 + v * nor2);
    // // glm::vec2 tex = w * tex0 + u * tex1 + v * tex2;

    // return Hit(t, w, u, v, this);
    return Hit();
}

Triangle Triangle::applyTransformation(glm::mat4 matrix) {
    return Triangle (
        glm::vec3(matrix * glm::vec4(pos0, 1.0f)),
        glm::vec3(matrix * glm::vec4(pos1, 1.0f)),
        glm::vec3(matrix * glm::vec4(pos2, 1.0f)),
        glm::vec3(inverse(transpose(matrix)) * glm::vec4(nor0, 0.0f)),
        glm::vec3(inverse(transpose(matrix)) * glm::vec4(nor1, 0.0f)),
        glm::vec3(inverse(transpose(matrix)) * glm::vec4(nor2, 0.0f)),
        tex0, tex1, tex2
    );
}

Tri Triangle::convertPosToTri() {
    return Tri(
		Vec(pos0.x, pos0.y, pos0.z),
		Vec(pos1.x, pos1.y, pos1.z),
		Vec(pos2.x, pos2.y, pos2.z)
	);
}

glm::vec3 Triangle::interpolatePos(float w, float u, float v) {
    return w * pos0 + u * pos1 + v * pos2;
}

glm::vec3 Triangle::interpolateNor(float w, float u, float v) {
    return w * nor0 + u * nor1 + v * nor2;
}

glm::vec2 Triangle::interpolateTex(float w, float u, float v) {
    return w * tex0 + u * tex1 + v * tex2;
}

/* PRIVATE */

float Triangle::computeArea(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2) {
    return 0.5*((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y));
}