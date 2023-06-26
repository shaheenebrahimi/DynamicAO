#include "Sphere.h"

Sphere::Sphere() {
    this->position = glm::vec3(0.0f);
    this->rotation = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
    this->scale = glm::vec3(1.0f);
    this->mat = nullptr;
    this->transform = glm::mat4(1);
    this->radius = 1.0f;
}

Sphere::Sphere(glm::vec3 position, glm::vec3 scale, float radius, Material* mat) {
    this->position = position;
    this->scale = scale;
    this->radius = radius;
    this->mat = mat;
}

Hit* Sphere::collider(Ray& ray) {
    glm::vec3 pc = ray.p - position; // center is position of sphere
    float a = dot(ray.v, ray.v);
    float b = 2 * dot(ray.v, pc);
    float c = dot(pc, pc) - pow(radius, 2);
    float d = pow(b, 2) - (4 * a * c);
    if (d > 0) {
        float t1 = (-b + pow(d, 0.5)) / (2 * a);
        float t2 = (-b - pow(d, 0.5)) / (2 * a);
        glm::vec3 x1 = ray.p + t1*ray.v;
        glm::vec3 x2 = ray.p + t2*ray.v;
        // return closest hit in front of camera
        float epsilon = 0.001f; // not same surface
        if (t1 > epsilon && t2 > epsilon) { // both valid
            if (t1 < t2) // return closest
                return new Hit (x1, (x1 - position) / radius, glm::vec2(0), t1);
            else
                return new Hit (x2, (x2 - position) / radius, glm::vec2(0), t2);
        }
        else if (t1 > epsilon) { // only t1 valid
            return new Hit (x1, (x1 - position) / radius, glm::vec2(0), t1);
        }
        else if (t2 > epsilon) { // only t2 valid
            return new Hit (x2, (x2 - position) / radius, glm::vec2(0), t2);
        }
    }
    return nullptr;
}