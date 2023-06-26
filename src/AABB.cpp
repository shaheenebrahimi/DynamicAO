#include "AABB.h"

AABB::AABB() {
    this->minBound = glm::vec3(0,0,0); // X1, Y1, Z1
    this->maxBound = glm::vec3(0,0,0); // X2, Y2, Z2
}

AABB::AABB(glm::vec3 minBound, glm::vec3 maxBound) {
    this->minBound = minBound;
    this->maxBound = maxBound;
}


AABB::AABB(Shape* s) {
    this->minBound = s->getMinBound();
    this->maxBound = s->getMaxBound();
}

AABB::AABB(AABB* b1, AABB* b2) {
    this->minBound = glm::vec3(std::min(b1->minBound.x,b2->minBound.x), std::min(b1->minBound.y,b2->minBound.y), std::min(b1->minBound.z,b2->minBound.z));
    this->maxBound = glm::vec3(std::max(b1->maxBound.x,b2->maxBound.x), std::max(b1->maxBound.y,b2->maxBound.y), std::max(b1->maxBound.z,b2->maxBound.z));
}

Hit* AABB::collider(Ray& ray) { // could precompute division for faster performance
    std::pair<float, float> bound (0, FLT_MAX); // [tstart, tend]

    for (int i = 0; i < 3; ++i) { // iterate through each dimension: x, y, z
        if (ray.v[i] == 0 && (ray.p.x < this->minBound[i] || ray.p[i] > this->maxBound[i])) return nullptr;
        float t1 = (this->minBound[i] - ray.p[i]) / ray.v[i];
        float t2 = (this->maxBound[i] - ray.p[i]) / ray.v[i];
        if (t1 > t2) std::swap(t1, t2);
        if (t1 > bound.first) bound.first = t1;
        if (t2 < bound.second) bound.second = t2;
    }

    // include epsilon?
    if (bound.first > bound.second) return nullptr; // no intersection
    if (bound.second < 0) return nullptr; // behind origin

    if (bound.first > 0) return new Hit (ray.p + ray.v*bound.first, glm::vec3(0), glm::vec2(0), bound.first); // intersect tstart
    else return new Hit (ray.p + ray.v*bound.second, glm::vec3(0), glm::vec2(1), bound.second); // intersect tend
}