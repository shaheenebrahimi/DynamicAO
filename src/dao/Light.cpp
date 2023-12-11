#include "Light.h"

Light::Light(const glm::vec3 &position, float intensity) {
    this->position = position;
    this->intensity = intensity;
}