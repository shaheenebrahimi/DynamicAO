#include "Material.h"

#include <iostream>

Material::Material() {
    this->kd = glm::vec3(1.0f);
    this->ks = glm::vec3(1.0f);
    this->ka = glm::vec3(1.0f);
    this->s = 100.0f;
}

Material::Material(const glm::vec3 &kd, const glm::vec3 &ks, const glm::vec3 &ka, float s) {
    this->kd = kd;
    this->ks = ks;
    this->ka = ka;
    this->s = s;
}

glm::vec3 Material::computeFrag(const glm::vec3 &ray, const glm::vec3 &pos, const glm::vec3 &nor, std::vector<std::shared_ptr<Light>>& lights) {
    glm::vec3 fragColor = ka; // ambient
    
    for (std::shared_ptr<Light> l : lights) {
        glm::vec3 l_hat = normalize(l->position - pos);
        glm::vec3 e_hat = normalize(-ray);
        glm::vec3 h_hat = normalize(l_hat + e_hat);
        glm::vec3 n_hat = normalize(nor);

        glm::vec3 diffuse = kd * std::max(0.0f, dot(l_hat, n_hat));
        glm::vec3 specular = ks * std::pow(std::max(0.0f, dot(h_hat, n_hat)), s);

        glm::vec3 color = diffuse + specular;
        fragColor += color * l->intensity;
    }
    
    return fragColor;
}

