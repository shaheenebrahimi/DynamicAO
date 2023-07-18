#include "Object.h"

Object::Object() {
    this->T = glm::mat4(1);
    this->R = glm::mat4(1);
    this->S = glm::mat4(1);
}

Object::Object(const std::string& objPath) {
    this->mesh = std::make_shared<Mesh>(objPath);
    this->mat = std::make_shared<Material>();
    this->T = glm::mat4(1);
    this->R = glm::mat4(1);
    this->S = glm::mat4(1);
}

Object::Object(const std::string& objPath, std::shared_ptr<Material> mat) { 
    this->mesh = std::make_shared<Mesh>(objPath);
    this->mat = mat;
    this->T = glm::mat4(1);
    this->R = glm::mat4(1);
    this->S = glm::mat4(1);
}

Object::Object(const std::string& objPath, glm::vec3 pos, glm::vec4 rot, glm::vec3 sc, std::shared_ptr<Material> mat) { 
    this->mesh = std::make_shared<Mesh>(objPath);
    this->mat = mat;
    setPosition(pos);
    setRotation(rot);
    setScale(sc);
    mesh->setTransform(computeTransform());
}

void Object::translate(glm::vec3 translation) {
    this->T = glm::translate(this->T, translation);
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::rotate(glm::vec4 rotation) {
    this->R = glm::rotate(this->R, rotation[0], glm::vec3(rotation[1], rotation[2], rotation[3]));
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::scale(glm::vec3 scale) {
    this->S = glm::scale(this->S, scale);
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::setMaterial(glm::vec3 kd, glm::vec3 ks, glm::vec3 ka, float s) {
    this->mat = std::make_shared<Material>(kd, ks, ka, s);
}

void Object::setPosition(glm::vec3 position) {
    this->T = glm::translate(glm::mat4(1.0f), position);
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::setRotation(glm::vec4 rotation) {
    this->R = glm::rotate(glm::mat4(1.0f), rotation[0], glm::vec3(rotation[1], rotation[2], rotation[3]));
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::setScale(glm::vec3 scale) {
    this->S = glm::scale(glm::mat4(1.0f), scale);
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

glm::mat4 Object::computeTransform() {
    return T * R * S;
}