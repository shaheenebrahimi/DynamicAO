#include "Object.h"

Object::Object() {
    this->T = glm::mat4(1);
    this->R = glm::mat4(1);
    this->S = glm::mat4(1);
    this->transform = glm::mat4(1);
}

Object::Object(const std::string &objPath) {
    this->mesh = std::make_shared<Mesh>(objPath);
    this->mat = std::make_shared<Material>();
    this->T = glm::mat4(1);
    this->R = glm::mat4(1);
    this->S = glm::mat4(1);
    this->transform = glm::mat4(1);
}

Object::Object(const std::string &objPath, std::shared_ptr<Material> mat) { 
    this->mesh = std::make_shared<Mesh>(objPath);
    this->mat = mat;
    this->T = glm::mat4(1);
    this->R = glm::mat4(1);
    this->S = glm::mat4(1);
    this->transform = glm::mat4(1);
}

Object::Object(const std::string &objPath, const glm::vec3 &pos, const glm::vec4 &rot, const glm::vec3 &sc, std::shared_ptr<Material> mat) { 
    this->mesh = std::make_shared<Mesh>(objPath);
    this->mat = mat;
    setPosition(pos);
    setRotation(rot);
    setScale(sc);
    this->transform = computeTransform();
    mesh->setTransform(this->transform);
}

void Object::translate(const glm::vec3 &translation) {
    this->T = glm::translate(this->T, translation);
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::rotate(const glm::vec4 &rotation) {
    this->R = glm::rotate(this->R, rotation[0], glm::vec3(rotation[1], rotation[2], rotation[3]));
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::scale(const glm::vec3 &scale) {
    this->S = glm::scale(this->S, scale);
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::setMaterial(const glm::vec3 &kd, const glm::vec3 &ks, const glm::vec3 &ka, float s) {
    this->mat = std::make_shared<Material>(kd, ks, ka, s);
}

void Object::setPosition(const glm::vec3 &position) {
    this->T = glm::translate(glm::mat4(1.0f), position);
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::setRotation(const glm::vec4 &rotation) {
    this->R = glm::rotate(glm::mat4(1.0f), rotation[0], glm::vec3(rotation[1], rotation[2], rotation[3]));
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::setScale(const glm::vec3 &scale) {
    this->S = glm::scale(glm::mat4(1.0f), scale);
    this->transform = computeTransform();
    this->mesh->setTransform(transform);
}

void Object::addTexture(const std::string &texPath) {
    tex = std::make_shared<Texture>();
	tex->setFilename(texPath);
}

glm::mat4 Object::computeTransform() {
    return T * R * S;
}

void Object::draw(std::shared_ptr<Program> prog, std::shared_ptr<MatrixStack> P, std::shared_ptr<MatrixStack> MV) {
    glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
    glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
    glUniformMatrix4fv(prog->getUniform("itMV"), 1, GL_FALSE, glm::value_ptr(glm::transpose(glm::inverse(glm::mat4(MV->topMatrix())))));
    glUniform3f(prog->getUniform("ka"), mat->ka.x, mat->ka.y, mat->ka.z);
    glUniform3f(prog->getUniform("kd"), mat->kd.x, mat->kd.y, mat->kd.z);
    glUniform3f(prog->getUniform("ks"), mat->ks.x, mat->ks.y, mat->ks.z);
    glUniform1f(prog->getUniform("s"), mat->s);
    tex->bind(prog->getUniform("aoTexture"));
    mesh->drawMesh(prog);
    tex->unbind();
}