#include "Object.h"

Object::Object() {
    this->mesh = nullptr;
    this->mat = Material();
}

Object::Object(std::shared_ptr<Shape> mesh, glm::vec3 ka, glm::vec3 kd, glm::vec3 ks, float s) {
    this->mesh = mesh;
    this->mat = Material(ka, kd, ks, s);
}

Object::~Object() {}

void Object::draw(std::shared_ptr<Program> prog, std::shared_ptr<MatrixStack> P, std::shared_ptr<MatrixStack> MV) {
    glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
    glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
    glUniformMatrix4fv(prog->getUniform("itMV"), 1, GL_FALSE, glm::value_ptr(glm::transpose(glm::inverse(glm::mat4(MV->topMatrix())))));
    glUniform3f(prog->getUniform("ka"), mat.ka.x, mat.ka.y, mat.ka.z);
    glUniform3f(prog->getUniform("kd"), mat.kd.x, mat.kd.y, mat.kd.z);
    glUniform3f(prog->getUniform("ks"), mat.ks.x, mat.ks.y, mat.ks.z);
    glUniform1f(prog->getUniform("s"), mat.s);
    mesh->draw(prog);
}