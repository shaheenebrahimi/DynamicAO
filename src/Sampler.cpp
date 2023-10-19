#include "Sampler.h"

Sampler::Sampler(std::shared_ptr<Object> obj) {
    this->obj = obj;
}

void Sampler::sample(int count, const std::string &filename) {
    std::ofstream ofs (filename);
    std::uniform_real_distribution<float> randomFloats;
    std::default_random_engine generator;
    for (int i = 0; i < count; ++i) {
        for (auto tri : obj->mesh->getTriangles()) {
            randomFloats = std::uniform_real_distribution<float>(0.0, 1.0);
            float a = randomFloats(generator); // a
            randomFloats = std::uniform_real_distribution<float>(0.0, 1.0-a);
            float b = randomFloats(generator); // b
            float c = 1 - (a + b); // c
            glm::vec2 point = a*tri->tex0 + b*tri->tex1 + c*tri->tex2;
            int val = (int)obj->tex->getPixel((int)obj->tex->getHeight()*point.y, (int)obj->tex->getWidth()*point.x);
            ofs << point.x << "," << point.y << "," << val << std::endl;
        }
    }
    ofs.close();
}
