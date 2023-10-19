#pragma once
#ifndef SAMPLER_H
#define SAMPLER_H

#include "Texture.h"
#include "Object.h"
#include <memory>
#include <iostream>
#include <random>
#include <fstream>
#include <glm/glm.hpp>

class Sampler {
public:
    Sampler(std::shared_ptr<Object> obj);
    void sample(int count = 10000, const std::string &filename = "out.txt");

private:
    std::shared_ptr<Object> obj;
};

#endif