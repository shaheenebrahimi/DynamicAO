#pragma once
#ifndef RASTERIZER_H
#define RASTERIZER_H

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <random>

#include "GLSL.h"
#include "Object.h"
#include "Light.h"
#include "Hit.h"
#include "Ray.h"
#include "Camera.h"
#include "Scene.h"
#include "Image.h"
#include "Mesh.h"
#include "Program.h"

class Rasterizer {
public:
    int width; int height;
    Rasterizer();
    Rasterizer(int width, int height);
    ~Rasterizer();
    void init();
    void render();
    void setScene(Scene& scn) { this->scn = scn; init(); }

private:
    const std::string RES_DIR = "../resources/"; // Where the resources are loaded from
    GLFWwindow *window; // main application window
    std::shared_ptr<Program> prog;
    Scene scn;
};


#endif