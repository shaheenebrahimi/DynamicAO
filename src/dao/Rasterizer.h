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
//#include "Hit.h"
#include "Ray.h"
// #include "Camera.h"
#include "Scene.h"
#include "Image.h"
#include "Mesh.h"
#include "Program.h"
#include "RasterCam.h"

class Rasterizer {
public:
    int width; int height;
    Rasterizer();
    Rasterizer(int width, int height);
    ~Rasterizer();
    int init();
    void run();
    void setScene(Scene& scn) { this->scn = scn; }
    std::shared_ptr<RasterCam> getCam() { return camera; }
    
private:
    void render();
    
    const std::string RES_DIR =
        #ifdef _WIN32
        // on windows, visual studio creates _two_ levels of build dir
        "../../../resources/"
        #else
        // on linux, common practice is to have ONE level of build dir
        "../../resources/"
        #endif
    ;
    
    GLint textureCount = 0;
    GLFWwindow *window; // main application window
    std::shared_ptr<Program> prog;
    std::shared_ptr<RasterCam> camera;
    Scene scn;
};


#endif