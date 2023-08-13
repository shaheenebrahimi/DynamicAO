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
    
private:
    void render();

    const std::string RES_DIR = "../resources/"; // Where the resources are loaded from
    GLint textureCount = 0;
    GLFWwindow *window; // main application window
    std::shared_ptr<Program> prog;
    std::shared_ptr<RasterCam> camera;
    Scene scn;

    // static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods); // Set keyboard callback.
	// static void char_callback(GLFWwindow *window, unsigned int key); // Set char callback.
	// static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse); // Set cursor position callback.
	// static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods); // Set mouse button callback.

    // static bool keyToggles[256] = {false}; // only for English keyboards!

};


#endif