#pragma once
#ifndef RENDERER_H
#define RENDERER_H

#include <glm/glm.hpp>

#include <memory>

#include "Object.h"
#include "Light.h"
#include "Hit.h"
#include "Ray.h"
#include "Camera.h"
#include "Scene.h"
#include "Image.h"
#include "Mesh.h"

// abstract class
class Renderer {
public:
    Scene scn;
    int resolution;
    std::string filename;
    std::shared_ptr<Image> img;
    
    // shootray
};

#endif