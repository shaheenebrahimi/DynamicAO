#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include "Camera.h"

Camera::Camera() {
	this->resolution = 512.0f;
	this->position = glm::vec3(0.0f, 0.0f, 5.0f);
	this->lookat = glm::vec3(0.0f, 0.0f, -1.0f);
	this->up = glm::vec3(0.0f, 1.0f, 0.0f);
	this->fov = (float)(45.0f*M_PI/180.0f);
	this->focalDist = 1.0f;
	this->init();
}

Camera::Camera(const glm::vec3 &position, const glm::vec3 &lookat, const glm::vec3 &up, float fov) {
	this->resolution = 512.0f;
	this->position = position;
	this->lookat = normalize(lookat);
	this->up = normalize(up);
	this->fov = fov;
	this->focalDist = 1.0f;
	this->init();
}

Camera::~Camera() { }

void Camera::init() {
	float bound = focalDist * tan(fov/2);
	this->screen = position + focalDist*lookat;
	this->axis1 = cross(lookat, up);
	this->axis2 = up;
	this->startBound = screen - (bound * axis1) - (bound * axis2);
	this->stepSize = (2*bound) / float(resolution);
	this->offset = stepSize/2; // get middle of pixel
}

void Camera::setResolution(int resolution) {
	this->resolution = resolution;
	init();
}

// get pixel coordinate in world space
Ray Camera::getRay(int r, int c) {
	glm::vec3 pixelCoord = startBound + (c*stepSize + offset) * axis1 + (r*stepSize + offset) * axis2;
	glm::vec3 direction = normalize(pixelCoord - position);
	return Ray(this->position, direction);
}

// #define _USE_MATH_DEFINES
// #include <cmath> 
// #include <iostream>
// #include <glm/gtc/matrix_transform.hpp>
// #include "Camera.h"
// #include "MatrixStack.h"

// Camera::Camera() :
// 	aspect(1.0f),
// 	fovy((float)(45.0*M_PI/180.0)),
// 	znear(0.1f),
// 	zfar(1000.0f),
// 	rotations(0.0, 0.0),
// 	translations(0.0f, 0.0f, -5.0f),
// 	rfactor(0.01f),
// 	tfactor(0.001f),
// 	sfactor(0.005f)
// {
// }

// Camera::~Camera()
// {
// }

// void Camera::mouseClicked(float x, float y, bool shift, bool ctrl, bool alt)
// {
// 	mousePrev.x = x;
// 	mousePrev.y = y;
// 	if(shift) {
// 		state = Camera::TRANSLATE;
// 	} else if(ctrl) {
// 		state = Camera::SCALE;
// 	} else {
// 		state = Camera::ROTATE;
// 	}
// }

// void Camera::mouseMoved(float x, float y)
// {
// 	glm::vec2 mouseCurr(x, y);
// 	glm::vec2 dv = mouseCurr - mousePrev;
// 	switch(state) {
// 		case Camera::ROTATE:
// 			rotations += rfactor * dv;
// 			break;
// 		case Camera::TRANSLATE:
// 			translations.x -= translations.z * tfactor * dv.x;
// 			translations.y += translations.z * tfactor * dv.y;
// 			break;
// 		case Camera::SCALE:
// 			translations.z *= (1.0f - sfactor * dv.y);
// 			break;
// 	}
// 	mousePrev = mouseCurr;
// }

// void Camera::applyProjectionMatrix(std::shared_ptr<MatrixStack> P) const
// {
// 	// Modify provided MatrixStack
// 	P->multMatrix(glm::perspective(fovy, aspect, znear, zfar));
// }

// void Camera::applyViewMatrix(std::shared_ptr<MatrixStack> MV) const
// {
// 	MV->translate(translations);
// 	MV->rotate(rotations.y, glm::vec3(1.0f, 0.0f, 0.0f));
// 	MV->rotate(rotations.x, glm::vec3(0.0f, 1.0f, 0.0f));
// }

// Ray Camera::getRay(int r, int c) {
// 	glm::vec3 pixelCoord = startBound + (c*stepSize + offset) * axis1 + (r*stepSize + offset) * axis2;
// }