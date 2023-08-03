#include <cassert>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "GLSL.h"
#include "Program.h"
#include "MatrixStack.h"
#include "Camera.h"
#include "Shape.h"
#include "Object.h"
#include "Texture.h"

using namespace std;

GLFWwindow *window; // Main application window
const string RES_DIR = "../resources/"; // Where the resources are loaded from
enum occlusions { GTAO, SSAO, DAO }; // select different ambient occlusion methods: ground truth, screen space, dynamic (ours)
int occlusionMethod = GTAO;
bool cameraMovable = false;

shared_ptr<Program> prog;
shared_ptr<Texture> texture0;
shared_ptr<Camera> camera;
shared_ptr<Shape> sphere;
Object obj;
glm::vec3 lightPos;

bool keyToggles[256] = {false}; // only for English keyboards!

// This function is called when a GLFW error occurs
static void error_callback(int error, const char *description)
{
	cerr << description << endl;
}

// This function is called when a key is pressed
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

// This function is called when the mouse is clicked
static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	if(action == GLFW_PRESS) {
		bool shift = (mods & GLFW_MOD_SHIFT) != 0;
		bool ctrl  = (mods & GLFW_MOD_CONTROL) != 0;
		bool alt   = (mods & GLFW_MOD_ALT) != 0;
		camera->mouseClicked((float)xmouse, (float)ymouse, shift, ctrl, alt);
	}
}

// This function is called when the mouse moves
static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if(state == GLFW_PRESS) {
		camera->mouseMoved((float)xmouse, (float)ymouse);
	}
}

// This function is for handling chars in key press
static void char_callback(GLFWwindow *window, unsigned int key)
{
	keyToggles[key] = !keyToggles[key];
}

// If the window is resized, capture the new size and reset the viewport
static void resize_callback(GLFWwindow *window, int width, int height)
{
	glViewport(0, 0, width, height);
}

// This function is called once to initialize the scene and OpenGL
static void init()
{
	// Initialize GL
	glfwSetTime(0.0); // Initialize time.
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // Set background color.
	glEnable(GL_DEPTH_TEST); // Enable z-buffer test.

	// Initialize Shaders
	prog = make_shared<Program>();
	prog->setShaderNames(RES_DIR + "shaders/vert.glsl", RES_DIR + "shaders/frag.glsl");
	prog->setVerbose(true);
	prog->init();
	prog->addAttribute("aPos");
	prog->addAttribute("aNor");
	prog->addAttribute("aTex");
	prog->addUniform("P");
	prog->addUniform("MV");
	prog->addUniform("itMV");
	prog->addUniform("lightPos");
	prog->addUniform("ka");
	prog->addUniform("kd");
	prog->addUniform("ks");
	prog->addUniform("s");
	prog->addUniform("texture0");
	prog->setVerbose(false);

	// Initialize Textures
	texture0 = make_shared<Texture>();
	texture0->setFilename(RES_DIR + "/textures/aoTexture.png");
	texture0->init();
	texture0->setUnit(0);
	texture0->setWrapModes(GL_REPEAT, GL_REPEAT);

	// Initialize Meshes
	sphere = make_shared<Shape>();
	sphere->loadMesh(RES_DIR + "models/sphere2.obj");
	sphere->fitToUnitBox();
	sphere->init();

	// Initialize Scene
	camera = make_shared<Camera>();
	camera->setInitDistance(2.0f);
	lightPos = glm::vec3(1.0f, 1.0f, 1.0f);
	obj = Object(
		sphere,
		glm::vec3(0.2f, 0.2f, 0.2f),
		glm::vec3(0.8f, 0.7f, 0.7f),
		glm::vec3(1.0f, 0.9f, 0.8f),
		200.0f
	);
	
	GLSL::checkError(GET_FILE_LINE);
}

// This function is called every frame to draw the scene.
static void render()
{
	// Clear framebuffer.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if(keyToggles[(unsigned)'c']) {
		glEnable(GL_CULL_FACE);
	} else {
		glDisable(GL_CULL_FACE);
	}
	if(keyToggles[(unsigned)'z']) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
	// Get current frame buffer size.
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	camera->setAspect((float)width/(float)height);
	
	// Matrix stacks
	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();
	
	// Apply camera transforms
	P->pushMatrix();
	camera->applyProjectionMatrix(P);
	MV->pushMatrix();
	camera->applyViewMatrix(MV);
	
	// Draw scene
	prog->bind();
		glUniform3f(prog->getUniform("lightPos"), lightPos.x, lightPos.y, lightPos.z); // send light position to GPU
		obj.draw(prog, P, MV);
	prog->unbind();
	
	MV->popMatrix();
	P->popMatrix();
	
	GLSL::checkError(GET_FILE_LINE);
}

int main(int argc, char **argv)
{
	// Set up arguments
	if (argc >= 2) {
		occlusionMethod = atoi(argv[1]);
		if (occlusionMethod > DAO) {
			cout << "Not a vailid occlusion method" << endl;
			return -1;
		}
		else if (occlusionMethod != GTAO) {
			cameraMovable = true;
		}
	}

	// Set error callback.
	glfwSetErrorCallback(error_callback);
	
	// Initialize the library.
	if(!glfwInit()) {
		return -1;
	}
	
	// Create a windowed mode window and its OpenGL context.
	window = glfwCreateWindow(640, 480, "DynamicAO", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
	
	// Make the window's context current.
	glfwMakeContextCurrent(window);
	
	// Initialize GLEW.
	glewExperimental = true;
	if(glewInit() != GLEW_OK) {
		cerr << "Failed to initialize GLEW" << endl;
		return -1;
	}
	glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
	cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
	cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	GLSL::checkVersion();
	
	glfwSwapInterval(1); // Set vsync.
	glfwSetKeyCallback(window, key_callback); // Set keyboard callback.
	glfwSetCharCallback(window, char_callback); // Set char callback.
	glfwSetCursorPosCallback(window, cursor_position_callback); // Set cursor position callback.
	glfwSetMouseButtonCallback(window, mouse_button_callback); // Set mouse button callback.
	glfwSetFramebufferSizeCallback(window, resize_callback); // Set the window resize call back.
	
	// Initialize scene.
	init();
	
	// Loop until the user closes the window.
	while(!glfwWindowShouldClose(window)) {
		// Render scene.
		render();
		// Swap front and back buffers.
		glfwSwapBuffers(window);
		// Poll for and process events.
		glfwPollEvents();
	}
	
	// Quit program.
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
