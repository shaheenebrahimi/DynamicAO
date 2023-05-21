#define _USE_MATH_DEFINES
#include <cassert>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "GLSL.h"

using namespace std;

GLFWwindow *window; // Main application window
string RESOURCE_DIR = "./"; // Where the resources are loaded from
bool OFFLINE = false;

const int windowWidth = 1024;
const int windowHeight = 1024;

bool keyToggles[256] = {false}; // only for English keyboards!

// This function is called when a GLFW error occurs
static void error_callback(int error, const char *description) {
	cerr << description << endl;
}

// This function is called when a key is pressed
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

// This function is called when the mouse is clicked
static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
	// Get the current mouse position.
	// double xmouse, ymouse;
	// glfwGetCursorPos(window, &xmouse, &ymouse);
	// // Get current window size.
	// int width, height;
	// glfwGetWindowSize(window, &width, &height);
	// if(action == GLFW_PRESS) {
	// 	bool shift = (mods & GLFW_MOD_SHIFT) != 0;
	// 	bool ctrl  = (mods & GLFW_MOD_CONTROL) != 0;
	// 	bool alt   = (mods & GLFW_MOD_ALT) != 0;
	// 	camera->mouseClicked((float)xmouse, (float)ymouse, shift, ctrl, alt);
	// }
}

// This function is called when the mouse moves
static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse) {
	// int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	// if(state == GLFW_PRESS) {
	// 	camera->mouseMoved((float)xmouse, (float)ymouse);
	// }
}

// This function handles key character logic
static void char_callback(GLFWwindow *window, unsigned int key) {
	// switch(key) {
	// 	case 'w': { // move camera forward
	// 		camera->moveForward();
	// 		break;
	// 	}
	// 	case 'a': { // move camera left
	// 		camera->moveLeft();
	// 		break;
	// 	}
	// 	case 's': { // move camera backward
	// 		camera->moveBackward();
	// 		break;
	// 	}
	// 	case 'd': { // move camera right
	// 		camera->moveRight();
	// 		break;
	// 	}
	// 	case 'z': { // zoom in
	// 		camera->zoomIn();
	// 		break;
	// 	}
	// 	case 'Z': { // zoom out
	// 		camera->zoomOut();
	// 		break;
	// 	}
	// }
}

// https://lencerf.github.io/post/2019-09-21-save-the-opengl-rendering-to-image-file/
static void saveImage(const char *filepath, GLFWwindow *w) {
	int width, height;
	glfwGetFramebufferSize(w, &width, &height);
	GLsizei nrChannels = 3;
	GLsizei stride = nrChannels * width;
	stride += (stride % 4) ? (4 - stride % 4) : 0;
	GLsizei bufferSize = stride * height;
	std::vector<char> buffer(bufferSize);
	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadBuffer(GL_BACK);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
	stbi_flip_vertically_on_write(true);
	int rc = stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
	if(rc) {
		cout << "Wrote to " << filepath << endl;
	} else {
		cout << "Couldn't write to " << filepath << endl;
	}
}

// This function is called once to initialize the scene and OpenGL
static void init() {
	srand(time(0)); // Initialize seed.
	glfwSetTime(0.0); // Initialize time.
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color.
	glEnable(GL_DEPTH_TEST); // Enable z-buffer test.
	
	GLSL::checkError(GET_FILE_LINE);
}

// This function is called every frame to draw the scene.
static void render() {
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

	GLSL::checkError(GET_FILE_LINE);
	
	// if(OFFLINE) {
	// 	saveImage("output.png", window);
	// 	GLSL::checkError(GET_FILE_LINE);
	// 	glfwSetWindowShouldClose(window, true);
	// }
}

int main(int argc, char **argv)
{
	// Optional argument
	// if(argc >= 1) {
	// 	OFFLINE = atoi(argv[1]) != 0;
	// }

	glfwSetErrorCallback(error_callback); // Set error callback.
	
	// Initialize the library.
	if(!glfwInit()) {
		return -1;
	}

	// Create a windowed mode window and its OpenGL context.
	window = glfwCreateWindow(windowWidth, windowHeight, "DynamicAO", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window); // Make the window's context current.
	
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
	
	init(); // Initialize scene.
	
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
