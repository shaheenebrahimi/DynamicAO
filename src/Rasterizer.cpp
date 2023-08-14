#include "Rasterizer.h"
#include "MatrixStack.h"

bool keyToggles[256] = {false}; // only for English keyboards!
Rasterizer* raster = nullptr;

// This function is for handling key clicks
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

// This function is for handling mouse clicks
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	if (action == GLFW_PRESS) {
		bool shift = (mods & GLFW_MOD_SHIFT) != 0;
		bool ctrl  = (mods & GLFW_MOD_CONTROL) != 0;
		bool alt   = (mods & GLFW_MOD_ALT) != 0;
		raster->getCam()->mouseClicked((float)xmouse, (float)ymouse, shift, ctrl, alt);
	}
}

// This function is called when the mouse moves
void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if(state == GLFW_PRESS) {
		raster->getCam()->mouseMoved((float)xmouse, (float)ymouse);
	}
}

// This function is for handling chars in key press
void char_callback(GLFWwindow *window, unsigned int key)
{
	keyToggles[key] = !keyToggles[key];
}


/* PUBLIC */
Rasterizer::Rasterizer() { 
    this->width = 1366;
    this->height = 768;
	raster = this;
    // init();
}

Rasterizer::Rasterizer(int width, int height) {
    this->width = width;
    this->height = height;
	raster = this;

    // init();
}

Rasterizer::~Rasterizer() { }

int Rasterizer::init() {
	// Set error callback.
	// glfwSetErrorCallback(error_callback);
	
	// Initialize the library.
	if(!glfwInit()) {
		return -1;
	}

    // Initialize Window
    window = glfwCreateWindow(this->width, this->height, "DynamicAO", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
    glfwMakeContextCurrent(window);

	// Initiialize GLEW
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW" << std::endl;
	}
	
	glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
	std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
	GLSL::checkVersion();

	// Add Callbacks
	glfwSwapInterval(1); // Set vsync.
	glfwSetKeyCallback(window, key_callback); // Set keyboard callback.
	glfwSetCharCallback(window, char_callback); // Set char callback.
	glfwSetCursorPosCallback(window, cursor_position_callback); // Set cursor position callback.
	glfwSetMouseButtonCallback(window, mouse_button_callback); // Set mouse button callback.

    // // Initialize GL
	glfwSetTime(0.0); // Initialize time.
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // Set background color.
	glEnable(GL_DEPTH_TEST); // Enable z-buffer test.

	// // Initialize Shaders
	prog = std::make_shared<Program>();
	prog->setShaderNames(RES_DIR + "shaders/vert.glsl", RES_DIR + "shaders/frag.glsl"); // TODO: make dynamic?
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
	prog->addUniform("aoTexture");
	prog->setVerbose(false);

	// Initialize Camera
	camera = std::make_shared<RasterCam>();
	camera->setInitDistance(-3.0f);

	// Send Objects to GPU
	for (auto obj : scn.objects) { // TODO texture issue
		if (obj->tex) {
			obj->tex->init();
			obj->tex->setUnit(textureCount++);
			obj->tex->setWrapModes(GL_REPEAT, GL_REPEAT);
		}
		obj->mesh->loadBuffers();
	}
	
	GLSL::checkError(GET_FILE_LINE);
	return 0;
}

void Rasterizer::render() {
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
	int w, h;
	glfwGetFramebufferSize(window, &w, &h); // TODO: use width height of class
	camera->setAspect((float)w/(float)h);
	
	// // Matrix stacks
	auto P = std::make_shared<MatrixStack>();
	auto MV = std::make_shared<MatrixStack>();
	
	// Apply camera transforms
	P->pushMatrix();
	camera->applyProjectionMatrix(P);
	MV->pushMatrix();
	camera->applyViewMatrix(MV);
	
	// Draw scene
	prog->bind();
		auto light = scn.lights[0]; // TODO: more lights
		glUniform3f(prog->getUniform("lightPos"), light->position.x, light->position.y, light->position.z); // send light position to GPU
		for (auto obj : scn.objects) {
			MV->pushMatrix();
				MV->multMatrix(obj->transform);
				obj->draw(prog, P, MV);
			MV->popMatrix();
		}
	prog->unbind();
	
	MV->popMatrix();
	P->popMatrix();
	
	GLSL::checkError(GET_FILE_LINE);
}

void Rasterizer::run() {
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
}