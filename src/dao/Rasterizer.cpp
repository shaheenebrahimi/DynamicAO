#include "Rasterizer.h"
#include "MatrixStack.h"

#include <cuda_gl_interop.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define RAD_TO_DEG 180.0/3.1415
#define DEG_TO_RAD 3.1415/180.0 

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
	if (ImGui::GetIO().WantCaptureMouse) return; // Interacting with UI

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
	if (ImGui::GetIO().WantCaptureMouse) return; // Interacting with UI
	
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
	this->window = nullptr;
	raster = this;
}

Rasterizer::Rasterizer(int width, int height) {
    this->width = width;
    this->height = height;
	this->window = nullptr;
	raster = this;
}

Rasterizer::~Rasterizer() { }

int Rasterizer::init() {
	// Set error callback.
	// glfwSetErrorCallback(error_callback);
	// 
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

	// Initialize GLEW
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
	glClearColor(0.5f, 0.5f, 0.5f, 1.0f); // Set background color.
	glEnable(GL_DEPTH_TEST); // Enable z-buffer test.

	// Initialize ImGUI
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark(); //ImGui::StyleColorsLight();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(nullptr);

	// // Initialize Shaders
	prog = std::make_shared<Program>();
	prog->setShaderNames(RES_DIR + "shaders/vert.glsl", RES_DIR + "shaders/frag.glsl");
	prog->setVerbose(true);
	prog->init();
	prog->addAttribute("aPos");
	prog->addAttribute("aNor");
	prog->addAttribute("aTex");
	prog->addAttribute("aOcc");
	prog->addUniform("P");
	prog->addUniform("MV");
	prog->addUniform("itMV");
	prog->addUniform("lightPos");
	prog->addUniform("ka");
	prog->addUniform("kd");
	prog->addUniform("ks");
	prog->addUniform("s");
	prog->addUniform("aoTexture");
	prog->addUniform("genTexture");
	prog->addUniform("groundTruth");
	prog->setVerbose(false);

	// Initialize Camera
	camera = std::make_shared<RasterCam>();
	camera->setInitDistance(-50.0f);
	//camera->setInitPos(0.0f, 110.0f, 310.0f);

	// Send Objects to GPU
	for (std::shared_ptr<Object> obj : scn.objects) { // TODO: texture issue
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

void Rasterizer::clear() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (keyToggles[(unsigned)'c']) {
		glEnable(GL_CULL_FACE);
	}
	else {
		glDisable(GL_CULL_FACE);
	}
	if (keyToggles[(unsigned)'z']) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void Rasterizer::renderUI() {
	static float Ex = 0.0f;
	static float Ey = 0.0f;
	static float Ez = 0.0f;
	static int bone = 1;
	bool rotated = false;

	// Create new UI Frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Bone Orientations"); // Create a window and append into it.

	// TODO: Select object in scene
	std::shared_ptr<Object> target = scn.objects[0];
	const int numBones = target->mesh->getBoneCount();

	// Bone cycle
	ImGui::Text("Select Bone");
	ImGui::SameLine();
	if (ImGui::ArrowButton("prev", ImGuiDir_Left)) {// Buttons return true when clicked (most widgets return true when edited/activated)
		bone = (bone < 1) ? 0 : bone - 1;
		glm::vec3 rotation = target->mesh->getBoneRotation(bone);
		Ex = rotation.x * RAD_TO_DEG; Ey = rotation.y * RAD_TO_DEG; Ez = rotation.z * RAD_TO_DEG;
	}
	ImGui::SameLine();
	ImGui::Text("%d", bone);
	ImGui::SameLine();
	if (ImGui::ArrowButton("next", ImGuiDir_Right)) {
		++bone %= numBones;
		glm::vec3 rotation = target->mesh->getBoneRotation(bone);
		Ex = rotation.x * RAD_TO_DEG; Ey = rotation.y * RAD_TO_DEG; Ez = rotation.z * RAD_TO_DEG;
	}

	// Orientation sliders
	ImGui::Text("Relative Orientation");
	if (ImGui::SliderFloat("Euler X", &Ex, -180.0f, 180.0f))
		rotated = true;
	if (ImGui::SliderFloat("Euler Y", &Ey, -180.0f, 180.0f))
		rotated = true;
	if (ImGui::SliderFloat("Euler Z", &Ez, -180.0f, 180.0f))
		rotated = true;
	if (rotated)
		target->mesh->setBone(bone, glm::vec3(Ex * DEG_TO_RAD, Ey * DEG_TO_RAD, Ez * DEG_TO_RAD));

	// Mode
	ImGui::Checkbox("Use Texture", &keyToggles[(unsigned)' ']);

	//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
	ImGui::End();

	ImGui::EndFrame();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Rasterizer::render() {
	// Get current frame buffer size.
	int w, h;
	glfwGetFramebufferSize(window, &w, &h); // TODO: use width height of class
	camera->setAspect((float)w/(float)h);
	
	 // Matrix stacks
	auto P = std::make_shared<MatrixStack>();
	auto MV = std::make_shared<MatrixStack>();
	
	// Apply camera transforms
	P->pushMatrix();
	camera->applyProjectionMatrix(P);
	MV->pushMatrix();
	camera->applyViewMatrix(MV);
	
	// Draw scene
	prog->bind();
		glUniform1i(prog->getUniform("groundTruth"), keyToggles[(unsigned)' ']);
		std::shared_ptr<Light> light = scn.lights[0]; // TODO: more lights
		glUniform3f(prog->getUniform("lightPos"), light->position.x, light->position.y, light->position.z); // send light position to GPU
		for (std::shared_ptr<Object> obj : scn.objects) {
			MV->pushMatrix();
				MV->multMatrix(obj->transform);
				obj->update();
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
		// Poll for and process events.
		glfwPollEvents();

		// Clear framebuffer and Render scene.
		clear();
		render();
		renderUI();

		// Swap front and back buffers.
		glfwSwapBuffers(window);
	}
	
	// Quit program.
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}