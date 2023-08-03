#include "Rasterizer.h"
#include "MatrixStack.h"

/* PUBLIC */
Rasterizer::Rasterizer() { 
    this->width = 512;
    this->height = 512;
    init();
}

Rasterizer::Rasterizer(int width, int height) {
    this->width = width;
    this->height = height;
    init();
}

Rasterizer::~Rasterizer() { }

void Rasterizer::init() {
    // Initialize Window
    window = glfwCreateWindow(this->width, this->height, "DynamicAO", NULL, NULL);
	if(!window) {
		glfwTerminate();
	}

    glfwMakeContextCurrent(window);

    // Initialize GL
	glfwSetTime(0.0); // Initialize time.
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // Set background color.
	glEnable(GL_DEPTH_TEST); // Enable z-buffer test.

	// Initialize Shaders
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
	prog->addUniform("texture0");
	prog->setVerbose(false);

	// // Initialize Meshes
	// sphere = make_shared<Shape>();
	// sphere->loadMesh(RES_DIR + "models/sphere2.obj");
	// sphere->fitToUnitBox();
	// sphere->init();

	// // Initialize Scene
	// camera = make_shared<Camera>();
	// camera->setInitDistance(2.0f);
	// lightPos = glm::vec3(1.0f, 1.0f, 1.0f);
	// obj = Object(
	// 	sphere,
	// 	glm::vec3(0.2f, 0.2f, 0.2f),
	// 	glm::vec3(0.8f, 0.7f, 0.7f),
	// 	glm::vec3(1.0f, 0.9f, 0.8f),
	// 	200.0f
	// );
	
	GLSL::checkError(GET_FILE_LINE);
}

void Rasterizer::render() {
	// Clear framebuffer.
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// if(keyToggles[(unsigned)'c']) {
	// 	glEnable(GL_CULL_FACE);
	// } else {
	// 	glDisable(GL_CULL_FACE);
	// }
	// if(keyToggles[(unsigned)'z']) {
	// 	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	// } else {
	// 	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// }
	
	// Get current frame buffer size.
	// int w, h;
	// glfwGetFramebufferSize(window, &w, &h); // TODO: use width height of class
	// camera->setAspect((float)w/(float)h);
	
	// // Matrix stacks
	// auto P = std::make_shared<MatrixStack>();
	// auto MV = std::make_shared<MatrixStack>();
	
	// // Apply camera transforms
	// P->pushMatrix();
	// camera->applyProjectionMatrix(P);
	// MV->pushMatrix();
	// camera->applyViewMatrix(MV);
	
	// // Draw scene
	// prog->bind();
	// 	glUniform3f(prog->getUniform("lightPos"), lightPos.x, lightPos.y, lightPos.z); // send light position to GPU
	// 	obj.draw(prog, P, MV);
	// prog->unbind();
	
	// MV->popMatrix();
	// P->popMatrix();
	
	GLSL::checkError(GET_FILE_LINE);
}