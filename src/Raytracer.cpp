#include "Raytracer.h"

/* PUBLIC */
Raytracer::Raytracer() { 
    this->filename = "out.png";
    this->resolution = 1024;
    init();
}

Raytracer::Raytracer(std::string& filename, int resolution) {
    this->filename = filename;
    this->resolution = resolution;
    init();
}

void Raytracer::init() {
    this->img = std::make_shared<Image>(resolution, resolution);
    this->scn.cam.setResolution(resolution);
}

void Raytracer::render() {
	for (int r = 0; r < resolution; r++) { // iterate through pixels
		for (int c = 0; c < resolution; c++) {
			Ray ray = scn.cam.getRay(r, c);
			glm::vec3 fragColor = 255.0f * computeColor(ray);
			float red = std::clamp(fragColor[0], 0.0f, 255.0f);
			float green = std::clamp(fragColor[1], 0.0f, 255.0f);
			float blue = std::clamp(fragColor[2], 0.0f, 255.0f);
			img->setPixel(c, r, (unsigned char)red, (unsigned char)green, (unsigned char)blue);
		}
	}
    img->writeToFile(filename);
}

/* PRIVATE */
Collision* Raytracer::shootRay(Ray& ray) {
    Hit* closestHit = nullptr;
    Shape* closestObj = nullptr;
    for (Shape* obj : scn.shapes) {
        Hit* hit = obj->collider(ray);
        if (hit) {
            if (!closestHit || hit->t < closestHit->t) {
                closestHit = hit;
                closestObj = obj;
            }
        }
    }
    if (!closestHit) return nullptr;
    return new Collision(closestHit, closestObj);
}

glm::vec3 Raytracer::computeColor(Ray& ray) {
    Collision* col = shootRay(ray);
    if (col) { // if ray intersects
        glm::vec3 fragPos = col->hit->pos;
        glm::vec3 fragNor = col->hit->nor;
        std::vector<Light*> activeLights;
        for (Light* l : scn.lights) { // determine visible lights from hit
            glm::vec3 l_vec = l->position - fragPos;
            glm::vec3 offset = 0.005f * fragNor;
            Ray sray (fragPos + offset, normalize(l_vec));
            Collision* scol = shootRay(sray);
            if (!scol || length(scol->hit->pos - fragPos) > length(l_vec)) { // if not occluded
                activeLights.push_back(l);
            }
        }
        return col->obj->mat->computeFrag(ray.v, fragPos, fragNor, activeLights);
    }
    else {
        return scn.bkgColor;
    }
}