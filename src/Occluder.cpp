#include "Occluder.h"

/* PUBLIC */
Occluder::Occluder() { 
    this->filename = "out.png";
    this->resolution = 1024;
    this->samples = 200;
    this->radius = 1.0f;
    init();
}

Occluder::Occluder(const std::string& filename, int resolution) {
    this->filename = filename;
    this->resolution = resolution;
    this->samples = 200;
    this->radius = 1.0f;
    init();
}

Occluder::~Occluder() { 
	kernel.clear();
	noise.clear();
}

void Occluder::init() {
    this->img = std::make_shared<Image>(resolution, resolution);
	this->img->setWhite();
    this->scn.cam.setResolution(resolution);
	genOcclusionHemisphere();
}

void Occluder::render() {
	// compute ao for every vertex
	for (int r = 0; r < resolution; r++) { // iterate through pixels
		for (int c = 0; c < resolution; c++) {
			Ray ray = scn.cam.getRay(r, c);
			float ao = computeRayOcclusion(ray);
			img->setPixel(c, r, 255*ao, 255*ao, 255*ao);
		}
	}
    img->writeToFile(filename);
}

void Occluder::renderTexture(std::shared_ptr<Object> target) {
	std::vector<std::vector<bool>> cache (resolution, std::vector<bool> (resolution, false));
	float texelStep = 1.0f / (float) resolution;
	for (std::shared_ptr<Triangle> tri : target->mesh->getTriangles()) {
		// Get bounding box
		Tri2D t = tri->convertTexToTri();
		BBox2D box = BBox2D(t.p0).extend(t.p1).extend(t.p2);
		// Iterate bounding box
		for (float y = box.min[1]; y <= box.max[1]; y += texelStep) {
			for (float x = box.min[0]; x <= box.max[0]; x += texelStep) {
				glm::vec2 texCoord (x, y);
				glm::vec2 texel = glm::vec2(int(texCoord.x * (resolution-1)), int(texCoord.y * (resolution-1)));
				
				// Compute barycentric
				glm::vec3 bary = tri->computeBarycentric(texCoord);
				if (bary.x >= 0 && bary.x <= 1 && bary.y >= 0 && bary.y <= 1 && bary.z >= 0 && bary.z <= 1) {
					if (cache[texel.y][texel.x]) continue;
					cache[texel.y][texel.x] = true;

					glm::vec3 pos = (bary.x * tri->pos0 + bary.y * tri->pos1 + bary.z * tri->pos2);
					glm::vec3 nor = (bary.x * tri->nor0 + bary.y * tri->nor1 + bary.z * tri->nor2);
					glm::vec3 worldPos = glm::vec3(glm::vec4(pos, 1.0f) * target->transform);
					glm::vec3 worldNor = glm::vec3(glm::vec4(nor, 0.0f) * inverse(transpose(target->transform)));
					float ao = computePointOcclusion(worldPos, worldNor);
					img->setPixel(texel.x, texel.y, 255*ao, 255*ao, 255*ao); // bottom left to top right image
				}

			}
		}
	}
    img->writeToFile(filename);
}

void Occluder::renderTextureLegacy(std::shared_ptr<Object> target) {
	for (int ty = 0; ty < resolution; ++ty) { // texels 0 0 bottom left
		for (int tx = 0; tx < resolution; ++tx) {
			glm::vec2 texel (tx, ty);
			glm::vec2 texCoord = texel / (float) resolution;
			img->setPixel(tx, ty, 255, 255, 255); // default white
			for (std::shared_ptr<Triangle> tri : target->mesh->getTriangles()) { // does this texel intersect any triangles? 
				glm::vec3 bary = tri->computeBarycentric(texCoord); // x = a, y = b, z = c
				if (bary.x >= 0 && bary.x <= 1 && bary.y >= 0 && bary.y <= 1 && bary.z >= 0 && bary.z <= 1) {
					glm::vec3 pos = (bary.x * tri->pos0 + bary.y * tri->pos1 + bary.z * tri->pos2);
					glm::vec3 nor = (bary.x * tri->nor0 + bary.y * tri->nor1 + bary.z * tri->nor2);
					glm::vec3 worldPos = glm::vec3(glm::vec4(pos, 1.0f) * target->transform);
					glm::vec3 worldNor = glm::vec3(glm::vec4(nor, 0.0f) * inverse(transpose(target->transform)));
					float ao = computePointOcclusion(worldPos, worldNor);
					img->setPixel(tx, ty, 255*ao, 255*ao, 255*ao); // bottom left to top right image
					break;
				}
			}
		}
	}
    img->writeToFile(filename);
}

/* PRIVATE */
std::optional<Hit> Occluder::shootRay(Ray& ray) {
    std::optional<Hit> closestHit = std::nullopt;
    for (std::shared_ptr<Object> obj : scn.objects) {
        auto hit = obj->mesh->collider(ray);
        if (hit) {
            if (!closestHit || hit->t < closestHit->t) { // no intersection
                closestHit = hit;
            }
        }
    }
	return closestHit;
}

void Occluder::genOcclusionHemisphere() {
	std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0]
	std::default_random_engine generator;

	for (int i = 0; i < samples; ++i) {
		// sample random vectors in unit hemisphere
		glm::vec3 pointSample(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) // ignore bottom half since not sphere
		);
		pointSample = glm::normalize(pointSample);
		// float scale = (float)i / samples; 
		// scale = lerp(0.1f, 1.0f, scale * scale);
		// pointSample *= scale;
		kernel.push_back(pointSample);

		// generate noise
		glm::vec3 noiseSample(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) * 2.0 - 1.0,
			0.0f // rotate along z
		);
		noise.push_back(noiseSample);
	}
}

float Occluder::computePointOcclusion(glm::vec3 pos, glm::vec3 nor) {
    glm::vec3 noiseSample = noise[rand() % noise.size()];
    int occlusionCount = 0; // compute occlusion factor
    // std::cout << col->hit->nor.x << " " << col->hit->nor.y << " " << col->hit->nor.z << std::endl;
    for (int i = 0; i < kernel.size(); ++i) {
        // get normal at point for sphere
        glm::vec3 normal = normalize(nor);
        glm::vec3 tangent = normalize(noiseSample - normal * dot(noiseSample, normal));
        glm::vec3 bitangent = cross(normal, tangent);
        glm::mat3 TBN = glm::mat3(tangent, bitangent, normal);
        glm::vec3 sampleDir = normalize(TBN * kernel[i]);
        glm::vec3 offset = 0.005f * normal;
        Ray oray (pos + offset, sampleDir);
        auto hit = shootRay(oray);
        if (hit) {
            if (length(hit->pos - pos) <= radius) {
                occlusionCount++;
            }
        }
    }
    return 1.0f - (occlusionCount / (float) kernel.size());
}

float Occluder::computeRayOcclusion(Ray& ray) {
    auto hit = shootRay(ray);
    if (hit) { // if ray intersects
        return computePointOcclusion(hit->pos, hit->nor);
    }
    return 1.0f; // white -- no occlusion
}
