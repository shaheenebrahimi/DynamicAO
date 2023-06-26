#include "Scene.h"
#include "Ellipsoid.h"
#include "Plane.h"
#include "BlinnPhong.h"
#include "Reflective.h"
#include "Triangle.h"
#include "Mesh.h"

#include <iostream>
#include <random>

Scene::Scene() { this->background = glm::vec3(0.0f, 0.0f, 0.0f); }


Collision* Scene::shootRay(Ray& ray) {
    Hit* closestHit = nullptr;
    Shape* closestObj = nullptr;
    for (Shape* obj : shapes) {
        Hit* hit = obj->collider(ray);
        if (hit) {
            if (closestHit == nullptr || hit->t < closestHit->t) {
                closestHit = hit;
                closestObj = obj;
            }
        }
    }
    if (closestHit) return new Collision(closestHit, closestObj);
    else return nullptr;
}

float Scene::computePointAmbientOcclusion(glm::vec3 pos, glm::vec3 nor, std::vector<glm::vec3>& kernel, std::vector<glm::vec3>& noise, float radius) {
    glm::vec3 noiseSample = noise[rand() % noise.size()];
        // compute occlusion factor
        int occlusionCount = 0;
        // std::cout << col->hit->nor.x << " " << col->hit->nor.y << " " << col->hit->nor.z << std::endl;
        for (int i = 0; i < kernel.size(); ++i) {
            // get normal at point for sphere
            glm::vec3 normal = normalize(nor);
            glm::vec3 tangent = normalize(noiseSample - normal * dot(noiseSample, normal));
            glm::vec3 bitangent = cross(normal, tangent);
            glm::mat3 TBN = glm::mat3(tangent, bitangent, normal);
            glm::vec3 sampleDir = normalize(TBN * kernel[i]);
            // if (dot(sampleDir,normal) < 0) std::cout << "issue" << std::endl;
            Ray oray (pos, sampleDir);
            Collision* ocol = shootRay(oray);
            if (ocol) {
                if (length(ocol->hit->pos - pos) <= radius) {
                    occlusionCount++;
                }
            }
        }
        return 1.0f - (occlusionCount / (float) kernel.size());
}

float Scene::computeRayAmbientOcclusion(Ray& ray, std::vector<glm::vec3>& kernel, std::vector<glm::vec3>& noise, float radius) {
    Collision* col = shootRay(ray);
    if (col) { // if ray intersects
        return computePointAmbientOcclusion(col->hit->pos, col->hit->nor, kernel, noise, radius);
    }
    return 1.0f; // white -- no occlusion
}

glm::vec3 Scene::computeColor(Ray& ray, int depth) {
    Collision* col = shootRay(ray);
    if (col) { // if ray intersects
        glm::vec3 fragPos = col->hit->pos;
        glm::vec3 fragNor = normalize(col->hit->nor);
        if (col->obj->mat->type == Material::BLINN_PHONG) {
            std::vector<Light*> activeLights;
            for (Light* l : lights) {
                glm::vec3 l_vec = l->position - fragPos;
                Ray sray (fragPos, normalize(l_vec));
                Collision* scol = shootRay(sray);
                if (!scol || length(scol->hit->pos - fragPos) > length(l_vec)) { // if not occluded
                    activeLights.push_back(l);
                }
            }
            return col->obj->mat->computeFrag(ray.v, fragPos, fragNor, activeLights);
        }
        else if (col->obj->mat->type == Material::REFLECTIVE) {
            if (depth == Scene::MAX_BOUNCES) {
                return background;
            }
            glm::vec3 reflection = glm::reflect(ray.v, fragNor);
            Ray rray (fragPos, reflection);
            return computeColor(rray, depth+1);
        }
        return background;
    }
    else {
        return background;
    }
}