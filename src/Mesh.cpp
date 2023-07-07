#include "Mesh.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>

using namespace std;

Mesh::Mesh(const string& meshName, glm::vec3 position, glm::vec4 rotation, glm::vec3 scale, Material* mat) {
    this->position = position;
	this->rotation = rotation;
	this->scale = scale;
	this->mat = mat;
	this->transform = getTransformationMatrix();
    loadMesh(meshName); // load obj and populate the triangles
}

Hit* Mesh::collider(Ray& ray) {
	static constexpr bool should_permute = true;
	static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();
    static constexpr size_t stack_size = 64;
    static constexpr bool use_robust_traversal = false;

    auto prim_id = invalid_id;
    Scalar u, v;
	Ray3 r = Ray3 {
        Vec3(ray.p.x, ray.p.y,ray.p.z),   // Ray origin
        Vec3(ray.v.x, ray.v.y,ray.v.z),   // Ray direction
        0.0f,    						  // Minimum intersection distance
        100.0f   						  // Maximum intersection distance
    };

    // Traverse the BVH and get the u, v coordinates of the closest intersection.
    bvh::v2::SmallStack<BVH::Index, stack_size> stack;
    bvh.intersect<false, use_robust_traversal>(r, this->bvh.get_root().index, stack,
        [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                size_t j = should_permute ? i : this->bvh.prim_ids[i];
                if (auto hit = precomputed_tris[j].intersect(r)) {
                    prim_id = i;
                    std::tie(u, v) = *hit;
                }
            }
            return prim_id != invalid_id;
        });

    if (prim_id != invalid_id) {
		Triangle* tri = triangles[bvh.prim_ids[prim_id]];
		return tri->collider(ray);
		// // auto ptri = precomputed_tris[prim_id];
		// Scalar w = 1.0f - u - v;
		// glm::vec3 x = w * tri->vert0 + u * tri->vert1 + v * tri->vert2;
		// glm::vec3 n = normalize(w * tri->nor0 + u * tri->nor1 + v * tri->nor2);
        // return new Hit(x, n, glm::vec2(0), r.tmax);
    } else {
        return nullptr;
    }

    // Hit* closestHit = nullptr;

	// glm::vec3 p_prime = inverse(transform) * glm::vec4(ray.p, 1.0f);
    // glm::vec3 v_prime = normalize(inverse(transform) * glm::vec4(ray.v, 0.0f));
	// Ray pray (p_prime, v_prime);

	// if (box->collider(pray)) { // determine if intersects bounding shape
    //     for (Triangle* tri : triangles) {
    //         Hit* hit_prime = tri->collider(pray);
	// 		if (hit_prime) {
	// 			glm::vec3 x = transform * glm::vec4(hit_prime->pos, 1.0f);
	// 			glm::vec3 n = normalize(inverse(transpose(transform)) * glm::vec4(hit_prime->nor, 0.0f));
	// 			float t = length(x - ray.p);
	// 			if (dot(ray.v, x - ray.p) < 0) t *= -1;
	// 			if (t > 0) {
	// 				Hit* hit = new Hit(x, n, hit_prime->tex, t);
	// 				if (closestHit == nullptr || t < closestHit->t) {
	// 					closestHit = hit;
	// 				}
	// 			}
    //         }
    //     }
	// 	return closestHit;
	// }
    // return nullptr;
}

void Mesh::loadMesh(const string& meshName) {
    vector<float> posBuf;
    vector<float> norBuf;
    vector<float> texBuf;

	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;
	string errStr;
	bool rc = tinyobj::LoadObj(&attrib, &shapes, &materials, &errStr, meshName.c_str());
	if(!rc) {
		cerr << errStr << endl;
	} else {
		// Loop over shapes
		for(size_t s = 0; s < shapes.size(); s++) {
			// Loop over faces (polygons)
			size_t index_offset = 0;
			for(size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
				size_t fv = shapes[s].mesh.num_face_vertices[f];
				// Loop over vertices in the face.
				for(size_t v = 0; v < fv; v++) {
					// access to vertex
					tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+0]);
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+1]);
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+2]);
					if(!attrib.normals.empty()) {
						norBuf.push_back(attrib.normals[3*idx.normal_index+0]);
						norBuf.push_back(attrib.normals[3*idx.normal_index+1]);
						norBuf.push_back(attrib.normals[3*idx.normal_index+2]);
					}
					if(!attrib.texcoords.empty()) {
						texBuf.push_back(attrib.texcoords[2*idx.texcoord_index+0]);
						texBuf.push_back(attrib.texcoords[2*idx.texcoord_index+1]);
					}
					else {
						cerr << "Error: model does not have texture coords" << endl;
						exit(1);
					}
				}
				index_offset += fv;
			}
		}
        bufToTriangles(posBuf, norBuf, texBuf);
		computeBounds(posBuf);
		initializeBVH();
		this->box = new AABB(this->minBound, this->maxBound);
	}
}

void Mesh::bufToTriangles(vector<float>& posBuf, vector<float>& norBuf, vector<float>& texBuf) {
    for (int i = 0; i < posBuf.size()/9; i++) {
        glm::vec3 vert0 (posBuf[9*i], posBuf[9*i+1], posBuf[9*i+2]);
        glm::vec3 vert1 (posBuf[9*i+3], posBuf[9*i+4], posBuf[9*i+5]);
        glm::vec3 vert2 (posBuf[9*i+6], posBuf[9*i+7], posBuf[9*i+8]);
        glm::vec3 nor0 (norBuf[9*i], norBuf[9*i+1], norBuf[9*i+2]);
        glm::vec3 nor1 (norBuf[9*i+3], norBuf[9*i+4], norBuf[9*i+5]);
        glm::vec3 nor2 (norBuf[9*i+6], norBuf[9*i+7], norBuf[9*i+8]);
		glm::vec2 tex0 (texBuf[6*i], texBuf[6*i+1]);
        glm::vec2 tex1 (texBuf[6*i+2], texBuf[6*i+3]);
        glm::vec2 tex2 (texBuf[6*i+4], texBuf[6*i+5]);
        Triangle* tri = new Triangle(vert0, vert1, vert2, nor0, nor1, nor2, tex0, tex1, tex2, this->mat);
        triangles.push_back(tri);
    }
}

void Mesh::computeBounds(vector<float>& posBuf) {
	glm::vec3 minBound(posBuf[0], posBuf[1], posBuf[2]);
	glm::vec3 maxBound(posBuf[0], posBuf[1], posBuf[2]);
	for(int i = 0; i < (int)posBuf.size(); i += 3) {
		glm::vec3 v(posBuf[i], posBuf[i+1], posBuf[i+2]);
		minBound.x = min(minBound.x, v.x);
		minBound.y = min(minBound.y, v.y);
		minBound.z = min(minBound.z, v.z);
		maxBound.x = max(maxBound.x, v.x);
		maxBound.y = max(maxBound.y, v.y);
		maxBound.z = max(maxBound.z, v.z);
	}
	this->minBound = minBound;
	this->maxBound = maxBound;
}

void Mesh::initializeBVH() {
    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    // Get triangle centers and bounding boxes (required for BVH builder)
    std::vector<BBox> bboxes(triangles.size());
    std::vector<Vec3> centers(triangles.size());
    executor.for_each(0, triangles.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
			Tri t = Tri(
				Vec3(triangles[i]->vert0.x, triangles[i]->vert0.y, triangles[i]->vert0.z),
				Vec3(triangles[i]->vert1.x, triangles[i]->vert1.y, triangles[i]->vert1.z),
				Vec3(triangles[i]->vert2.x, triangles[i]->vert2.y, triangles[i]->vert2.z));
            bboxes[i]  = t.get_bbox();
            centers[i] = t.get_center();
        }
    });

    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
    this->bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);

    // Permuting the primitive data allows to remove indirections during traversal, which makes it faster.
    static constexpr bool should_permute = true;

    // This precomputes some data to speed up traversal further.
    this->precomputed_tris = std::vector<PrecomputedTri>(triangles.size());
    executor.for_each(0, triangles.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            auto j = should_permute ? this->bvh.prim_ids[i] : i;
            precomputed_tris[i] = Tri(
				Vec3(triangles[j]->vert0.x, triangles[j]->vert0.y, triangles[j]->vert0.z),
				Vec3(triangles[j]->vert1.x, triangles[j]->vert1.y, triangles[j]->vert1.z),
				Vec3(triangles[j]->vert2.x, triangles[j]->vert2.y, triangles[j]->vert2.z));;
        }
    });
}
