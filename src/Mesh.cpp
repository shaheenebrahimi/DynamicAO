#include "Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>

using namespace std;

/* PUBLIC  */
Mesh::Mesh() {
    this->transform = glm::mat4(1);
}

Mesh::Mesh(const string& meshName) {
    this->transform = glm::mat4(1);
    loadMesh(meshName); // load obj and populate the triangles
	constructBVH();
}

Mesh::~Mesh() {
	triangles.clear();
	transformed.clear();
	precomputed.clear();
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
					}
				}
				index_offset += fv;
			}
		}
        bufToTriangles(posBuf, norBuf, texBuf);
	}
}

void Mesh::constructBVH() {
    bvh::v2::ThreadPool thread_pool;
    bvh::v2::ParallelExecutor executor(thread_pool);

    // Get triangle centers and bounding boxes (required for BVH builder)
	this->transformed.resize(triangles.size());
    std::vector<BBox> bboxes(triangles.size());
    std::vector<Vec> centers(triangles.size());
    executor.for_each(0, triangles.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
			this->transformed[i] = triangles[i].applyTransformation(transform);
			Tri t = this->transformed[i].convertPosToTri();
            bboxes[i]  = t.get_bbox();
            centers[i] = t.get_center();
        }
    });

    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
    this->accel = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);

    // This precomputes some data to speed up traversal further.
    this->precomputed = std::vector<PrecomputedTri>(triangles.size());
    executor.for_each(0, triangles.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            auto j = should_permute ? this->accel.prim_ids[i] : i;
            precomputed[i] = transformed[j].convertPosToTri();
        }
    });

}

std::optional<Hit> Mesh::collider(Ray& ray) {
    auto prim_id = invalid_id;
    Scalar u, v;
	BvhRay r = BvhRay {
        Vec(ray.p.x, ray.p.y, ray.p.z),   // Ray origin
        Vec(ray.v.x, ray.v.y, ray.v.z),   // Ray direction
        0.0f,    						  // Minimum intersection distance
        100.0f   						  // Maximum intersection distance
    };

    // Traverse the BVH and get the u, v coordinates of the closest intersection.
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    this->accel.intersect<false, use_robust_traversal>(r, this->accel.get_root().index, stack,
        [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                size_t j = should_permute ? i : this->accel.prim_ids[i];
                if (auto hit = precomputed[j].intersect(r)) {
                    prim_id = i;
                    std::tie(u, v) = *hit;
                }
            }
            return prim_id != invalid_id;
        });
    if (prim_id != invalid_id) {
		// size_t index = should_permute ?  this->accel.prim_ids[prim_id] : prim_id;
		Triangle tri = transformed[this->accel.prim_ids[prim_id]];
		// auto ptri = precomputed[prim_id];
		Scalar w = 1.0f - u - v;
        return Hit(r.tmax, tri.interpolatePos(w,u,v), tri.interpolateNor(w,u,v), tri.interpolateTex(w,u,v));
    } 

    return std::nullopt;
}

/* PRIVATE */

void Mesh::bufToTriangles(vector<float>& posBuf, vector<float>& norBuf, vector<float>& texBuf) {
    for (int i = 0; i < posBuf.size()/9; i++) {
        glm::vec3 pos0 (posBuf[9*i], posBuf[9*i+1], posBuf[9*i+2]);
        glm::vec3 pos1 (posBuf[9*i+3], posBuf[9*i+4], posBuf[9*i+5]);
        glm::vec3 pos2 (posBuf[9*i+6], posBuf[9*i+7], posBuf[9*i+8]);
        glm::vec3 nor0 (norBuf[9*i], norBuf[9*i+1], norBuf[9*i+2]);
        glm::vec3 nor1 (norBuf[9*i+3], norBuf[9*i+4], norBuf[9*i+5]);
        glm::vec3 nor2 (norBuf[9*i+6], norBuf[9*i+7], norBuf[9*i+8]);
		glm::vec2 tex0 (texBuf[6*i], texBuf[6*i+1]);
        glm::vec2 tex1 (texBuf[6*i+2], texBuf[6*i+3]);
        glm::vec2 tex2 (texBuf[6*i+4], texBuf[6*i+5]);
        Triangle tri (pos0, pos1, pos2, nor0, nor1, nor2, tex0, tex1, tex2);
        triangles.push_back(tri);
    }
}