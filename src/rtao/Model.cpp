// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"
//std
#include <set>

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  


  /*! find vertex with given position, normal, texcoord, and return
      its vertex ID, or, if it doesn't exit, add it to the mesh, and
      its just-created index */
  int addVertex(TriangleMesh *mesh,
                tinyobj::attrib_t &attributes,
                const tinyobj::index_t &idx,
                std::map<tinyobj::index_t,int> &knownVertices)
  {
    if (knownVertices.find(idx) != knownVertices.end())
      return knownVertices[idx];

    const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
    const vec3f *normal_array   = (const vec3f*)attributes.normals.data();
    const vec2f *texcoord_array = (const vec2f*)attributes.texcoords.data();
    
    int newID = mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
      while (mesh->normal.size() < mesh->vertex.size())
        mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
      while (mesh->texcoord.size() < mesh->vertex.size())
        mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
      mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
      mesh->normal.resize(mesh->vertex.size());
    
    return newID;
  }
  
  Model *loadOBJ(const std::string &objFile)
  {
    Model *model = new Model;

    //const std::string mtlDir
    //  = objFile.substr(0,objFile.rfind('/')+1);
    //PRINT(mtlDir);
    
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
        //= tinyobj::LoadObj(&attributes,
        //                   &shapes,
        //                   &materials,
        //                   &err,
        //                   &err,
                          // objFile.c_str(),
        //                   mtlDir.c_str(),
        //                   /* triangulate */true);
        = tinyobj::LoadObj(&attributes, &shapes, &materials, &err, objFile.c_str());
    if (!readOK) {
      throw std::runtime_error("Could not read OBJ model from "  +objFile + " : " + err);
    }

    //if (materials.empty())
    //  throw std::runtime_error("could not parse materials ...");

    std::cout << "Done loading obj file." << std::endl;
    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
      tinyobj::shape_t &shape = shapes[shapeID];

      std::set<int> materialIDs;
      for (auto faceMatID : shape.mesh.material_ids)
        materialIDs.insert(faceMatID);
      
      for (int materialID : materialIDs) {
        std::map<tinyobj::index_t,int> knownVertices;
        TriangleMesh *mesh = new TriangleMesh;
        
        for (int faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
          if (shape.mesh.material_ids[faceID] != materialID) continue;
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
          
          vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                    addVertex(mesh, attributes, idx1, knownVertices),
                    addVertex(mesh, attributes, idx2, knownVertices));
          mesh->index.push_back(idx);
 /*         mesh->diffuse = (const vec3f&)materials[materialID].diffuse;
          mesh->diffuse = gdt::randomColor(materialID);*/
        }

        if (mesh->vertex.empty())
          delete mesh;
        else
          model->meshes.push_back(mesh);
      }
    }

    // Assert
    for (auto mesh : model->meshes) {
        if (mesh->texcoord.empty())
            throw std::runtime_error("no texture coordinates.");
        if (mesh->vertex.size() != mesh->normal.size() || mesh->vertex.size() != mesh->texcoord.size())
            throw std::runtime_error("vertex attribute count not equal.");
    }

    // of course, you should be using tbb::parallel_for for stuff
    // like this:
    for (auto mesh : model->meshes)
      for (auto vtx : mesh->vertex)
        model->bounds.extend(vtx);

    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
  }
}

//namespace osc {
//	Mesh *loadOBJ(const std::string& meshName) {
//		tinyobj::attrib_t attrib;
//		std::vector<tinyobj::shape_t> shapes;
//		std::vector<tinyobj::material_t> materials;
//		std::string errStr;
//		bool rc = tinyobj::LoadObj(&attrib, &shapes, &materials, &errStr, meshName.c_str());
//		if (!rc) {
//			std::cerr << errStr << std::endl;
//		}
//		else {
//			// Loop over shapes
//			for (size_t s = 0; s < shapes.size(); s++) {
//				// Loop over faces (polygons)
//				size_t index_offset = 0;
//				for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
//					size_t fv = shapes[s].mesh.num_face_vertices[f];
//					// Loop over vertices in the face.
//					for (size_t v = 0; v < fv; v++) {
//						// access to vertex
//						tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
//						posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
//						posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
//						posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 2]);
//						if (!attrib.normals.empty()) {
//							norBuf.push_back(attrib.normals[3 * idx.normal_index + 0]);
//							norBuf.push_back(attrib.normals[3 * idx.normal_index + 1]);
//							norBuf.push_back(attrib.normals[3 * idx.normal_index + 2]);
//						}
//						if (!attrib.texcoords.empty()) {
//							texBuf.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
//							texBuf.push_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
//						}
//						else {
//							cerr << "Error: model does not have texture coords" << endl;
//						}
//					}
//					index_offset += fv;
//				}
//			}
//			bufToTriangles();
//		}
//	}
//}
