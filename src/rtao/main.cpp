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

#include "SampleRenderer.h"
#include <GL/gl.h>
#include <string>
#include <fstream>
#include <sstream>

/*! \namespace osc - Optix Siggraph Course */
/* Generates ray traced ambient occlusion training samples for neural network */

const std::string RES_DIR =
    #ifdef _WIN32
    // on windows, visual studio creates _two_ levels of build dir
    "../../../resources/"
    #else
    // on linux, common practice is to have ONE level of build dir
    "../../resources/"
    #endif
;

namespace osc {
    std::vector<float> parseHeader(const std::string & filename) {
        //// get header
        //std::ifstream in;
        //std::string line;

        //in.open(filename);
        //if (!in.good()) {
        //    std::cout << "Cannot read " << filename << std::endl;
        //}

        //// skip instruction line
        //char comment;
        //getline(in, line);

        //// theta count line
        //int dimensions;
        //getline(in, line);
        //std::stringstream ssd(line);
        //ssd >> comment;
        //ssd >> dimensions;

        //// theta value line
        //std::vector<float> thetas (dimensions);
        //getline(in, line);

        //std::stringstream sst(line);
        //sst >> comment;
        //for (int i = 0; i < dimensions; ++i) {
        //    sst >> thetas[i];
        //}
        //return thetas;
        // get header
        std::ifstream in;
        std::string line;

        in.open(filename);
        if (!in.good()) {
            std::cout << "Cannot read " << filename << std::endl;
        }

        // skip instruction line
        char comment;
        getline(in, line);

        // theta count line
        int bones;
        getline(in, line);
        std::stringstream ssd(line);
        ssd >> comment;
        ssd >> bones;

        // theta value line
        std::vector<float> orientations(4*bones);
        getline(in, line);

        std::stringstream sst(line);
        sst >> comment; // char
        for (int i = 0; i < bones; ++i) {
            sst >> orientations[4 * i];
            sst >> orientations[4 * i + 1];
            sst >> orientations[4 * i + 2];
            sst >> orientations[4 * i + 3];
        }
        return orientations;
}

  extern "C" int main(int ac, char **av)
  {
      const std::string name = "arm";

      // Read in base occlusion values
      const std::string baseFilename = RES_DIR + "occlusion/base.txt";
      std::ifstream in;
      in.open(baseFilename);
      if (!in.good()) {
          std::cout << "Couldn't read base occlusion file" << std::endl;
          return -1;
      }

      std::string line;
      std::stringstream ss;
      std::getline(in, line);

      // Get meta data
      int bones; int verts;
      ss = std::stringstream(line);
      ss >> bones; ss >> verts;

      getline(in, line);
      ss = std::stringstream(line);
      std::vector<float> baseOcclusion(verts);
      for (int vert = 0; vert < verts; ++vert) {
          ss >> baseOcclusion[vert];
      }
      in.close();
      
      // Export to file
      const std::string outputFilename = RES_DIR + "occlusion/data" + name + ".txt";
      std::ofstream out;
      //out.open(outputFilename, std::ios_base::app); // keep appending to data
      out.open(outputFilename);

      // Create header
      out << bones << " " << verts << std::endl; // first two lines num inputs, num len data
      // Iterate through distinct random poses
      int poseCount = 101; // TODO: read all of poses in folder
      for (int i = 0; i < poseCount; ++i) {
          std::string filename = RES_DIR + "data/" + name + std::to_string(i) + ".obj";
          std::vector<float> orientations = parseHeader(filename);

          // Load model
          Model* model;
          try {
              model = loadOBJ(filename);
              
          }
          catch (std::runtime_error& e) {
              std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what() << GDT_TERMINAL_DEFAULT << std::endl;
              std::cout << "Could not load obj file" << std::endl;
              exit(1);
          }

          // Create renderer and render model
          //const int sampleCount = 25000; // how many points we are sampling on the mesh
          const int rayCount = 250;
          SampleRenderer renderer(model, rayCount);
          renderer.render(); // only need to render the frame once

          // Retrieve occlusion values from GPU
          std::vector<float> occlusionValues;
          renderer.downloadBuffer(occlusionValues); // download from buffer
          //std::vector<vec2f> uvs = renderer.getUVs();

          // Write occlusion values to output file
          int inputs = orientations.size();
          int outputs = occlusionValues.size();
          //std::cout << "Input dimensionality: " << inputs << std::endl;
          for (float orient : orientations) { // theta0, theta1, ...
              out << orient << " ";
          }   
          for (int i = 0; i < outputs; ++i) { // theta0 theta1 ... occlusion value
              float deltaAO = occlusionValues[i]; // compute delta: baseOcclusion[i] - 
              out << deltaAO;
              (i == outputs - 1) ? out << "\n" : out << " ";
          }
          //for (int i = 0; i < outputs; ++i) { // u v theta0 theta1 ... occlusion value
          //    //out << uvs[i].x << " " << uvs[i].y << " ";
          //    for (float theta : thetas) {
          //        out << theta << " ";
          //    }
          //    out << occlusionValues[i] << std::endl;
          //}
      }
      // total data points = sample count * pose count
      out.close();
    return 0;
  }
  
} // ::osc
