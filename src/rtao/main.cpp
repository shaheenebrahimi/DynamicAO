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
        std::vector<float> orientations(3*bones); // euler angles
        getline(in, line);

        std::stringstream sst(line);
        sst >> comment; // char
        for (int i = 0; i < bones; ++i) {
            sst >> orientations[3 * i];
            sst >> orientations[3 * i + 1];
            sst >> orientations[3 * i + 2];
        }
        return orientations;
    }

    std::pair<int,int> getDimensions(const std::string& filename) {
        std::ifstream in;
        in.open(filename);
        if (!in.good()) {
            std::cout << "Couldn't read base occlusion file" << std::endl;
            return {};
        }
        std::string line;
        std::stringstream ss;
        std::getline(in, line);
        int bones; int verts;
        ss = std::stringstream(line);
        ss >> bones; ss >> verts;
        return { bones, verts };
    }

  extern "C" int main(int ac, char **av)
  {
      // Define consts
      const std::string name = "warrior";
      const int rayCount = 250;
      const int poseCount = 31; // TODO: read all of poses in folder
      const bool is_train = false;

      // Read in base occlusion values
      const std::string baseFilename = RES_DIR + "occlusion/base.txt";
      std::pair<int, int> dim = getDimensions(baseFilename);
      int bones = dim.first, verts = dim.second;
      
      // Open export file
      const std::string outputFilename = RES_DIR + "occlusion/_" + name + ((is_train) ? "_train_" : "_test_") + "data.txt";
      std::ofstream out;
      out.open(outputFilename);
      out << bones << " " << verts << std::endl; // first two lines num inputs, num len data

      // Initialize renderer
      SampleRenderer renderer;

      // Iterate through distinct random poses
      for (int i = 0; i < poseCount; ++i) {
          std::string filename = RES_DIR + "data/_" + name + ((is_train) ? "_train_" : "_test_") + std::to_string(i) + ".obj";

          std::vector<float> orientations = parseHeader(filename);

          // Load model
          const Model* model;
          try {
              model = loadOBJ(filename);
              
          }
          catch (std::runtime_error& e) {
              std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what() << GDT_TERMINAL_DEFAULT << std::endl;
              std::cout << "Could not load obj file" << std::endl;
              exit(1);
          }

          // Create renderer and render model
          renderer.set(model);
          renderer.render(rayCount); // only need to render the frame once

           //Retrieve occlusion values from GPU
          std::vector<float> occlusionValues;
          renderer.downloadBuffer(occlusionValues); // download from buffer

          // Write values to output file: rx0, ry0, rz0, ..., ... aoN
          int inputs = orientations.size();
          for (float orient : orientations) { // rx0, ry0, rz0, ...
              out << orient << " ";
          }
          int outputs = occlusionValues.size();
          for (int i = 0; i < outputs; ++i) { 
              float ao = occlusionValues[i];
              out << ao; (i == outputs - 1) ? out << "\n" : out << " ";
          }
          renderer.reset();
          delete model;
      }
      // total data points = sample count * pose count
      out.close();
    return 0;
  }
  
} // ::osc
