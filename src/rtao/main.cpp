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
        int dimensions;
        getline(in, line);
        std::stringstream ssd(line);
        ssd >> comment;
        ssd >> dimensions;

        // theta value line
        std::vector<float> thetas (dimensions);
        getline(in, line);

        std::stringstream sst(line);
        sst >> comment;
        for (int i = 0; i < dimensions; ++i) {
            sst >> thetas[i];
        }
        return thetas;
}

  extern "C" int main(int ac, char **av)
  {

      // Export to file
      std::ofstream out;
      //out.open(RES_DIR + "occlusion/data.txt", std::ios_base::app); // keep appending to data
      out.open(RES_DIR + "occlusion/dataTest.txt");

      // Render data points for distinct poses
      int poseCount = 1; // TODO: read all of poses in folder
      for (int i = 0; i < poseCount; ++i) {
          std::string filename = RES_DIR + "data/arm" + std::to_string(i) + ".obj";
          std::vector<float> thetas = parseHeader(filename);

          Model* model;
          try {
              model = loadOBJ(filename);
              
          }
          catch (std::runtime_error& e) {
              std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what() << GDT_TERMINAL_DEFAULT << std::endl;
              std::cout << "Could not load obj file" << std::endl;
              exit(1);
          }

          // Create SampleRenderer
          const int triSamples = 10;
          const int rayCount = 250;
          SampleRenderer renderer(model, triSamples, rayCount);
          renderer.render(); // only need to render the frame once

          std::vector<float> occlusionValues;
          renderer.downloadBuffer(occlusionValues); // download from buffer
          std::vector<vec2f> uvs = renderer.getUVs();

          int inputs = thetas.size() + 2;
          int outputs = occlusionValues.size();
          std::cout << "Input dimensionality: " << inputs << std::endl;
          //out << inputs << " " << outputs << std::endl; // first two lines num inputs, num len data
          for (int i = 0; i < outputs; ++i) { // u v theta0 theta1 ... occlusion value
              out << uvs[i].x << " " << uvs[i].y << " ";
              for (float theta : thetas) {
                  out << theta << " ";
              }
              out << occlusionValues[i] << std::endl;
          }
      }
      out.close();
    return 0;
  }
  
} // ::osc
