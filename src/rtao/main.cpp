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
  extern "C" int main(int ac, char **av)
  {
      Model* model;
    try {
      std::string path = RES_DIR + "models/sphere2.obj";
      model = loadOBJ(path);
    } catch (std::runtime_error& e) {
      std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what() << GDT_TERMINAL_DEFAULT << std::endl;
	    std::cout << "Could not load obj file" << std::endl;
	    exit(1);
    }

    // TODO: Rename to OcclusionRenderer
    // TODO: apply some random transform and repeat x amount of times
    // Create SampleRenderer
    SampleRenderer renderer (model, 1, 250); // 10 points per tri and 250 rays per point
    renderer.render(); // only need to render the frame once

    std::vector<float> occlusionValues;
    renderer.downloadBuffer(occlusionValues); // download from buffer

    for (int i = 0; i < occlusionValues.size(); ++i) {
        std::cout << occlusionValues[i] << std::endl;
    }


    return 0;
  }
  
} // ::osc
