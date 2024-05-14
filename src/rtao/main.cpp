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
#include "Image.h"
#include <GL/gl.h>
#include <string>
#include <fstream>
#include <sstream>

/*! \namespace osc - Optix Siggraph Course */
/* Generates ray traced ambient occlusion training samples for neural network */

#define RENDER_TEXTURE

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
    std::string parseHeader(const std::string & filename) {
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
        std::vector<vec3f> orientations(3*bones); // euler angles
        getline(in, line);

        return line.substr(2); // ignore "# "
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
        const int poseCount = 1; // TODO: read all of poses in folder
        const bool is_train = true;

        // Read in base occlusion values
        const std::string baseFilename = RES_DIR + "occlusion/base.txt";
        std::pair<int, int> dim = getDimensions(baseFilename);
        int bones = dim.first, verts = dim.second;
      
        // Open export file
        const std::string outputFilename = RES_DIR + "occlusion/uv_" + name + ((is_train) ? "_train_" : "_test_") + "data.txt";
        std::ofstream out;
        out.open(outputFilename);
        out << bones << " " << verts << std::endl; // first two lines num inputs, num len data

        // Initialize renderer
        SampleRenderer renderer;

        // Iterate through distinct random poses
        for (int i = 0; i < poseCount; ++i) {
            std::string objFilename = RES_DIR + "data/_warrior.obj";
            std::string orientations = parseHeader(objFilename);

            // Load model
            const Model* model;
            try {
                model = loadOBJ(objFilename);
              
            }
            catch (std::runtime_error& e) {
                std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what() << GDT_TERMINAL_DEFAULT << std::endl;
                std::cout << "Could not load obj file" << std::endl;
                exit(1);
            }

            // Set target and render model
            renderer.set(model);

            // Render and output
#ifdef RENDER_TEXTURE
            int resolution = 1024;
            std::shared_ptr<Image> img = std::make_shared<Image>(resolution, resolution);
            img->setWhite();
            renderer.renderToTexture(rayCount, img, RES_DIR + "textures/" + name + ".png");
#else
            int sampleCount = 10000;
            renderer.renderToFile(rayCount, sampleCount, orientations, out); // only need to render the frame once
#endif

            renderer.reset();
            delete model;
        }
        // total data points = sample count * pose count
        out.close();
    return 0;
    }
  
} // ::osc
