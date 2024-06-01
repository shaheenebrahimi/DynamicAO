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

        // skip lines
        getline(in, line);
        getline(in, line);

        // theta count line
        char comment;
        int bones;
        getline(in, line);
        std::stringstream ssd(line);
        ssd >> comment; // #
        ssd >> bones;

        // theta value line
        getline(in, line);
        replace(line.begin(), line.end(), ' ', ',');
        if (line.back() == ',') line.pop_back();

        return line.substr(2); // ignore "# " rather "#," now
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

//#define TRAIN_TEST
//#define RENDER_TEXTURE

    extern "C" int main(int ac, char **av)
    {

        // Define consts
        const std::string name = "research";
        const int accumulations = 10;
        const int rayCount = 4096; // 8192
        const int sampleCount = 10000; // samples per mesh - doesn't really matter if < tri count
        const int poseCount = 0; // TODO: read all poses
        const bool is_train = false;

        // Read in base occlusion values
        const std::string baseFilename = RES_DIR + "occlusion/" + name + "_base.txt";
        std::pair<int, int> dim = getDimensions(baseFilename);
        int bones = dim.first, verts = dim.second;
      
        // Open export file
#ifdef TRAIN_TEST
        const std::string outputFilename = RES_DIR + "occlusion/" + name + ((is_train) ? "_train_" : "_test_") + "data.csv";
#else
        const std::string outputFilename = RES_DIR + "occlusion/" + name + "_data.csv";
#endif

#ifndef RENDER_TEXTURE
        std::ofstream out;
        out.open(outputFilename, std::ios_base::app);
        //out << bones << " " << verts << std::endl; // first two lines num inputs, num len data
#endif

        const Model* base;
        base = loadOBJ(RES_DIR + "data/" + name + ".obj");

        // Initialize renderer
        SampleRenderer renderer(rayCount);

        // Iterate through distinct random poses
        for (int i = -1; i < poseCount; ++i) {
            std::string tail = (i == -1) ? "" : ("_" + std::to_string(i));
            std::string objFilename = RES_DIR + "data/" + name + tail + ".obj";
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

            // Render and output
#ifdef RENDER_TEXTURE
#ifdef TRAIN_TEST
            std::string imgPath = RES_DIR + "occlusion/" + (is_train ? "train/" : "test/") + name + tail + ".png";
#else
            std::string imgPath = RES_DIR + "textures/" + name + ".png";

#endif
            int resolution = 512;
            std::shared_ptr<Image> img = std::make_shared<Image>(resolution, resolution);
            // Set target and render model
            renderer.set(model);
            renderer.sampleData(Mode::Texture, resolution);
            renderer.renderToTexture(rayCount, img, imgPath);
#else
            
            // Set target and render model
            renderer.set(model);
            renderer.sampleData(Mode::Random, sampleCount);
            for (int f = 0; f < accumulations; ++f) {
                renderer.render();
                renderer.downloadBuffer();
            }
            std::vector<float> skinned = renderer.getAccumulation();
            for (int x = 0; x < skinned.size(); ++x) {
                skinned[x] /= accumulations;
            }

            // Set base model and render
            std::vector<float> tpose;
            if (i == -1) {
                tpose = skinned;
            }
            else {
                renderer.set(base);
                renderer.lookupUVs();
                for (int f = 0; f < accumulations; ++f) {
                    renderer.render();
                    renderer.downloadBuffer();
                }
                tpose = renderer.getAccumulation();
                for (int x = 0; x < tpose.size(); ++x) {
                    tpose[x] /= accumulations;
                }
            }
            

            // Write to file
            std::vector<vec2f> uvs = renderer.getUVs();
            
            // Write values to output file: u0, v0, rx0, ry0, rz0, ..., ... aoN
            std::string ln = "";
            for (int i = 0; i < uvs.size(); ++i) {
                if (uvs[i].x < 0) uvs[i].x = 1.0f - ((float)(abs(uvs[i].x) - (int)abs(uvs[i].x)));
                if (uvs[i].y < 0) uvs[i].y = 1.0f - ((float)(abs(uvs[i].y) - (int)abs(uvs[i].y)));
                ln += (std::to_string(uvs[i].u) + "," + std::to_string(uvs[i].v) + "," + orientations + "," + std::to_string(tpose[i]) + "," + std::to_string(skinned[i]) + "\n");
            }
            out << ln;
            // NOTE: computes occlusion not color

            std::cout << "Traced pose " << i << std::endl;
            //renderer.renderToFile(rayCount, sampleCount, orientations, out); // only need to render the frame once
#endif

            renderer.reset();
            delete model;
        }
        // total data points = sample count * pose count
        delete base;
#ifndef RENDER_TEXTURE
        out.close();
        std::cout << "Wrote to " << outputFilename << std::endl;
#endif
    return 0;
    }
  
} // ::osc
