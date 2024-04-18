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

#pragma once

// our own classes, partly shared between host and device
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Image.h"
#include "Model.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  struct Point {
    /*! Vertex pos and nor in world */
    vec3f position;
    vec3f normal;
  };

  enum class Mode {
      Vertex,
      Random,
      Texture
  };
  
  /*! a sample OptiX-7 renderer that demonstrates how to set up
      context, module, programs, pipeline, SBT, etc, and perform a
      valid launch that renders some pixel (using a simple test
      pattern, in this case */
  class SampleRenderer
  {
      // ------------------------------------------------------------------
      // publicly accessible interface
      // ------------------------------------------------------------------
  public:
      /*! constructor - performs all setup, including initializing
        optix, creates module, pipeline, programs, SBT, etc. */
      SampleRenderer();

      /*! set object to be rendered */
      void set(const Model* model);

      /*! free memory */
      void reset();

      /*! render one frame */
      void render(const int rayCount, const Mode mode = Mode::Random);

      void renderImage(const int rayCount, std::shared_ptr<Image> img);

      /*! resize frame buffer to given resolution */
      //void resize(const vec2i &newSize);

      /*! download the rendered color buffer */
      void downloadBuffer(std::vector<float>& occlusions);

      /*! get input uv values */
      std::vector<vec2f> getUVs();

      std::string getInfo() {
          return "UVs: " + std::to_string(inputs.size()) + ", Verts: " + std::to_string(launchParams.origins.samples);
      }

    /*! set camera to render with */
    //void setCamera(const Camera &camera);
  protected:
    // ------------------------------------------------------------------
    // internal helper functions
    // ------------------------------------------------------------------

    /*! helper function that initializes optix and checks for errors */
    void initOptix();
  
    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void createContext();

    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void createModule();
    
    /*! does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();
    
    /*! does all setup for the miss program(s) we are going to use */
    void createMissPrograms();
    
    /*! does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void createPipeline();

    /*! constructs the shader binding table */
    void buildSBT();

    /*! constructs occlusion hemisphere with radius */
    void genHemisphere(int radius = 1, bool seeded = false);

    /*! generate occlusion sample points */
    void getRandomSamples();

    /*! generate occlusion sample points */
    void getVertexSamples();

    void getTextureSamples(int resolution);

    /*! build an acceleration structure for the given triangle mesh */
    OptixTraversableHandle buildAccel();

  private:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /*! @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;
    /*! @} */

    CUDABuffer occlusionBuffer;
    
    /*! the model we are going to trace rays against */
    std::vector<vec2f> inputs; // texcoord inputs
    const Model *model;
    int sampleCount; // number of points sampled per triangle
    int rayCount; // number of sampled in hemisphere rays per point

    /*! buffer containing occlusion sample vertices */
    CUDABuffer samplePoints;
    
    /*! one buffer per input mesh */
    std::vector<CUDABuffer> vertexBuffer;
    /*! one buffer per input mesh */
    std::vector<CUDABuffer> indexBuffer;
    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;
  };

} // ::osc
