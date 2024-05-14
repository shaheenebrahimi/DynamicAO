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
#include "gdt/math/vec.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <random>
#include <chrono>
#include <cmath>
#include <unordered_set>
#include <fstream>


/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  extern "C" char embedded_ptx_code[];

  /*! SBT record for a raygen program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a miss program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    //TriangleMeshSBTData data;
    void* data;
  };


  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer::SampleRenderer()
    : model(nullptr), sampleCount(1000), rayCount(256) // TODO: Fix sample count
  {
      initOptix();

      //std::cout << "#osc: creating optix context ..." << std::endl;
      createContext();

      //std::cout << "#osc: setting up module ..." << std::endl;
      createModule();

      //std::cout << "#osc: creating programs ..." << std::endl;
      createRaygenPrograms();
      createMissPrograms();
      createHitgroupPrograms();
      createPipeline();

      std::cout << GDT_TERMINAL_GREEN;
      std::cout << "#osc: Optix 7 fully set up" << std::endl;
      std::cout << GDT_TERMINAL_DEFAULT;
  }

  void SampleRenderer::set(const Model* model) {
      this->model = model;

      launchParams.traversable = buildAccel();

      buildSBT();

      launchParamsBuffer.alloc(sizeof(launchParams));
  }

  void SampleRenderer::reset() {
      inputs.clear();
      occlusionBuffer.free();
      samplePoints.free();
      for (int i = 0; i < vertexBuffer.size(); ++i) {
          vertexBuffer[i].free();
      }
      for (int i = 0; i < indexBuffer.size(); ++i) {
          indexBuffer[i].free();
      }
      raygenRecordsBuffer.free();
      missRecordsBuffer.free();
      hitgroupRecordsBuffer.free();
      asBuffer.free();
      launchParams.origins.positions.free();
      launchParams.origins.normals.free();
      launchParams.hemisphere.kernel.free();
      launchParams.hemisphere.noise.free();
      launchParamsBuffer.free();
  }

  void SampleRenderer::getVertexSamples() {
      std::vector<vec3f> pos;
      std::vector<vec3f> nor;

      // Sample points
      for (auto mesh : this->model->meshes) { // iter through meshes
          pos.insert(pos.end(), mesh->vertex.begin(), mesh->vertex.end());
          nor.insert(nor.end(), mesh->normal.begin(), mesh->normal.end());
      }

      // send data to GPU for raytracing
      sendSamplesToGPU(pos, nor);
  }

  void SampleRenderer::getRandomSamples(const int totalSamples) {
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0]
      std::default_random_engine generator(seed);
      std::vector<vec3f> pos;
      std::vector<vec3f> nor;
      std::vector<float> areas;
      float areaSum = 0.0;

      // Compute areas
      for (auto mesh : this->model->meshes) {
          for (vec3i i : mesh->index) { // iter through triangles
              // get position of vertices of triangle
              vec3f posA = mesh->vertex[i.x];
              vec3f posB = mesh->vertex[i.y];
              vec3f posC = mesh->vertex[i.z];
              // compute rays
              vec3f AB = posB - posA;
              vec3f AC = posC - posA;
              // compute area
              float area = 0.5 * gdt::length(gdt::cross(AB, AC)); // a = 1/2 * || AB x AC ||
              areas.push_back(area);
              areaSum += area; // running sum
          }
      }

      // Sample points
      for (auto mesh : this->model->meshes) { // iter through meshes
          int triCount = 0; // counter to access area of corresponding triangle
          for (vec3i i : mesh->index) { // iter through triangles
              // get texcoord of vertices of triangle
              vec2f texA = mesh->texcoord[i.x];
              vec2f texB = mesh->texcoord[i.y];
              vec2f texC = mesh->texcoord[i.z];
              // get position of vertices of triangle
              vec3f posA = mesh->vertex[i.x];
              vec3f posB = mesh->vertex[i.y];
              vec3f posC = mesh->vertex[i.z];
              // get position of vertices of triangle
              vec3f norA = mesh->normal[i.x];
              vec3f norB = mesh->normal[i.y];
              vec3f norC = mesh->normal[i.z];


              // randomly sample in triangle
              const float alpha = areas[triCount] / areaSum; // compute weighted sample count ratio
              const int triSampleGoal = ceil(alpha * totalSamples); // how many to sample within triangle

              int sampled = 0;
              while (sampled < triSampleGoal) {
                  float r1 = sqrt(randomFloats(generator));
                  float r2 = randomFloats(generator);

                  // barycentric
                  float a = 1 - r1;
                  float b = (1 - r2) * r1;
                  float c = r1 * r2;

                  // assert valid 
                  if (!(a >= 0 && a <= 1 && b >= 0 && b <= 1 && c >= 0 && c <= 1)) continue;

                  // sampled vertex
                  float px = posA.x * a + posB.x * b + posC.x * c;
                  float py = posA.y * a + posB.y * b + posC.y * c;
                  float pz = posA.z * a + posB.z * b + posC.z * c;
                  float nx = norA.x * a + norB.x * b + norC.x * c;
                  float ny = norA.y * a + norB.y * b + norC.y * c;
                  float nz = norA.z * a + norB.z * b + norC.z * c;
                  float u = texA.x * a + texB.x * b + texC.x * c;
                  float v = texA.y * a + texB.y * b + texC.y * c;

                  // store data
                  pos.push_back(vec3f(px, py, pz));
                  nor.push_back(normalize(vec3f(nx, ny, nz)));

                  // save input
                  this->inputs.push_back(vec2f(u, v));

                  // increment number of sample
                  sampled++;
              }
              triCount++;
          }
      }

      // send data to GPU for raytracing
      sendSamplesToGPU(pos, nor);
  }

  float dot(const vec2f & a, const vec2f & b) {
      return a.x * b.x + a.y * b.y;
  }

  void SampleRenderer::getTextureSamples(int resolution) {
      const float epsilon = 0.0001;
      std::vector<vec3f> pos;
      std::vector<vec3f> nor;
      std::vector<vec2f> tex;
      std::unordered_set<std::string> cache; // no duplicate coords
      float texelStep = 1.0 / (float)resolution;
      for (auto mesh : this->model->meshes) {
          for (vec3i i : mesh->index) { // iter through triangles

              // get texcoord of vertices of triangle
              vec2f texA = mesh->texcoord[i.x];
              vec2f texB = mesh->texcoord[i.y];
              vec2f texC = mesh->texcoord[i.z];

              // get position of vertices of triangle
              vec3f posA = mesh->vertex[i.x];
              vec3f posB = mesh->vertex[i.y];
              vec3f posC = mesh->vertex[i.z];

              // get position of vertices of triangle
              vec3f norA = mesh->normal[i.x];
              vec3f norB = mesh->normal[i.y];
              vec3f norC = mesh->normal[i.z];

              // get bounding box
              vec2f minBound = vec2f(min(min(texA.x, texB.x), texC.x), min(min(texA.y, texB.y), texC.y));
              vec2f maxBound = vec2f(max(max(texA.x, texB.x), texC.x), max(max(texA.y, texB.y), texC.y));

              // define constants
              vec2f v0 = texB - texA, v1 = texC - texA;
              float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
 
              // iter through texels in bounding box
              for (float y = minBound.y; y <= maxBound.y; y += texelStep) {
                  for (float x = minBound.x; x <= maxBound.x; x += texelStep) {
                      vec2f texCoord = vec2f(x, y);

                      // get barycentric - Cramer's rule
                      vec2f v2 = vec2f(x, y) - texA;
                      float d20 = dot(v2, v0), d21 = dot(v2, v1);
                      float denom = d00 * d11 - d01 * d01;
                      float b = (d11 * d20 - d01 * d21) / denom;
                      float c = (d00 * d21 - d01 * d20) / denom;
                      float a = 1.0f - b - c;

                      // assert valid
                      if (!(a >= 0 && a <= 1 && b >= 0 && b <= 1 && c >= 0 && c <= 1)) continue;
                      float u = texA.u * a + texB.u * b + texC.u * c;
                      float v = texA.v * a + texB.v * b + texC.v * c;;
                      assert(abs(texCoord.x - u) < epsilon && abs(texCoord.y - v) < epsilon); // should be about same value

                      // sampled vertex
                      float px = posA.x * a + posB.x * b + posC.x * c;
                      float py = posA.y * a + posB.y * b + posC.y * c;
                      float pz = posA.z * a + posB.z * b + posC.z * c;
                      float nx = norA.x * a + norB.x * b + norC.x * c;
                      float ny = norA.y * a + norB.y * b + norC.y * c;
                      float nz = norA.z * a + norB.z * b + norC.z * c;

                      // add to cache
                      std::string texStr = std::to_string(x) + " " + std::to_string(y);
                      if (cache.find(texStr) != cache.end())
                          continue;
                      else 
                          cache.insert(texStr);

                      // store data
                      pos.push_back(vec3f(px, py, pz));
                      nor.push_back(normalize(vec3f(nx, ny, nz)));

                      // save input
                      this->inputs.push_back(vec2f(u,v)); // use the reconstruct for more accurate
                  }
              }
                  
          }
      }

      // send data to GPU for raytracing
      sendSamplesToGPU(pos, nor);
  }

  void SampleRenderer::sendSamplesToGPU(const std::vector<vec3f>& pos, const std::vector<vec3f>& nor) {
      launchParams.origins.positions.alloc_and_upload(pos);
      launchParams.origins.normals.alloc_and_upload(nor);
      launchParams.origins.samples = pos.size(); // number of faces * sample per tri count

      occlusionBuffer.resize(launchParams.origins.samples * this->rayCount * sizeof(int));
      launchParams.result.occlusionBuffer = (int*)occlusionBuffer.d_ptr;
  }



  OptixTraversableHandle SampleRenderer::buildAccel()
  {
    //PING;
    
    vertexBuffer.resize(model->meshes.size());
    indexBuffer.resize(model->meshes.size());
    
    OptixTraversableHandle asHandle { 0 };
    
    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(model->meshes.size());
    std::vector<CUdeviceptr> d_vertices(model->meshes.size());
    std::vector<CUdeviceptr> d_indices(model->meshes.size());
    std::vector<uint32_t> triangleInputFlags(model->meshes.size());

    for (int meshID=0; meshID < model->meshes.size(); meshID++) {
      // upload the model to the device: the builder
      TriangleMesh &mesh = *model->meshes[meshID];
      vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
      indexBuffer[meshID].alloc_and_upload(mesh.index);

      triangleInput[meshID] = {};
      triangleInput[meshID].type
        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

      // create local variables, because we need a *pointer* to the
      // device pointers
      d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
      d_indices[meshID]  = indexBuffer[meshID].d_pointer();
      
      triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
      triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
      triangleInput[meshID].triangleArray.numVertices         = (int)mesh.vertex.size();
      triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];
    
      triangleInput[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      triangleInput[meshID].triangleArray.indexStrideInBytes  = sizeof(vec3i);
      triangleInput[meshID].triangleArray.numIndexTriplets    = (int)mesh.index.size();
      triangleInput[meshID].triangleArray.indexBuffer         = d_indices[meshID];
    
      triangleInputFlags[meshID] = 0 ;
    
      // in this example we have one SBT entry, and no per-primitive
      // materials:
      triangleInput[meshID].triangleArray.flags               = &triangleInputFlags[meshID];
      triangleInput[meshID].triangleArray.numSbtRecords               = 1;
      triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
      triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
      triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
      | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      ;
    accelOptions.motionOptions.numKeys  = 1;
    accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 triangleInput.data(),
                 (int)model->meshes.size(),  // num_build_inputs
                 &blasBufferSizes
                 ));
    
    // ==================================================================
    // prepare compaction
    // ==================================================================
    
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    
    OptixAccelEmitDesc emitDesc;
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();
    
    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
    
    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* stream */0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)model->meshes.size(),
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,
                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,
                                &asHandle,
                                &emitDesc,1
                                ));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize,1);
    
    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    
    return asHandle;
  }
  

  /*! helper function that initializes optix and checks for errors */
  void SampleRenderer::initOptix()
  {
    std::cout << "#osc: initializing optix..." << std::endl;
      
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }


  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }


  /*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
  void SampleRenderer::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }


  /*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
  void SampleRenderer::createModule()
  {
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
    pipelineLinkOptions.maxTraceDepth          = 2;
      
    const std::string ptxCode = embedded_ptx_code;
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
#if OPTIX_VERSION >= 70700
    OPTIX_CHECK(optixModuleCreate(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,&sizeof_log,
                                         &module
                                         ));
#else
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,      // Log string
                                         &sizeof_log,// Log string sizse
                                         &module
                                         ));
#endif
    if (sizeof_log > 1) PRINT(log);
  }


  /*! does all setup for the raygen program(s) we are going to use */
  void SampleRenderer::createRaygenPrograms()
  {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;
    pgDesc.raygen.entryFunctionName = "__raygen__occlusion";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &raygenPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    

  /*! does all setup for the miss program(s) we are going to use */
  void SampleRenderer::createMissPrograms()
  {
    // we do a single ray gen program in this example:
    missPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;           
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    

  /*! does all setup for the hitgroup program(s) we are going to use */
  void SampleRenderer::createHitgroupPrograms()
  {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH            = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    

  /*! assembles the full pipeline of all programs */
  void SampleRenderer::createPipeline()
  {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));
    if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
                (/* [in] The pipeline to configure the stack size for */
                 pipeline, 
                 /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */                 
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 2*1024,
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 1));
    if (sizeof_log > 1) PRINT(log);
  }


  /*! constructs the shader binding table */
  void SampleRenderer::buildSBT()
  {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i=0;i<raygenPGs.size();i++) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i=0;i<missPGs.size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID=0;meshID<numObjects;meshID++) {
      HitgroupRecord rec;
      // all meshes use the same code, so all same hit group
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0],&rec));
      //rec.data.color  = model->meshes[meshID]->diffuse;
      //rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
      //rec.data.index  = (vec3i*)indexBuffer[meshID].d_pointer();
      rec.data = nullptr; /* for now ... */
      hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
  }

  /*! render one frame */
  void SampleRenderer::render(const int rayCount, const Mode mode, const int data)
  {
      std::cout << "#osc: rendering ... ";

      // generate rays for occlusion hemisphere
      //std::cout << "#osc: generating hemisphere ..." << std::endl;
      genHemisphere();

      //std::cout << "#osc: total samples " << launchParams.origins.samples << std::endl;
      launchParamsBuffer.upload(&launchParams, 1);


      OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
          pipeline, stream,
          /*! parameters and SBT */
          launchParamsBuffer.d_pointer(),
          launchParamsBuffer.sizeInBytes,
          &sbt,
          /*! dimensions of the launch: */
          launchParams.origins.samples, // number of points to compute occlusion
          launchParams.hemisphere.samples, // number of rays to compute per point
          1
      ));
      std::cout << "done" << std::endl;
      // sync - make sure the frame is rendered before we download and
      CUDA_SYNC_CHECK();
  }

  void SampleRenderer::renderToTexture(const int rayCount, std::shared_ptr<Image> img, const std::string& filename) {
      const int accumulations = 60;
      const int resolution = img->getHeight();
      this->rayCount = rayCount;

      // Sample data points
      getTextureSamples();

      /*switch (mode) {
      case Mode::Random: getRandomSamples(data); break;
      case Mode::Texture: getTextureSamples(data); break;
      case Mode::Vertex: getVertexSamples(); break;
      }*/

      std::vector<float> accumulator(launchParams.origins.samples, 0); // initiailize to zeros
      for (int acc = 0; acc < accumulations; ++acc) {

          // Compute occlusion for all samples
          render(rayCount, Mode::Texture, resolution);

          // Output to image
          std::vector<float> occlusionValues;
          downloadBuffer(occlusionValues); // download from buffer

          // Add to accumulator
          for (int i = 0; i < accumulator.size(); ++i) {
              accumulator[i] += occlusionValues[i];
          }
          std::cout << accumulator[0] << " " << occlusionValues[0] << std::endl;


          //assert(this->inputs.size() == occlusionValues.size());
      }

      // Get average
      for (int i = 0; i < accumulator.size(); ++i) accumulator[i] /= accumulations;

      // Write to image
      for (int index = 0; index < accumulator.size(); ++index) {
          vec2f uv = inputs[index];
          int ao = (int)(255 * (1.0 - accumulator[index]));
          img->setPixel((int)(uv.x * resolution), (int)(uv.y * resolution), ao, ao, ao);
          //std::cout << std::to_string((int)(uv.x * resolution)) << ", " << std::to_string((int)(uv.y * resolution)) << " " << std::to_string(ao) << std::endl;

      }
      img->writeToFile(filename);
  }


  void SampleRenderer::renderToFile(const int rayCount, const int sampleCount, const std::string& orientations, std::ofstream& out) {
      // Compute occlusion for all samples
      render(rayCount, Mode::Random, sampleCount);

      // Get occlusion values from GPU
      std::vector<float> occlusionValues;
      downloadBuffer(occlusionValues); // download from buffer

      // Write values to output file: u0, v0, rx0, ry0, rz0, ..., ... aoN
      std::string ln = "";
      for (int i = 0; i < this->inputs.size(); ++i) {
          //out << uvs[i].u << " " << uvs[i].v << " " << orientations << " ";
          //out << occlusionValues[i] << "\n"; // output
          ln += (std::to_string(this->inputs[i].u) + " " + std::to_string(this->inputs[i].v) + " " + orientations + " " + std::to_string(occlusionValues[i]) + "\n");
      }
      out << ln;

      // Write values to output file: rx0, ry0, rz0, ..., ... aoN
      //out << orientations << " ";
      ////int outputs = occlusionValues.size();
      //for (int i = 0; i < outputs; ++i) {
      //    float ao = occlusionValues[i];
      //    out << ao; (i == outputs - 1) ? out << "\n" : out << " ";
      //}
  }

  /*
  
  
  def random_concentric(u, v):
    x = []
    y = []

    # Correct range [-1;1]
    u = 2 * u - 1
    v = 2 * v - 1

    for i, val in enumerate(u):
        a = u[i]
        b = v[i]

        if a > -b:
            if a > b:  # Region 1
                r = a
                phi = (np.pi / 4) * (b / a)
            else:  # Region 2
                r = b
                phi = (np.pi / 4) * (2 - (a / b))
        else:
            if a < b:  # Region 3
                r = -a
                phi = (np.pi / 4) * (4 + (b / a))
            else:  # Region 4
                r = -b
                if b != 0:
                    phi = (np.pi / 4) * (6 - (a / b))
                else:
                    phi = 0

        x.append(r * np.cos(phi))
        y.append(r * np.sin(phi))

    return x, y
  
  
  
  */

  void SampleRenderer::genHemisphere(float radius, bool seeded) {
    std::vector<vec3f> kernel(this->rayCount);
    std::vector<vec3f> noise(this->rayCount);

    unsigned seed = (seeded) ? std::chrono::system_clock::now().time_since_epoch().count() : 0;
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0]
    std::default_random_engine generator(seed);

    for (int i = 0; i < this->rayCount; ++i) {
        // sample random vectors in unit hemisphere

        vec3f dirSample(
            randomFloats(generator) * 2.0 - 1.0, // set range: [-1, 1]
            randomFloats(generator) * 2.0 - 1.0,
            randomFloats(generator) // ignore bottom half since not sphere
        );
        dirSample = normalize(dirSample);
        kernel[i] = dirSample;

        // generate noise
        vec3f noiseSample(
            randomFloats(generator) * 2.0 - 1.0,
            randomFloats(generator) * 2.0 - 1.0,
            0.0f // rotate along z
        );
        noise[i] = noiseSample;
    }
    // Save data to device
    launchParams.hemisphere.kernel.alloc_and_upload(kernel);
    launchParams.hemisphere.noise.alloc_and_upload(noise);
    launchParams.hemisphere.radius = radius;
    launchParams.hemisphere.samples = this->rayCount;
  }


  /*! download the rendered color buffer */
  void SampleRenderer::downloadBuffer(std::vector<float>& occlusions)
  {
      // device to host
      std::vector<int> occlusionTable;
      occlusionTable.resize(launchParams.origins.samples * launchParams.hemisphere.samples);
      occlusionBuffer.download(occlusionTable.data(), launchParams.origins.samples * launchParams.hemisphere.samples);

      // count number of occlusions per point
      occlusions.resize(launchParams.origins.samples);
      for (int i = 0; i < occlusionTable.size(); ++i) {
          int index = i / launchParams.hemisphere.samples;
          occlusions[index] += occlusionTable[i];
      }

      // compute occlusion factor
      for (int i = 0; i < occlusions.size(); ++i) {
          occlusions[i] /= (float)launchParams.hemisphere.samples;
      }
  }

  std::vector<vec2f> osc::SampleRenderer::getUVs()
  {
      return this->inputs;
  }
  
} // ::osc
