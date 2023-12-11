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

#include <optix_device.h>

#include "LaunchParams.h"

using namespace osc;

namespace osc {
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  // for this simple example, we have a single ray type
  enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };
  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  {
      // TODO: NO HITS
    /*const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();*/

    // compute normal:
   /* const int   primID = optixGetPrimitiveIndex();
    const vec3i index  = sbtData.index[primID];
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    const vec3f Ng     = normalize(cross(B-A,C-A));

    const vec3f rayDir = optixGetWorldRayDirection();
    const float cosDN  = 0.2f + .8f*fabsf(dot(rayDir,Ng));
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = cosDN * sbtData.color;*/

    bool& prd = *(bool*)getPRD<bool>();
    //prd = (optixGetRayTmax() < optixLaunchParams.hemisphere.radius); // is occluded or not within radius
    //printf("HIT!\n");
    prd = true;
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }


  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    bool &prd = *(bool*)getPRD<bool>();
    prd = false; // set to not occluded
  }

  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__occlusion()
  {
    const int ix = optixGetLaunchIndex().x; // index of the point
    const int iy = optixGetLaunchIndex().y; // index of the ray

    auto &hemisphere = optixLaunchParams.hemisphere;
    auto &origins = optixLaunchParams.origins;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    bool occlusionPRD = false;

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&occlusionPRD, u0, u1);
   
    // transform ray direction
    vec3f sampledRay = ((vec3f*) hemisphere.kernel.d_ptr)[iy];
    vec3f normal = ((vec3f*) origins.normals.d_ptr)[ix];
    vec3f binormal = normalize(cross(normal, vec3f(0.0f, 0.0f, 1.0f)));
    vec3f tangent = normalize(cross(binormal, normal));
    vec3f rayDir = normalize(sampledRay.x * binormal + sampledRay.y * tangent + sampledRay.z * normal);
    vec3f origin = ((vec3f*) origins.positions.d_ptr)[ix];

    optixTrace(optixLaunchParams.traversable,
               origin,
               rayDir,
               0.001f, // tmin, epsilon for self intersections
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SURFACE_RAY_TYPE,             // missSBTIndex 
               u0, u1);

    //printf("sample point: %d and ray: %d was occluded %d\n", ix, iy, occlusionPRD);
    // and write to frame buffer ...
    optixLaunchParams.result.occlusionBuffer[optixLaunchParams.hemisphere.samples * ix + iy] = (uint32_t)occlusionPRD; // number of occlusions per vertex
  }
  
} // ::osc