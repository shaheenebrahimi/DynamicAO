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
    const float t = optixGetRayTmax();
    const vec3f rayDir = optixGetWorldRayDirection();
    const float rayDist = gdt::length(t * rayDir);
    prd = (rayDist < optixLaunchParams.hemisphere.radius); // occlusion if within radius
    //if (prd) {
    //    std::printf("Hit!! t = %f, dist = %f, dir = (%f,%f,%f)\n", t, rayDist, rayDir.x, rayDir.y, rayDir.z);
    //}
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
    const int point_index = optixGetLaunchIndex().x; // index of the point
    const int ray_index = optixGetLaunchIndex().y; // index of the ray

    auto &hemisphere = optixLaunchParams.hemisphere;
    auto &origins = optixLaunchParams.origins;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    bool occlusionPRD = false;

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&occlusionPRD, u0, u1);
   
    // retrieve origin
    vec3f origin = ((vec3f*)origins.positions.d_ptr)[point_index];

    // transform ray direction
    vec3f ray = ((vec3f*) hemisphere.kernel.d_ptr)[ray_index];
    vec3f normal = normalize(((vec3f*) origins.normals.d_ptr)[point_index]); // target
    
    // axis angle
    float angle = acos(dot(normal, vec3f(0, 0, 1)));
    vec3f axis = normalize(cross(normal, vec3f(0, 0, 1)));
    float c = cos(angle);
    float s = sin(angle);
    float t = 1.0 - c;

    // Rotation Matrix (column vectors)
    vec3f R0, R1, R2;

    if (normal == vec3f(0, 0, 1)) {
        R0 = vec3f(1, 0, 0);
        R1 = vec3f(0, 1, 0);
        R2 = vec3f(0, 0, 1);
    }
    else if (normal == vec3f(0, 0, -1)) {
        R0 = vec3f(1, 0, 0);
        R1 = vec3f(0, 1, 0);
        R2 = vec3f(0, 0, -1);
    }
    else {
        R0 = vec3f(axis.x * axis.x * t + c,
            axis.x * axis.y * t - axis.z * s,
            axis.x * axis.z * t + axis.y * s);
        R1 = vec3f(
            axis.x * axis.y * t + axis.z * s,
            axis.y * axis.y * t + c,
            axis.y * axis.z * t - axis.x * s);
        R2 = vec3f(
            axis.x * axis.z * t - axis.y * s,
            axis.y * axis.z * t + axis.x * s,
            axis.z * axis.z * t + c);
    }

    vec3f rayDir = normalize(ray.x * R0 + ray.y * R1 + ray.z * R2);

    optixTrace(optixLaunchParams.traversable,
               origin,
               rayDir,
               1e-3f, // tmin, epsilon for self intersections
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
    optixLaunchParams.result.occlusionBuffer[optixLaunchParams.hemisphere.samples * point_index + ray_index] = (uint32_t)occlusionPRD; // number of occlusions per vertex
  }
  
} // ::osc
