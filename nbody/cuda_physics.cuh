/************************************************************
 *  Author(s):         Bruno Donoso-Tapia, Nicholas Tibbetts
 *  Date:
 *  Course Code:
 *  License:        Copyright 2024 Nic Tibbetts
 *  References:     _
 *  Description:    _
 ***********************************************************/


#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <sm_20_atomic_functions.h>
#include <iostream>

struct Vec2;
struct Body;

#include "Vec2.h"
#include "Body.h"

#define BLOCK_SIZE 256
#define MAX_NODES (1024 * 1024)
#define WARP_SIZE 32

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                     << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while(0)

__device__ float2 operator+(const float2& a, const float2& b);
__device__ float2 operator-(const float2& a, const float2& b);
__device__ float2 operator*(float2 a, float b);
__device__ float dot(float2 a, float2 b);
__device__ float length2(float2 v);

struct alignas(32) CUDANode
{
    struct alignas(16)
    {
        float2 pos;       // 8 bytes
        float mass;       // 4 bytes
        uint32_t subnode; // 4 bytes
    } active;

    struct alignas(16)
    {
        uint32_t next; // 4 bytes
        struct
        {
            uint32_t start; // 4 bytes
            uint32_t end;   // 4 bytes
        } range;
        struct alignas(8)
        {
            float2 center; // 8 bytes
            float size;    // 4 bytes
        } quad;
    } storage;

    __device__ bool is_leaf() const {
        return active.subnode == 0;
    }
};

struct alignas(32) CUDABody
{
    float2 pos;
    float2 vel;
    float2 acc;
    float mass;
    float radius;
};

__device__ float2 accelerationKernel(
    const CUDABody& body,
    const CUDANode* nodes,
    const CUDABody* bodies,
    size_t node_idx,
    float t_sq,
    float e_sq);

__global__ void buildKernel(
    CUDABody* bodies,
    CUDANode* nodes,
    int* nodeCount,
    float rootSize,
    int numBodies);

__global__ void forceComponent(
    CUDABody* bodies,
    const CUDANode* nodes,
    int numBodies,
    float theta,
    float epsilon);

__global__ void integrateKernel(
    CUDABody* bodies,
    int numBodies,
    float dt);

class CUDAPhysics
{

public:
 
    CUDAPhysics(size_t maxBodies, float theta, float epsilon, size_t leaf_capacity);
    ~CUDAPhysics()
    {
      
        cleanup();
    }
  
    void initialize(const std::vector<Body>& bodies);
    void updateBodies(const std::vector<Body>& bodies);
    void step(float dt, float theta, float epsilon);
    void getBodies(std::vector<Body>& bodies);

    void resize(size_t newSize);



private:

    CUDABody* d_bodies = nullptr;
    CUDANode* d_nodes = nullptr;
    int* d_nodeCount = nullptr;
    size_t maxBodies = 0;
    size_t numBodies = 0;
    float t_sq = 0.0f;
    float e_sq = 0.0f;
    size_t leaf_capacity = 0;


    void allocateMemory();
    void cleanup();
    static bool debugPrint(const char* msg) {
        std::cerr << "[CUDA DEBUG] " << msg << std::endl;
        return true;
    }



};




  

