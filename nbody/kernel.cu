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
#include <device_atomic_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <sm_20_atomic_functions.h>
//#include <device_functions.h>

#include "cuda_physics.cuh"
#include "Vec2.h"
#include "Body.h"

#include <string>




__device__ float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ float2 operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

__device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

__device__ float length2(float2 v)
{
    return dot(v, v);

}
    __device__ __forceinline__ float fast_inv_sqrt(float x) {
        return rsqrtf(x);
    }

__device__ __forceinline__ void syncthreads() {
    __syncthreads();
}

__device__ size_t find_quadrant(float2 pos, float2 center)
{
    return ((pos.y > center.y) << 1) | (pos.x > center.x);
}

__device__ void into_quadrant(float2& new_center, float& new_size,
    const float2& center, float size, size_t quadrant)
{
    float half_size = size * 0.5f;
    float offset_x = (quadrant & 1) ? half_size : -half_size;
    float offset_y = (quadrant & 2) ? half_size : -half_size;
    new_center = make_float2(center.x + offset_x, center.y + offset_y);
    new_size = half_size;
}
__device__ float2 accelerationKernel(
    const CUDABody& body,
    const CUDANode* nodes,
    const CUDABody* bodies,
    size_t node_idx,
    float t_sq,
    float e_sq)
{
    float2 acc = make_float2(0.0f, 0.0f);

    while (true)
    {
        const CUDANode& n = nodes[node_idx];
        float2 d = n.active.pos - body.pos;
        float d_sq = length2(d);

        if (n.storage.quad.size * n.storage.quad.size < d_sq * t_sq)
        {
            if (d_sq > 0)
            {
                float inv_dist = fast_inv_sqrt(d_sq + e_sq);
                float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
                acc = acc + d * (n.active.mass * inv_dist_cubed);
            }

            if (n.storage.next == 0)
                break;
            node_idx = n.storage.next;
        }
        else if (n.is_leaf())
        {
            for (size_t i = n.storage.range.start; i < n.storage.range.end; ++i)
            {
                const CUDABody& other = bodies[i];
                float2 r = other.pos - body.pos;
                float r_sq = length2(r);

                if (r_sq > 0)
                {
                    float inv_dist = rsqrtf(r_sq + e_sq);
                    float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
                    acc = acc + r * (other.mass * inv_dist_cubed);
                }
            }

            if (n.storage.next == 0)
                break;
            node_idx = n.storage.next;
        }
        else
        {
            node_idx = n.active.subnode;
        }
    }

    return acc;
}

//multi phase build kernel
__global__ void buildKernel(
    CUDABody* bodies,
    CUDANode* nodes,
    int* nodeCount,
    float rootSize,
    int numBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies)
        return;

    // start at root node (thread 0 only)
    if (idx == 0)
    {
        nodes[0].active.pos = make_float2(0.0f, 0.0f);
        nodes[0].active.mass = 0.0f;
        nodes[0].active.subnode = 0;
        nodes[0].storage.quad.center = make_float2(0.0f, 0.0f);
        nodes[0].storage.quad.size = rootSize;
        *nodeCount = 1;
    }
    __syncthreads();

    // shared memory for mass/position reduction
    __shared__ float shared_mass[BLOCK_SIZE];
    __shared__ float2 shared_pos[BLOCK_SIZE];

    const CUDABody& body = bodies[idx];
    size_t currentNode = 0;

    // phase 1: places body in tree
    while (true)
    {
        CUDANode& node = nodes[currentNode];

        if (node.is_leaf())
        {
            if (node.active.mass == 0.0f)
            {
                // empty leaf? - claim it...
                node.active.pos = body.pos;
                node.active.mass = body.mass;
                break;
            }

            // need to subdivide !
            int newNodeIdx = atomicAdd(nodeCount, 4);
            if (newNodeIdx + 4 > MAX_NODES)
                break;

            node.active.subnode = newNodeIdx;
            float quarterSize = node.storage.quad.size * 0.25f;

            // init all child nodes
            for (int i = 0; i < 4; i++)
            {
                CUDANode& child = nodes[newNodeIdx + i];
                float2 offset = make_float2(
                    ((i & 1) ? quarterSize : -quarterSize),
                    ((i & 2) ? quarterSize : -quarterSize));
                child.storage.quad.center = node.storage.quad.center + offset;
                child.storage.quad.size = node.storage.quad.size * 0.5f;
                child.active.mass = 0.0f;
                child.active.subnode = 0;
            }

            // continue going down all appropriate quadrant
            size_t quadrant = ((body.pos.y > node.storage.quad.center.y) << 1) |
                (body.pos.x > node.storage.quad.center.x);
            currentNode = newNodeIdx + quadrant;
        }
        else
        {
            // (internal node - descend tree)
            size_t quadrant = ((body.pos.y > node.storage.quad.center.y) << 1) |
                (body.pos.x > node.storage.quad.center.x);
            currentNode = node.active.subnode + quadrant;
        }
    }

    // phase 2: the mass/center reduction
    shared_mass[threadIdx.x] = body.mass;
    shared_pos[threadIdx.x] = body.pos * body.mass;
    __syncthreads();

    // reduction within blocks
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            shared_mass[threadIdx.x] += shared_mass[threadIdx.x + s];
            shared_pos[threadIdx.x].x += shared_pos[threadIdx.x + s].x;
            shared_pos[threadIdx.x].y += shared_pos[threadIdx.x + s].y;
        }
        __syncthreads();
    }
    // position updates aren't properly weighted by mass i think
    // so multiple blocks might be updating the same node's mass and position concurrently
    // if (threadIdx.x == 0)
    // {
    //     float blockMass = shared_mass[0];
    //     float2 blockCenter = shared_pos[0] * (1.0f / blockMass);

    //     // Race condition here - multiple blocks updating same node
    //     atomicAdd(&nodes[currentNode].active.mass, blockMass);
    //     atomicAdd(&nodes[currentNode].active.pos.x, blockCenter.x);
    //     atomicAdd(&nodes[currentNode].active.pos.y, blockCenter.y);
    // }

    // Do one atomic update per block
    if (threadIdx.x == 0)
    {
        float blockMass = shared_mass[0];
        float2 blockCenter = shared_pos[0] * (1.0f / blockMass);

        atomicAdd(&nodes[currentNode].active.mass, blockMass);
        atomicAdd(&nodes[currentNode].active.pos.x, blockCenter.x);
        atomicAdd(&nodes[currentNode].active.pos.y, blockCenter.y);
    }
}
// Not to be confused with the "full" acc quadtree barnes hut cpu implementation
// however it is the force calculation component of barnes hut, working with the tree structure
// but requires the complete pipeline (build->compute->integrate) to implement the full optimization.
__global__ void forceComponent(
    CUDABody* bodies,
    const CUDANode* nodes,
    int numBodies,
    float theta,
    float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies)
        return;

    bodies[idx].acc = accelerationKernel(
        bodies[idx],
        nodes,
        bodies,
        0, // Start from root
        theta * theta,
        epsilon * epsilon);
}
__global__ void integrateKernel(
    CUDABody* bodies,
    int numBodies,
    float dt) //iterate
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBodies)
        return;

    bodies[idx].vel.x += bodies[idx].acc.x * dt;
    bodies[idx].vel.y += bodies[idx].acc.y * dt;
    bodies[idx].pos.x += bodies[idx].vel.x * dt;
    bodies[idx].pos.y += bodies[idx].vel.y * dt;
}
CUDAPhysics::CUDAPhysics(size_t maxBodies, float theta, float epsilon, size_t leaf_capacity) {
    debugPrint("Constructor start");

    // Check CUDA devices
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    std::cerr << "Found " << deviceCount << " CUDA devices" << std::endl;

    // Print capabilities of all devices
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cerr << "Device " << i << ": " << prop.name
            << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cerr << "Using GPU: " << prop.name << std::endl;

    this->maxBodies = maxBodies;
    this->t_sq = theta * theta;
    this->e_sq = epsilon * epsilon;
    this->leaf_capacity = leaf_capacity;

    try {
        allocateMemory();
        debugPrint("Memory allocated");
    }
    catch (const std::exception& e) {
        debugPrint(e.what());
        cleanup();
        throw;
    }
}



void CUDAPhysics::allocateMemory() {
    debugPrint("Allocating memory");

    size_t bodiesSize = maxBodies * sizeof(CUDABody);
    CUDA_CHECK(cudaMalloc(&d_bodies, bodiesSize));

    size_t nodesSize = MAX_NODES * sizeof(CUDANode);
    CUDA_CHECK(cudaMalloc(&d_nodes, nodesSize));

    CUDA_CHECK(cudaMalloc(&d_nodeCount, sizeof(int)));

    debugPrint("Memory allocated successfully");
}

void CUDAPhysics::cleanup() {
    if (this == nullptr) return;

    if (d_bodies) {
        cudaFree(d_bodies);
        d_bodies = nullptr;
    }
    if (d_nodes) {
        cudaFree(d_nodes);
        d_nodes = nullptr;
    }
    if (d_nodeCount) {
        cudaFree(d_nodeCount);
        d_nodeCount = nullptr;
    }
}
void CUDAPhysics::resize(size_t newSize)
{
    if (newSize > maxBodies)
    {
        cleanup();
        maxBodies = newSize;
        allocateMemory();
    }
}


void CUDAPhysics::updateBodies(const std::vector<Body>& bodies) {



    if (this == nullptr) {
        throw std::runtime_error("CUDAPhysics object is null");
    }

    if (d_bodies == nullptr) {
        throw std::runtime_error("CUDA device memory not allocated");
    }

    if (!d_bodies) {
        throw std::runtime_error("CUDA memory not allocated");
    }

    numBodies = bodies.size();
    if (numBodies > maxBodies) {
        resize(numBodies);
    }

    std::vector<CUDABody> hostBodies(numBodies);

    for (size_t i = 0; i < numBodies; i++) {
        if (i >= bodies.size()) {
            throw std::runtime_error("Index out of bounds");
        }
        hostBodies[i].pos = make_float2(bodies[i].pos.x, bodies[i].pos.y);
        hostBodies[i].vel = make_float2(bodies[i].vel.x, bodies[i].vel.y);
        hostBodies[i].acc = make_float2(0.0f, 0.0f);
        hostBodies[i].mass = bodies[i].mass;
        hostBodies[i].radius = bodies[i].radius;
    }

    CUDA_CHECK(cudaMemcpy(d_bodies, hostBodies.data(),
        numBodies * sizeof(CUDABody),
        cudaMemcpyHostToDevice));
}


    

void CUDAPhysics::step(float dt, float theta, float epsilon) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE);

    CUDA_CHECK(cudaMemset(d_nodeCount, 0, sizeof(int)));

    buildKernel << <grid, block >> > (d_bodies, d_nodes, d_nodeCount, 1000.0f, numBodies);
    CUDA_CHECK(cudaGetLastError());

    forceComponent << <grid, block >> > (d_bodies, d_nodes, numBodies, theta, epsilon);
    CUDA_CHECK(cudaGetLastError());

    integrateKernel << <grid, block >> > (d_bodies, numBodies, dt);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}


void CUDAPhysics::getBodies(std::vector<Body>& bodies)
{
    std::vector<CUDABody> hostBodies(numBodies);

    CUDA_CHECK(cudaMemcpy(hostBodies.data(), d_bodies,
        numBodies * sizeof(CUDABody),
        cudaMemcpyDeviceToHost));

    bodies.resize(numBodies);
    for (size_t i = 0; i < numBodies; i++)
    {
        bodies[i].pos = Vec2(hostBodies[i].pos.x, hostBodies[i].pos.y);
        bodies[i].vel = Vec2(hostBodies[i].vel.x, hostBodies[i].vel.y);
        bodies[i].acc = Vec2(hostBodies[i].acc.x, hostBodies[i].acc.y);
        bodies[i].mass = hostBodies[i].mass;
        bodies[i].radius = hostBodies[i].radius;
    }
}

