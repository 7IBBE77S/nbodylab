#pragma once
#include "Vec2.h"
#include "Quad.h"
#include <vector>
#include <algorithm>
#include <array>
#include <iostream>


struct Range
{
    size_t start;
    size_t end;

    Range(size_t start = 0, size_t end = 0) : start(start), end(end) {}
    Range(const Range& other) : start(other.start), end(other.end) {}

    Range& operator=(const Range& other)
    {
        if (this != &other)
        {
            start = other.start;
            end = other.end;
        }
        return *this;
    }

    size_t size() const { return end - start; }
};

struct alignas(32) Node
{
    // put frequently accessed data grouped together for cache efficiency
    // Hot data together
    struct alignas(32)
    {
        Vec2 pos;   // 8 bytes
        float mass; // 4 bytes
        Quad quad;  // 12 bytes
    } data;          // total 24 byte aligned
    // Cold data
    size_t children; // 8 bytes - Index to first child
    size_t next;     // 8 bytes - Index to next node
    Range bodies;    // 16 bytes - Body range using semantic struct
    size_t depth;

    Node(size_t next = 0, Quad quad = Quad(), size_t depth = 0)
        : data{ Vec2::zero(), 0.0f, quad }, children(0), next(next), bodies{ 0, 0 }, depth(depth) {}

    bool is_leaf() const { return children == 0; }
    bool is_branch() const { return children != 0; }
    bool is_empty() const { return data.mass == 0.0f; }
};
