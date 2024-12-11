#pragma once

#include <cuda_runtime.h>

#include "Vec2.h"
#include <vector>
#include <future>
#include <thread>

struct alignas(32) Body
{
    Vec2 pos;     // 8 bytes
    Vec2 vel;     // 8 bytes
    Vec2 acc;     // 8 bytes
    float mass;   // 4 bytes
    float radius; // 4 bytes
    // total: 32 bytes cache aligned

    Body() = default;
    Body(Vec2 pos, Vec2 vel, float mass, float radius)
        : pos(pos), vel(vel), acc(Vec2::zero()), mass(mass), radius(radius) {}

    static Body at_rest(Vec2 pos, float mass, float radius)
    {
        return Body(pos, Vec2::zero(), mass, radius);
    }

    static Body with_velocity(Vec2 pos, Vec2 vel, float mass, float radius)
    {
        return Body(pos, vel, mass, radius);
    }
    static unsigned int get_hardware_threads()
    {
        return std::thread::hardware_concurrency();
    }

    // physics update
    inline void update(float dt)
    {
        vel += acc * dt; // 152ms of performance loss
        pos += vel * dt; //230ms of performance loss
    }




    bool collides_with(const Body& other) const
    {
        Vec2 d = other.pos - pos;
        float r = radius + other.radius;
        return d.mag_sq() <= r * r;
    }

    void resolve_collision(Body& other)
    {
        Vec2 d = other.pos - pos;
        float r = radius + other.radius;

        if (d.mag_sq() > r * r)
            return;

        Vec2 n = d.normalized();
        Vec2 v = other.vel - vel;

        float j = -(1.0f + 0.5f) * v.dot(n);
        j /= 1.0f / mass + 1.0f / other.mass;

        vel -= n * (j / mass);
        other.vel += n * (j / other.mass);

        float overlap = r - d.mag();
        Vec2 separation = n * (overlap * 0.5f);
        pos -= separation;
        other.pos += separation;
    }

    float kinetic_energy() const
    {
        return 0.5f * mass * vel.mag_sq();
    }

    Vec2 momentum() const
    {
        return vel * mass;
    }
};
