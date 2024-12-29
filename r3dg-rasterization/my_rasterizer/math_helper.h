#ifndef CUDA_RASTERIZER_MATH_HELPER_H_INCLUDED
#define CUDA_RASTERIZER_MATH_HELPER_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <math.h>

__device__ float2 normalize(const float2& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y);
    if (len > 0) {
        return make_float2(v.x / len, v.y / len);
    }
    return v;
}

__device__ float2 operator*(float s, const float2& v) {
    return make_float2(v.x * s, v.y * s);
}

__device__ float2 operator*(const float2& v, float s) {
    return make_float2(v.x * s, v.y * s);
}

__device__ float dot(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}

__device__ float2& operator*=(float2& v, float s) {
    v.x *= s;
    v.y *= s;
    return v;
}

#endif