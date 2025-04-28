#include <iostream>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_set>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define alog(...) printf("\033[38;5;1m\033[48;5;15m(^(OO)^) /V/\033[0m\t" __VA_ARGS__)
#define alogt(tag, ...) printf("\033[38;5;1m\033[48;5;15m [%d] (^(OO)^) /V/\033[0m\t" tag, __VA_ARGS__)

class Texture;

void cuMain(float voxelSize, std::vector<float3>& host_points, std::vector<float3>& host_normals, std::vector<float3>& host_colors, float3 center);

bool ForceGPUPerformance();
void GenerateCUDATexture(unsigned int textureID, unsigned int width, unsigned int height, unsigned int xOffset, int yOffset);
void UpdateCUDATexture(unsigned int textureID, unsigned int width, unsigned int height, unsigned int xOffset, int yOffset);