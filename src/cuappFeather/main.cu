#pragma warning(disable : 4819)

#include "main.cuh"

#include <nvtx3/nvToolsExt.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include <Serialization.hpp>

void cuMain(
	float voxelSize,
	std::vector<float3>& host_points,
	std::vector<float3>& host_normals,
	std::vector<float3>& host_colors,
	float3 center)
{
}
