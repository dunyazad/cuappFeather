#include "main.cuh"

__global__ void hello_kernel() {
	printf("Hello from CUDA kernel!\\n");
}

void TestCUDA()
{
	hello_kernel << <1, 1 >> > ();
	cudaDeviceSynchronize();
	std::cout << "CUDA finished\\n";
}

//#include <Serialization.hpp>

#include <nvtx3/nvToolsExt.h>

//int cuMain(const std::vector<float3>& host_points);
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

namespace Clustering
{
	struct Voxel
	{
		//float3 position;
		unsigned int label;
	};

	__global__ void Kernel_ClearVoxels(
		Voxel* d_voxels,
		unsigned int numberOfVoxels,
		dim3 volumeDimensions,
		float voxelSize,
		float3 volumeMin,
		float3 volumeCenter)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= volumeDimensions.x * volumeDimensions.y * volumeDimensions.z) return;

		//d_voxels[threadid].position = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		d_voxels[threadid].label = 0;
	}

	void ClearVoxels(
		Voxel* d_voxels,
		unsigned int numberOfVoxels,
		dim3 volumeDimensions,
		float voxelSize,
		float3 volumeMin,
		float3 volumeCenter)
	{
		nvtxRangePushA("ClearVoxels");

		unsigned int blockSize = 256;
		unsigned int gridSize = (numberOfVoxels + blockSize - 1) / blockSize;
		Kernel_ClearVoxels << <gridSize, blockSize >> > (d_voxels, numberOfVoxels, volumeDimensions, voxelSize, volumeMin, volumeCenter);

		cudaDeviceSynchronize();
		nvtxRangePop();
	}

	__global__ void Kernel_OccupyVoxels(
		float* d_points,
		unsigned int numberOfPoints,
		Voxel* d_voxels,
		unsigned int numberOfVoxels,
		dim3 volumeDimensions,
		float voxelSize,
		float3 volumeMin,
		float3 volumeCenter,
		uint3* occupiedVoxelIndices,
		unsigned int* numberOfOccupiedVoxelIndices,
		unsigned int* occupiedPointIndices,
		unsigned int* numberOfOccupiedPointIndices)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= numberOfPoints) return;

		auto gx = d_points[threadid * 3];
		auto gy = d_points[threadid * 3 + 1];
		auto gz = d_points[threadid * 3 + 2];

		if (gx < volumeMin.x || gx > volumeMin.x + volumeDimensions.x * voxelSize ||
			gy < volumeMin.y || gy > volumeMin.y + volumeDimensions.y * voxelSize ||
			gz < volumeMin.z || gz > volumeMin.z + volumeDimensions.z * voxelSize)
		{
			return;
		}

		unsigned int ix = (unsigned int)floorf((gx - volumeMin.x) / voxelSize);
		unsigned int iy = (unsigned int)floorf((gy - volumeMin.y) / voxelSize);
		unsigned int iz = (unsigned int)floorf((gz - volumeMin.z) / voxelSize);

		if (ix >= volumeDimensions.x || iy >= volumeDimensions.y || iz >= volumeDimensions.z) return;

		unsigned int volumeIndex = iz * volumeDimensions.x * volumeDimensions.y + iy * volumeDimensions.x + ix;
		auto& voxel = d_voxels[volumeIndex];

		//voxel.position.x = volumeMin.x + ix * voxelSize;
		//voxel.position.y = volumeMin.y + iy * voxelSize;
		//voxel.position.z = volumeMin.z + iz * voxelSize;
		voxel.label = volumeIndex;

		//alog("%f, %f, %f\n", voxel.position.x, voxel.position.y, voxel.position.z);

		auto voxelIndex = atomicAdd(numberOfOccupiedVoxelIndices, 1);
		occupiedVoxelIndices[voxelIndex] = make_uint3(ix, iy, iz);

		auto pointIndex = atomicAdd(numberOfOccupiedPointIndices, 1);
		occupiedPointIndices[pointIndex] = threadid;

		//alog("%d\n", index);
	}

	void OccupyVoxels(
		float* d_points,
		unsigned int numberOfPoints,
		Voxel* d_voxels,
		unsigned int numberOfVoxels,
		dim3 volumeDimensions,
		float voxelSize,
		float3 volumeMin,
		float3 volumeCenter,
		uint3* occupiedVoxelIndices,
		unsigned int* numberOfOccupiedVoxelIndices,
		unsigned int* occupiedPointIndices,
		unsigned int* numberOfOccupiedPointIndices)
	{
		nvtxRangePush("OccupyVoxels");

		unsigned int blockSize = 256;
		unsigned int gridSize = (numberOfPoints + blockSize - 1) / blockSize;

		Kernel_OccupyVoxels << <gridSize, blockSize >> > (
			d_points,
			numberOfPoints,
			d_voxels,
			numberOfVoxels,
			volumeDimensions,
			voxelSize,
			volumeMin,
			volumeCenter,
			occupiedVoxelIndices,
			numberOfOccupiedVoxelIndices,
			occupiedPointIndices,
			numberOfOccupiedPointIndices);

		cudaDeviceSynchronize();
		nvtxRangePop();
	}

	__device__ __forceinline__ unsigned int FindRoot(Voxel* voxels, unsigned int idx) {
		while (true) {
			unsigned int parent = voxels[idx].label;
			unsigned int grand = voxels[parent].label;
			if (parent == idx) break;
			if (parent != grand) voxels[idx].label = grand;
			idx = parent;
		}
		return idx;
	}

	__device__ __forceinline__ void Union(Voxel* voxels, unsigned int a, unsigned int b) {
		unsigned int rootA = FindRoot(voxels, a);
		unsigned int rootB = FindRoot(voxels, b);
		if (rootA != rootB) {
			if (rootA < rootB)
				atomicMin(&voxels[rootB].label, rootA);
			else
				atomicMin(&voxels[rootA].label, rootB);
		}
	}

	__global__ void Kernel_InterBlockMerge(
		Voxel* voxels,
		uint3* occupiedIndices,
		unsigned int numOccupied,
		dim3 dims)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= numOccupied) return;

		uint3 idx = occupiedIndices[tid];
		unsigned int center = idx.z * dims.y * dims.x + idx.y * dims.x + idx.x;
		//if (voxels[center].position.x == FLT_MAX) return;
		if (0 == voxels[center].label) return;

		for (int dz = -1; dz <= 1; dz++) {
			int nz = idx.z + dz;
			if (nz < 0 || nz >= dims.z) continue;
			for (int dy = -1; dy <= 1; dy++) {
				int ny = idx.y + dy;
				if (ny < 0 || ny >= dims.y) continue;
				for (int dx = -1; dx <= 1; dx++) {
					int nx = idx.x + dx;
					if (nx < 0 || nx >= dims.x) continue;
					if (dx == 0 && dy == 0 && dz == 0) continue;

					unsigned int neighbor = nz * dims.y * dims.x + ny * dims.x + nx;
					//if (voxels[neighbor].position.x != FLT_MAX) {
					if (0 != voxels[neighbor].label) {
						Union(voxels, center, neighbor);
					}
				}
			}
		}
	}

	__global__ void Kernel_InterBlockMergeP(
		Voxel* voxels,
		uint3* occupiedIndices,
		unsigned int numOccupied,
		dim3 dims)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= numOccupied) return;

		uint3 idx = occupiedIndices[tid];
		unsigned int center = idx.z * dims.y * dims.x + idx.y * dims.x + idx.x;
		//if (voxels[center].position.x == FLT_MAX) return;
		if (0 == voxels[center].label) return;

		for (int dz = 0; dz <= 1; dz++) {
			int nz = idx.z + dz;
			if (nz < 0 || nz >= dims.z) continue;
			for (int dy = 0; dy <= 1; dy++) {
				int ny = idx.y + dy;
				if (ny < 0 || ny >= dims.y) continue;
				for (int dx = 0; dx <= 1; dx++) {
					int nx = idx.x + dx;
					if (nx < 0 || nx >= dims.x) continue;
					if (dx == 0 && dy == 0 && dz == 0) continue;

					unsigned int neighbor = nz * dims.y * dims.x + ny * dims.x + nx;
					//if (voxels[neighbor].position.x != FLT_MAX) {
					if (0 != voxels[neighbor].label) {
						Union(voxels, center, neighbor);
					}
				}
			}
		}
	}

	__global__ void Kernel_InterBlockMergeN(
		Voxel* voxels,
		uint3* occupiedIndices,
		unsigned int numOccupied,
		dim3 dims)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= numOccupied) return;

		uint3 idx = occupiedIndices[tid];
		unsigned int center = idx.z * dims.y * dims.x + idx.y * dims.x + idx.x;
		//if (voxels[center].position.x == FLT_MAX) return;
		if (0 == voxels[center].label) return;

		for (int dz = -1; dz <= 0; dz++) {
			int nz = idx.z + dz;
			if (nz < 0 || nz >= dims.z) continue;
			for (int dy = -1; dy <= 0; dy++) {
				int ny = idx.y + dy;
				if (ny < 0 || ny >= dims.y) continue;
				for (int dx = -1; dx <= 0; dx++) {
					int nx = idx.x + dx;
					if (nx < 0 || nx >= dims.x) continue;
					if (dx == 0 && dy == 0 && dz == 0) continue;

					unsigned int neighbor = nz * dims.y * dims.x + ny * dims.x + nx;
					//if (voxels[neighbor].position.x != FLT_MAX) {
					if (0 != voxels[neighbor].label) {
						Union(voxels, center, neighbor);
					}
				}
			}
		}
	}

	__global__ void Kernel_CompressLabels(Voxel* voxels, unsigned int N) {
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= N) return;

		//if (voxels[tid].position.x != FLT_MAX) {
		if (0 != voxels[tid].label) {
			voxels[tid].label = FindRoot(voxels, tid);
		}
	}

	void ConnectedComponentLabeling(
		Voxel* d_voxels,
		uint3* occupiedVoxelIndices,
		unsigned int numberOfOccupiedVoxelIndices,
		dim3 volumeDimensions)
	{
		unsigned int totalVoxels = volumeDimensions.x * volumeDimensions.y * volumeDimensions.z;
		unsigned int blockSize = 256;
		unsigned int gridVoxels = (totalVoxels + blockSize - 1) / blockSize;
		unsigned int gridOccupied = (numberOfOccupiedVoxelIndices + blockSize - 1) / blockSize;

		Kernel_InterBlockMerge << <gridOccupied, blockSize >> > (d_voxels, occupiedVoxelIndices, numberOfOccupiedVoxelIndices, volumeDimensions);
		cudaDeviceSynchronize();

		//Kernel_InterBlockMergeP << <gridOccupied, blockSize >> > (d_voxels, occupiedVoxelIndices, numberOfOccupiedVoxelIndices, volumeDimensions);
		//cudaDeviceSynchronize();

		//Kernel_InterBlockMergeN << <gridOccupied, blockSize >> > (d_voxels, occupiedVoxelIndices, numberOfOccupiedVoxelIndices, volumeDimensions);
		//cudaDeviceSynchronize();

		//Kernel_CompressLabels << <gridVoxels, blockSize >> > (d_voxels, totalVoxels);
		//cudaDeviceSynchronize();
	}

__global__ void Kernel_GetLabels(
	float* d_points,
	unsigned int numberOfPoints,
	Voxel* d_voxels,
	unsigned int numberOfVoxels,
	dim3 volumeDimensions,
	float voxelSize,
	float3 volumeMin,
	float3 volumeCenter,
	unsigned int* d_labels)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= numberOfPoints) return;

	auto gx = d_points[threadid * 3];
	auto gy = d_points[threadid * 3 + 1];
	auto gz = d_points[threadid * 3 + 2];

	if (gx < volumeMin.x || gx > volumeMin.x + volumeDimensions.x * voxelSize ||
		gy < volumeMin.y || gy > volumeMin.y + volumeDimensions.y * voxelSize ||
		gz < volumeMin.z || gz > volumeMin.z + volumeDimensions.z * voxelSize)
	{
		return;
	}

	unsigned int ix = (unsigned int)floorf((gx - volumeMin.x) / voxelSize);
	unsigned int iy = (unsigned int)floorf((gy - volumeMin.y) / voxelSize);
	unsigned int iz = (unsigned int)floorf((gz - volumeMin.z) / voxelSize);

	if (ix >= volumeDimensions.x || iy >= volumeDimensions.y || iz >= volumeDimensions.z) return;

	unsigned int volumeIndex = iz * volumeDimensions.x * volumeDimensions.y + iy * volumeDimensions.x + ix;
	auto& voxel = d_voxels[volumeIndex];

	d_labels[threadid] = voxel.label;
}

std::vector<unsigned int> GetLabels(
	float* d_points,
	unsigned int numberOfPoints,
	Voxel* d_voxels,
	unsigned int numberOfVoxels,
	dim3 volumeDimensions,
	float voxelSize,
	float3 volumeMin,
	float3 volumeCenter)
{
	unsigned int* d_labels = nullptr;
	cudaMalloc(&d_labels, sizeof(unsigned int) * numberOfPoints);
	cudaMemset(d_labels, -1, sizeof(unsigned int) * numberOfPoints);
	cudaDeviceSynchronize();

	unsigned int blockSize = 256;
	unsigned int gridSize = (numberOfPoints + blockSize - 1) / blockSize;

	Kernel_GetLabels << <gridSize, blockSize >> > (
		d_points,
		numberOfPoints,
		d_voxels,
		numberOfVoxels,
		volumeDimensions,
		voxelSize,
		volumeMin,
		volumeCenter,
		d_labels);

	cudaDeviceSynchronize();
	nvtxRangePop();

	std::vector<unsigned int> result(numberOfPoints);
	cudaMemcpy(result.data(), d_labels, sizeof(unsigned int) * numberOfPoints, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(d_labels);

	return result;
}

void VisualizeVoxels(
	Voxel* d_voxels,
	unsigned int numberOfVoxels,
	dim3 volumeDimensions,
	float voxelSize,
	float3 volumeMin)
{
	nvtxRangePush("VisualizeVoxels");

	Voxel* h_voxels = new Voxel[numberOfVoxels];
	cudaMemcpy(h_voxels, d_voxels, sizeof(Voxel) * numberOfVoxels, cudaMemcpyDeviceToHost);

	std::unordered_map<unsigned int, std::tuple<unsigned char, unsigned char, unsigned char>> labelToColor;

	std::unordered_map<unsigned int, unsigned int> labelHistogram;

	for (size_t i = 0; i < numberOfVoxels; i++)
	{
		auto& voxel = h_voxels[i];

		//if (voxel.position.x != FLT_MAX) // Only visualize occupied voxels
		if (0 != voxel.label) // Only visualize occupied voxels
		{
			unsigned int label = voxel.label;

			// Assign a unique color per label using a hash function
			if (labelToColor.find(label) == labelToColor.end())
			{
				unsigned char r = (label * 53) % 256;
				unsigned char g = (label * 97) % 256;
				unsigned char b = (label * 151) % 256;
				labelToColor[label] = std::make_tuple(r, g, b);
			}

			// Get the assigned color
			auto [r, g, b] = labelToColor[label];

			// Visualize the voxel with the computed color
/*               VD::AddCube("labeled voxels", { voxel.position.x, voxel.position.y, voxel.position.z },
				   0.05f, { r, g, b, 255 });*/

			if (0 == labelHistogram.count(voxel.label))
			{
				labelHistogram[voxel.label] = 1;
			}
			else
			{
				labelHistogram[voxel.label] += 1;
			}
		}
	}

	int i = 0;
	for (auto& [label, count] : labelHistogram)
	{
		alog("[%4d] voxel label - %16d : count - %8d\n", i++, label, count);
	}
	alog("\n");

	delete[] h_voxels;

	cudaDeviceSynchronize();
	nvtxRangePop();
}

struct ClusteringCacheInfo
{
	float voxelSize;
	dim3 cacheDimensions;
	unsigned int numberOfVoxels;
	float3 cacheMin;

	cudaArray* cacheData3D = nullptr;
	cudaSurfaceObject_t surfaceObject3D;

	uint3* occupiedVoxelIndices;
	unsigned int* numberOfOccupiedVoxelIndices;
};
}


std::vector<unsigned int> cuMain(const std::vector<float3>& host_points, float3 center)
{
	nvtxRangePush("TestClustering");

	float* d_points = nullptr;
	cudaMalloc(&d_points, sizeof(float) * host_points.size() * 3);
	cudaMemcpy(d_points, host_points.data(), sizeof(float) * host_points.size() * 3, cudaMemcpyHostToDevice);

	unsigned int numberOfPoints = host_points.size();
	//dim3 volumeDimensions(200, 300, 400);
	dim3 volumeDimensions(400, 400, 400);
	unsigned int numberOfVoxels = volumeDimensions.x * volumeDimensions.y * volumeDimensions.z;
	float voxelSize = 0.1f;
	//float3 volumeCenter = make_float3(3.9904f, -15.8357f, -7.2774f);
	//float3 volumeCenter = make_float3(4.0f, -15.0f, -7.0f);
	float3 volumeCenter = make_float3(center.x, center.y, center.z);
	float3 volumeMin = make_float3(
		volumeCenter.x - (float)(volumeDimensions.x / 2) * voxelSize,
		volumeCenter.y - (float)(volumeDimensions.y / 2) * voxelSize,
		volumeCenter.z - (float)(volumeDimensions.z / 2) * voxelSize);

	Clustering::Voxel* d_voxels = nullptr;
	cudaMalloc(&d_voxels, sizeof(Clustering::Voxel) * numberOfVoxels);

	uint3* occupiedVoxelIndices = nullptr;
	cudaMalloc(&occupiedVoxelIndices, sizeof(uint3) * 5000000);
	unsigned int* numberOfOccupiedVoxelIndices = nullptr;
	cudaMalloc(&numberOfOccupiedVoxelIndices, sizeof(unsigned int));
	cudaMemset(numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));

	unsigned int* occupiedPointIndices = nullptr;
	cudaMalloc(&occupiedPointIndices, sizeof(unsigned int) * 5000000);
	unsigned int* numberOfOccupiedPointIndices = nullptr;
	cudaMalloc(&numberOfOccupiedPointIndices, sizeof(unsigned int));
	cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));

	cudaDeviceSynchronize();

	for (size_t i = 0; i < 10; i++)
	{
		cudaMemset(numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));
		cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));

		ClearVoxels(d_voxels, numberOfVoxels, volumeDimensions, voxelSize, volumeMin, volumeCenter);

		OccupyVoxels(
			d_points,
			numberOfPoints,
			d_voxels,
			numberOfVoxels,
			volumeDimensions,
			voxelSize,
			volumeMin,
			volumeCenter,
			occupiedVoxelIndices,
			numberOfOccupiedVoxelIndices,
			occupiedPointIndices,
			numberOfOccupiedPointIndices);

		unsigned int h_numberOfOccupiedVoxelIndices = 0;
		cudaMemcpy(&h_numberOfOccupiedVoxelIndices, numberOfOccupiedVoxelIndices, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		nvtxRangePushA("CCL");

		ConnectedComponentLabeling(d_voxels, occupiedVoxelIndices, h_numberOfOccupiedVoxelIndices, volumeDimensions);

		nvtxRangePop();
	}
	//VisualizeVoxels(
	//	d_voxels,
	//	numberOfVoxels,
	//	volumeDimensions,
	//	voxelSize,
	//	volumeMin);

	std::vector<unsigned int> result = GetLabels(
		d_points,
		numberOfPoints,
		d_voxels,
		numberOfVoxels,
		volumeDimensions,
		voxelSize,
		volumeMin,
		volumeCenter);

	{
		std::unordered_map<unsigned int, unsigned int> labelHistogram;

		for (auto& i : result)
		{
			if (0 == labelHistogram.count(i))
			{
				labelHistogram[i] = 1;
			}
			else
			{
				labelHistogram[i] += 1;
			}
		}

		unsigned int i = 0;
		for (auto& [label, count] : labelHistogram)
		{
			alog("[%4d] point label - %16d : count - %8d\n", i++, label, count);
		}
		alog("\n");
	}

	unsigned int h_numberOfOccupiedPointIndices = 0;
	cudaMemcpy(&h_numberOfOccupiedPointIndices, numberOfOccupiedPointIndices, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int* h_occupiedPointIndices = new unsigned int[h_numberOfOccupiedPointIndices];
	cudaMemcpy(h_occupiedPointIndices, occupiedPointIndices, sizeof(unsigned int) * h_numberOfOccupiedPointIndices, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < h_numberOfOccupiedPointIndices; i++)
	{
		auto index = h_occupiedPointIndices[i];
		auto p = host_points[index];

		//VD::AddSphere("In Area", { x,y, z }, 0.05f, { 255, 0, 0 });
	}

	cudaFree(d_points);
	cudaFree(d_voxels);
	cudaFree(occupiedVoxelIndices);
	cudaFree(numberOfOccupiedVoxelIndices);
	cudaFree(occupiedPointIndices);
	cudaFree(numberOfOccupiedPointIndices);

	delete[] h_occupiedPointIndices;

	cudaDeviceSynchronize();
	nvtxRangePop();

	return result;
}