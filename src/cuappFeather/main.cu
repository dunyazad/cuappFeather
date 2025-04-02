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

namespace Clustering
{
	struct Voxel
	{
		unsigned int label;
	};

	struct ClusteringCacheInfo
	{
		float voxelSize = 0.1f;
		dim3 cacheDimensions = dim3(200, 300, 400);
		unsigned int numberOfVoxels;
		float3 cacheMin = make_float3(0.0f, 0.0f, 0.0f);

		Voxel* voxels = nullptr;

		uint3* occupiedVoxelIndices = nullptr;
		unsigned int* numberOfOccupiedVoxelIndices = nullptr;
	};

	class ClusteringCache
	{
	public:
		ClusteringCache();
		ClusteringCache(const ClusteringCacheInfo& info);
		~ClusteringCache();

		void Initialize();
		void Terminate();

		void Clear();
		void OccupyVoxels();

		void SetHostPoints(const std::vector<float3>& hostPoints, const std::vector<float3>& hostNormals, const std::vector<float3>& hostColors);
		void SetDevicePoints(float* devicePoints, float* deviceNormals, float* deviceColors, unsigned int numberOfPoints);

		void RunClustering(int iterations = 1);
		std::vector<unsigned int> GetLabels();

		void ConnectedComponentLabeling(unsigned int numberOfOccupiedVoxelIndices);

	private:
		ClusteringCacheInfo info;

		bool setFromHost = false;
		float* d_points = nullptr;
		float* d_normals = nullptr;
		float* d_colors = nullptr;
		unsigned int numberOfPoints = 0;
		unsigned int* d_labels = nullptr;

		unsigned int* occupiedPointIndices = nullptr;
		unsigned int* numberOfOccupiedPointIndices = nullptr;
	};

	ClusteringCache::ClusteringCache()
	{
	}

	ClusteringCache::ClusteringCache(const ClusteringCacheInfo& info)
		: info(info)
	{
	}

	ClusteringCache::~ClusteringCache()
	{
	}

	void ClusteringCache::Initialize()
	{
		cudaMalloc(&info.voxels, sizeof(Voxel) * info.numberOfVoxels);
		cudaMemset(info.voxels, 0, sizeof(Voxel) * info.numberOfVoxels);

		cudaMalloc(&info.occupiedVoxelIndices, info.numberOfVoxels * sizeof(uint3));
		cudaMalloc(&info.numberOfOccupiedVoxelIndices, sizeof(unsigned int));
		cudaMemset(info.numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));
	}

	void ClusteringCache::Terminate()
	{
		if (info.voxels)
			cudaFree(info.voxels);

		if (info.occupiedVoxelIndices)
			cudaFree(info.occupiedVoxelIndices);

		if (info.numberOfOccupiedVoxelIndices)
			cudaFree(info.numberOfOccupiedVoxelIndices);

		if (occupiedPointIndices)
			cudaFree(occupiedPointIndices);

		if (numberOfOccupiedPointIndices)
			cudaFree(numberOfOccupiedPointIndices);

		if (setFromHost)
		{
			if (d_points)
				cudaFree(d_points);

			if (d_normals)
				cudaFree(d_normals);

			if (d_colors)
				cudaFree(d_colors);
		}
	}

	__global__ void Kernel_ClearVoxels(ClusteringCacheInfo info)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= info.cacheDimensions.x * info.cacheDimensions.y * info.cacheDimensions.z) return;

		info.voxels[threadid].label = 0;
	}

	void ClusteringCache::Clear()
	{
		nvtxRangePushA("ClearVoxels");

		unsigned int blockSize = 256;
		unsigned int gridSize = (info.numberOfVoxels + blockSize - 1) / blockSize;
		Kernel_ClearVoxels << <gridSize, blockSize >> > (info);

		cudaDeviceSynchronize();
		nvtxRangePop();
	}

	__global__ void Kernel_OccupyVoxels(
		ClusteringCacheInfo info,
		float* d_points,
		float* d_colors,
		unsigned int numberOfPoints,
		unsigned int* occupiedPointIndices,
		unsigned int* numberOfOccupiedPointIndices)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= numberOfPoints) return;

		auto gx = d_points[threadid * 3];
		auto gy = d_points[threadid * 3 + 1];
		auto gz = d_points[threadid * 3 + 2];

		if (gx < info.cacheMin.x || gx > info.cacheMin.x + info.cacheDimensions.x * info.voxelSize ||
			gy < info.cacheMin.y || gy > info.cacheMin.y + info.cacheDimensions.y * info.voxelSize ||
			gz < info.cacheMin.z || gz > info.cacheMin.z + info.cacheDimensions.z * info.voxelSize)
		{
			return;
		}

		unsigned int ix = (unsigned int)floorf((gx - info.cacheMin.x) / info.voxelSize);
		unsigned int iy = (unsigned int)floorf((gy - info.cacheMin.y) / info.voxelSize);
		unsigned int iz = (unsigned int)floorf((gz - info.cacheMin.z) / info.voxelSize);

		if (ix >= info.cacheDimensions.x || iy >= info.cacheDimensions.y || iz >= info.cacheDimensions.z) return;

		unsigned int cacheIndex = iz * info.cacheDimensions.x * info.cacheDimensions.y + iy * info.cacheDimensions.x + ix;
		info.voxels[cacheIndex].label = cacheIndex;

		auto voxelIndex = atomicAdd(info.numberOfOccupiedVoxelIndices, 1);
		info.occupiedVoxelIndices[voxelIndex] = make_uint3(ix, iy, iz);

		auto pointIndex = atomicAdd(numberOfOccupiedPointIndices, 1);
		occupiedPointIndices[pointIndex] = threadid;
	}

	void ClusteringCache::OccupyVoxels()
	{
		nvtxRangePush("OccupyVoxels");

		unsigned int blockSize = 256;
		unsigned int gridSize = (numberOfPoints + blockSize - 1) / blockSize;

		Kernel_OccupyVoxels << <gridSize, blockSize >> > (
			info,
			d_points,
			d_colors,
			numberOfPoints,
			occupiedPointIndices,
			numberOfOccupiedPointIndices);

		cudaDeviceSynchronize();
		nvtxRangePop();
	}

	void ClusteringCache::SetHostPoints(const std::vector<float3>& hostPoints, const std::vector<float3>& hostNormals, const std::vector<float3>& hostColors)
	{
		setFromHost = true;

		numberOfPoints = static_cast<unsigned int>(hostPoints.size());
		
		cudaMalloc(&d_points, sizeof(float3) * numberOfPoints);
		cudaMalloc(&d_normals, sizeof(float3) * numberOfPoints);
		cudaMalloc(&d_colors, sizeof(float3) * numberOfPoints);

		cudaMalloc(&occupiedPointIndices, sizeof(unsigned int) * numberOfPoints);
		cudaMalloc(&numberOfOccupiedPointIndices, sizeof(unsigned int));
		cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));

		cudaMemcpy(d_points, hostPoints.data(), sizeof(float3) * numberOfPoints, cudaMemcpyHostToDevice);
		cudaMemcpy(d_normals, hostNormals.data(), sizeof(float3) * numberOfPoints, cudaMemcpyHostToDevice);
		cudaMemcpy(d_colors, hostColors.data(), sizeof(float3) * numberOfPoints, cudaMemcpyHostToDevice);
	}

	void ClusteringCache::SetDevicePoints(float* devicePoints, float* deviceNormals, float* deviceColors, unsigned int numberOfPoints)
	{
		this->numberOfPoints = numberOfPoints;
		d_points = devicePoints;
		d_normals = deviceNormals;
		d_colors = deviceColors;

		cudaMalloc(&occupiedPointIndices, sizeof(unsigned int) * numberOfPoints);
		cudaMalloc(&numberOfOccupiedPointIndices, sizeof(unsigned int));
		cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));
	}

	void ClusteringCache::RunClustering(int iterations)
	{
		for (int i = 0; i < iterations; ++i)
		{
			cudaMemset(info.numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));
			cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));

			Clear();

			OccupyVoxels();

			unsigned int h_occupiedCount = 0;
			cudaMemcpy(&h_occupiedCount, info.numberOfOccupiedVoxelIndices, sizeof(unsigned int), cudaMemcpyDeviceToHost);

			nvtxRangePushA("CCL");

			ConnectedComponentLabeling(h_occupiedCount);

			nvtxRangePop();
		}
	}

	__global__ void Kernel_GetLabels(
		ClusteringCacheInfo info,
		float* d_points,
		unsigned int numberOfPoints,
		unsigned int* d_labels)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= numberOfPoints) return;

		auto gx = d_points[threadid * 3];
		auto gy = d_points[threadid * 3 + 1];
		auto gz = d_points[threadid * 3 + 2];

		if (gx < info.cacheMin.x || gx > info.cacheMin.x + info.cacheDimensions.x * info.voxelSize ||
			gy < info.cacheMin.y || gy > info.cacheMin.y + info.cacheDimensions.y * info.voxelSize ||
			gz < info.cacheMin.z || gz > info.cacheMin.z + info.cacheDimensions.z * info.voxelSize)
		{
			return;
		}

		unsigned int ix = (unsigned int)floorf((gx - info.cacheMin.x) / info.voxelSize);
		unsigned int iy = (unsigned int)floorf((gy - info.cacheMin.y) / info.voxelSize);
		unsigned int iz = (unsigned int)floorf((gz - info.cacheMin.z) / info.voxelSize);

		if (ix >= info.cacheDimensions.x || iy >= info.cacheDimensions.y || iz >= info.cacheDimensions.z) return;

		unsigned int cacheIndex = iz * info.cacheDimensions.x * info.cacheDimensions.y + iy * info.cacheDimensions.x + ix;
		auto& voxel = info.voxels[cacheIndex];

		d_labels[threadid] = voxel.label;
	}

	std::vector<unsigned int> ClusteringCache::GetLabels()
	{
		unsigned int* d_labels = nullptr;
		cudaMalloc(&d_labels, sizeof(unsigned int) * numberOfPoints);
		cudaMemset(d_labels, -1, sizeof(unsigned int) * numberOfPoints);
		cudaDeviceSynchronize();

		unsigned int blockSize = 256;
		unsigned int gridSize = (numberOfPoints + blockSize - 1) / blockSize;

		Kernel_GetLabels << <gridSize, blockSize >> > (
			info,
			d_points,
			numberOfPoints,
			d_labels);

		cudaDeviceSynchronize();
		nvtxRangePop();

		std::vector<unsigned int> result(numberOfPoints);
		cudaMemcpy(result.data(), d_labels, sizeof(unsigned int) * numberOfPoints, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		cudaFree(d_labels);

		return result;
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

	__global__ void Kernel_InterBlockMerge26Way(ClusteringCacheInfo info)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= *info.numberOfOccupiedVoxelIndices) return;

		uint3 idx = info.occupiedVoxelIndices[threadid];
		unsigned int center = idx.z * info.cacheDimensions.y * info.cacheDimensions.x + idx.y * info.cacheDimensions.x + idx.x;
		if (0 == info.voxels[center].label) return;

		for (int zOffset = -1; zOffset <= 1; zOffset++) {
			int nz = idx.z + zOffset;
			if (nz < 0 || nz >= info.cacheDimensions.z) continue;
			for (int dy = -1; dy <= 1; dy++) {
				int ny = idx.y + dy;
				if (ny < 0 || ny >= info.cacheDimensions.y) continue;
				for (int dx = -1; dx <= 1; dx++) {
					int nx = idx.x + dx;
					if (nx < 0 || nx >= info.cacheDimensions.x) continue;
					if (dx == 0 && dy == 0 && zOffset == 0) continue;

					unsigned int neighbor = nz * info.cacheDimensions.y * info.cacheDimensions.x + ny * info.cacheDimensions.x + nx;
					if (0 != info.voxels[neighbor].label) {
						Union(info.voxels, center, neighbor);
					}
				}
			}
		}
	}

	__global__ void Kernel_CompressLabels(ClusteringCacheInfo info)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= *info.numberOfOccupiedVoxelIndices) return;

		uint3 idx = info.occupiedVoxelIndices[threadid];
		unsigned int center = idx.z * info.cacheDimensions.y * info.cacheDimensions.x + idx.y * info.cacheDimensions.x + idx.x;
		if (0 == info.voxels[center].label) return;

		info.voxels[center].label = FindRoot(info.voxels, info.voxels[center].label);
	}

	void ClusteringCache::ConnectedComponentLabeling(unsigned int numberOfOccupiedVoxelIndices)
	{
		unsigned int blockSize = 256;
		unsigned int gridOccupied = (numberOfOccupiedVoxelIndices + blockSize - 1) / blockSize;

		Kernel_InterBlockMerge26Way << <gridOccupied, blockSize >> > (info);
		cudaDeviceSynchronize();

		Kernel_CompressLabels << <gridOccupied, blockSize >> > (info);
		cudaDeviceSynchronize();
	}
}


std::vector<unsigned int> cuMain(float voxelSize, const std::vector<float3>& host_points, const std::vector<float3>& host_normals, const std::vector<float3>& host_colors, float3 center)
{
	Clustering::ClusteringCacheInfo info;
	info.voxelSize = voxelSize;
	info.cacheDimensions = dim3(200, 300, 400);
	//info.cacheDimensions = dim3(400, 400, 400);
	info.numberOfVoxels = info.cacheDimensions.x * info.cacheDimensions.y * info.cacheDimensions.z;
	info.cacheMin = make_float3(
		center.x - (info.cacheDimensions.x * info.voxelSize) / 2.0f,
		center.y - (info.cacheDimensions.y * info.voxelSize) / 2.0f,
		center.z - (info.cacheDimensions.z * info.voxelSize) / 2.0f);

	Clustering::ClusteringCache cache(info);
	cache.Initialize();

	float* d_points;
	float* d_normals;
	float* d_colors;

	//cache.SetHostPoints(host_points);

	cudaMalloc(&d_points, sizeof(float) * 3 * host_points.size());
	cudaMemcpy(d_points, host_points.data(), sizeof(float) * 3 * host_points.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&d_normals, sizeof(float) * 3 * host_normals.size());
	cudaMemcpy(d_normals, host_normals.data(), sizeof(float) * 3 * host_normals.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&d_colors, sizeof(float) * 3 * host_colors.size());
	cudaMemcpy(d_colors, host_colors.data(), sizeof(float) * 3 * host_colors.size(), cudaMemcpyHostToDevice);

	cache.SetDevicePoints(d_points, d_normals, d_colors, (unsigned int)host_points.size());

	cache.RunClustering(10);
	std::vector<unsigned int> labels = cache.GetLabels();

	cache.Terminate();

	cudaFree(d_points);
	cudaFree(d_normals);
	cudaFree(d_colors);

	return labels;
}
