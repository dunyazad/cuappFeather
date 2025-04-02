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
		//float3 position;
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

		void SetHostPoints(const std::vector<float3>& hostPoints);
		void SetDevicePoints(float* devicePoints, unsigned int numberOfPoints);

		void RunClustering(int iterations = 1);
		std::vector<unsigned int> GetLabels();

		void ConnectedComponentLabeling(
			Voxel* d_voxels,
			uint3* occupiedVoxelIndices,
			unsigned int numberOfOccupiedVoxelIndices,
			dim3 volumeDimensions);

	private:
		ClusteringCacheInfo info;

		float* d_points = nullptr;
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

		if (d_points)
			cudaFree(d_points);

		if (occupiedPointIndices)
			cudaFree(occupiedPointIndices);

		if (numberOfOccupiedPointIndices)
			cudaFree(numberOfOccupiedPointIndices);
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
		auto& voxel = info.voxels[cacheIndex];

		//voxel.position.x = cacheMin.x + ix * voxelSize;
		//voxel.position.y = cacheMin.y + iy * voxelSize;
		//voxel.position.z = cacheMin.z + iz * voxelSize;
		voxel.label = cacheIndex;

		//alog("%f, %f, %f\n", voxel.position.x, voxel.position.y, voxel.position.z);

		auto voxelIndex = atomicAdd(info.numberOfOccupiedVoxelIndices, 1);
		info.occupiedVoxelIndices[voxelIndex] = make_uint3(ix, iy, iz);

		auto pointIndex = atomicAdd(numberOfOccupiedPointIndices, 1);
		occupiedPointIndices[pointIndex] = threadid;

		//alog("%d\n", index);
	}

	void ClusteringCache::OccupyVoxels()
	{
		nvtxRangePush("OccupyVoxels");

		unsigned int blockSize = 256;
		unsigned int gridSize = (numberOfPoints + blockSize - 1) / blockSize;

		Kernel_OccupyVoxels << <gridSize, blockSize >> > (
			info,
			d_points,
			numberOfPoints,
			occupiedPointIndices,
			numberOfOccupiedPointIndices);

		cudaDeviceSynchronize();
		nvtxRangePop();
	}

	void ClusteringCache::SetHostPoints(const std::vector<float3>& hostPoints)
	{
		numberOfPoints = static_cast<unsigned int>(hostPoints.size());
		
		cudaMalloc(&d_points, sizeof(float3) * numberOfPoints);

		cudaMalloc(&occupiedPointIndices, sizeof(unsigned int) * numberOfPoints);
		cudaMalloc(&numberOfOccupiedPointIndices, sizeof(unsigned int));
		cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));

		cudaMemcpy(d_points, hostPoints.data(), sizeof(float3) * numberOfPoints, cudaMemcpyHostToDevice);
	}

	void ClusteringCache::SetDevicePoints(float* devicePoints, unsigned int numberOfPoints)
	{
		this->numberOfPoints = numberOfPoints;
		d_points = devicePoints;

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

			ConnectedComponentLabeling(info.voxels, info.occupiedVoxelIndices, h_occupiedCount, info.cacheDimensions);

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

	__global__ void Kernel_InterBlockMerge26Way(
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

	__global__ void Kernel_InterBlockMerge6Way(
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

		const int3 neighbors[6] = {
		{  1,  0,  0 },
		{ -1,  0,  0 },
		{  0,  1,  0 },
		{  0, -1,  0 },
		{  0,  0,  1 },
		{  0,  0, -1 }
		};

		for (int i = 0; i < 6; ++i) {
			int nx = idx.x + neighbors[i].x;
			int ny = idx.y + neighbors[i].y;
			int nz = idx.z + neighbors[i].z;

			if (nx < 0 || ny < 0 || nz < 0 || nx >= dims.x || ny >= dims.y || nz >= dims.z)
				continue;

			unsigned int neighbor = nz * dims.y * dims.x + ny * dims.x + nx;
			if (0 != voxels[neighbor].label) {
				Union(voxels, center, neighbor);
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

	void ClusteringCache::ConnectedComponentLabeling(
		Voxel* d_voxels,
		uint3* occupiedVoxelIndices,
		unsigned int numberOfOccupiedVoxelIndices,
		dim3 volumeDimensions)
	{
		unsigned int totalVoxels = volumeDimensions.x * volumeDimensions.y * volumeDimensions.z;
		unsigned int blockSize = 256;
		unsigned int gridVoxels = (totalVoxels + blockSize - 1) / blockSize;
		unsigned int gridOccupied = (numberOfOccupiedVoxelIndices + blockSize - 1) / blockSize;

		Kernel_InterBlockMerge26Way << <gridOccupied, blockSize >> > (d_voxels, occupiedVoxelIndices, numberOfOccupiedVoxelIndices, volumeDimensions);
		cudaDeviceSynchronize();

		Kernel_CompressLabels << <gridVoxels, blockSize >> > (d_voxels, totalVoxels);
		cudaDeviceSynchronize();
	}
}


std::vector<unsigned int> cuMain(const std::vector<float3>& host_points, float3 center)
{
	Clustering::ClusteringCacheInfo info;
	info.voxelSize = 0.1f;
	info.cacheDimensions = dim3(200, 300, 400);
	info.numberOfVoxels = info.cacheDimensions.x * info.cacheDimensions.y * info.cacheDimensions.z;
	info.cacheMin = make_float3(
		center.x - (info.cacheDimensions.x * info.voxelSize) / 2.0f,
		center.y - (info.cacheDimensions.y * info.voxelSize) / 2.0f,
		center.z - (info.cacheDimensions.z * info.voxelSize) / 2.0f);

	Clustering::ClusteringCache cache(info);
	cache.Initialize();

	float* d_points;

	//cache.SetHostPoints(host_points);

	cudaMalloc(&d_points, sizeof(float) * 3 * host_points.size());
	cudaMemcpy(d_points, host_points.data(), sizeof(float) * 3 * host_points.size(), cudaMemcpyHostToDevice);
	cache.SetDevicePoints(d_points, host_points.size());

	cache.RunClustering(10);
	std::vector<unsigned int> labels = cache.GetLabels();
	cache.Terminate();

	cudaFree(d_points);

	return labels;
}