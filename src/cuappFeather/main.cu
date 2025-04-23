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

#ifdef Voxel CCL
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
		unsigned int numberOfVoxels = 200 * 300 * 400;
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
#endif // Voxel CCL




// 최적화된 Voxel Hash 기반 Connected Component Labeling (CCL) - GPU 전용 occupied index 추출 커널 포함

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define TABLE_SIZE 10485760
#define MAX_PROBE 32
#define BLOCK_SIZE 256

struct Voxel {
	int3 coord;
	unsigned int label;
	int occupied;
};

__device__ __host__ inline size_t voxel_hash(int3 coord) {
	return ((size_t)(coord.x * 73856093) ^ (coord.y * 19349663) ^ (coord.z * 83492791)) % TABLE_SIZE;
}

__device__ __forceinline__ unsigned int FindRoot(Voxel* voxels, unsigned int idx) {
	while (voxels[idx].label != idx) {
		voxels[idx].label = voxels[voxels[idx].label].label;
		idx = voxels[idx].label;
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

__global__ void insert_voxels(float3* points, int n, float voxel_size, Voxel* table, size_t table_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float3 p = points[idx];
	int3 coord = make_int3(floorf(p.x / voxel_size), floorf(p.y / voxel_size), floorf(p.z / voxel_size));

	size_t h = voxel_hash(coord);
	for (int i = 0; i < MAX_PROBE; ++i) {
		size_t slot = (h + i) % table_size;
		if (!atomicExch(&table[slot].occupied, true)) {
			table[slot].coord = coord;
			table[slot].label = slot;
			return;
		}
	}
}

__global__ void extract_occupied_indices(Voxel* table, unsigned int* indices, unsigned int* counter, size_t table_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= table_size) return;
	if (table[idx].occupied) {
		unsigned int offset = atomicAdd(counter, 1);
		indices[offset] = idx;
	}
}

__global__ void Kernel_InterVoxelHashMerge26Way(Voxel* table, unsigned int* occupiedIndices, size_t numOccupied, size_t table_size) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= numOccupied) return;

	int idx = occupiedIndices[tid];
	Voxel voxel = table[idx];
	if (!voxel.occupied) return;

	const int3 offsets[26] = {
		{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
		{1,1,0}, {1,-1,0}, {-1,1,0}, {-1,-1,0},
		{1,0,1}, {1,0,-1}, {-1,0,1}, {-1,0,-1},
		{0,1,1}, {0,1,-1}, {0,-1,1}, {0,-1,-1},
		{1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1},
		{-1,1,1}, {-1,1,-1}, {-1,-1,1}, {-1,-1,-1}
	};

#pragma unroll
	for (int i = 0; i < 26; ++i) {
		int3 neighborCoord = make_int3(
			voxel.coord.x + offsets[i].x,
			voxel.coord.y + offsets[i].y,
			voxel.coord.z + offsets[i].z);

		size_t h = voxel_hash(neighborCoord);

		for (int j = 0; j < MAX_PROBE; ++j) {
			size_t probe = (h + j) % table_size;
			if (!__ldg(&table[probe].occupied)) break;

			int3 coord = table[probe].coord;
			if (coord.x == neighborCoord.x && coord.y == neighborCoord.y && coord.z == neighborCoord.z) {
				Union(table, idx, probe);
				break;
			}
		}
	}
}

__global__ void Kernel_CompressVoxelHashLabels(Voxel* table, size_t table_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= table_size) return;

	Voxel& voxel = table[idx];
	if (!voxel.occupied) return;

	voxel.label = FindRoot(table, voxel.label);
}

__global__ void get_labels(float3* points, int n, float voxel_size, Voxel* table, size_t table_size, unsigned int* labels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float3 p = points[idx];
	int3 coord = make_int3(floorf(p.x / voxel_size), floorf(p.y / voxel_size), floorf(p.z / voxel_size));

	size_t h = voxel_hash(coord);
	for (int i = 0; i < MAX_PROBE; ++i) {
		size_t slot = (h + i) % table_size;
		Voxel voxel = table[slot];
		if (!voxel.occupied) break;
		if (voxel.coord.x == coord.x && voxel.coord.y == coord.y && voxel.coord.z == coord.z) {
			labels[idx] = voxel.label;
			return;
		}
	}
	labels[idx] = 0xFFFFFFFF;
}

std::vector<unsigned int> cuMain(
	float voxelSize,
	std::vector<float3>& host_points,
	std::vector<float3>& host_normals,
	std::vector<float3>& host_colors,
	float3 center)
{
	(void)host_normals;
	(void)host_colors;
	(void)center;

	std::vector<unsigned int> labels(host_points.size(), 0);

	Voxel* d_table;
	cudaMalloc(&d_table, sizeof(Voxel) * TABLE_SIZE);
	cudaMemset(d_table, 0, sizeof(Voxel) * TABLE_SIZE);

	float3* d_points;
	cudaMalloc(&d_points, sizeof(float3) * host_points.size());
	cudaMemcpy(d_points, host_points.data(), sizeof(float3) * host_points.size(), cudaMemcpyHostToDevice);

	unsigned int* d_labels;
	cudaMalloc(&d_labels, sizeof(unsigned int) * host_points.size());

	unsigned int* d_occupiedIndices;
	unsigned int* d_counter;
	cudaMalloc(&d_counter, sizeof(unsigned int));
	cudaMemset(d_counter, 0, sizeof(unsigned int));
	cudaMalloc(&d_occupiedIndices, sizeof(unsigned int) * TABLE_SIZE);

	nvtxRangePushA("CCL");

	int num_points = static_cast<int>(host_points.size());
	int num_blocks = (num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
	insert_voxels << <num_blocks, BLOCK_SIZE >> > (d_points, num_points, voxelSize, d_table, TABLE_SIZE);
	cudaDeviceSynchronize();

	int extractBlocks = (TABLE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	extract_occupied_indices << <extractBlocks, BLOCK_SIZE >> > (d_table, d_occupiedIndices, d_counter, TABLE_SIZE);
	cudaDeviceSynchronize();

	unsigned int numOccupied = 0;
	cudaMemcpy(&numOccupied, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	int occupiedBlocks = (numOccupied + BLOCK_SIZE - 1) / BLOCK_SIZE;
	Kernel_InterVoxelHashMerge26Way << <occupiedBlocks, BLOCK_SIZE >> > (d_table, d_occupiedIndices, numOccupied, TABLE_SIZE);
	cudaDeviceSynchronize();

	int tableBlocks = (TABLE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	Kernel_CompressVoxelHashLabels << <tableBlocks, BLOCK_SIZE >> > (d_table, TABLE_SIZE);
	cudaDeviceSynchronize();

	nvtxRangePop();

	get_labels << <num_blocks, BLOCK_SIZE >> > (d_points, num_points, voxelSize, d_table, TABLE_SIZE, d_labels);
	cudaDeviceSynchronize();

	cudaMemcpy(labels.data(), d_labels, sizeof(unsigned int) * host_points.size(), cudaMemcpyDeviceToHost);

	cudaFree(d_counter);
	cudaFree(d_points);
	cudaFree(d_labels);
	cudaFree(d_table);
	cudaFree(d_occupiedIndices);

	return labels;
}
