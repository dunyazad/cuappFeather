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
	__global__ void Kernel_ClearVoxels(
		unsigned int* d_voxels,
		unsigned int numberOfVoxels,
		dim3 volumeDimensions,
		float voxelSize,
		float3 volumeMin,
		float3 volumeCenter)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= volumeDimensions.x * volumeDimensions.y * volumeDimensions.z) return;

		//d_voxels[threadid].position = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		d_voxels[threadid] = 0;
	}

	void ClearVoxels(
		unsigned int* d_voxels,
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
		unsigned int* d_voxels,
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
		voxel = volumeIndex;

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
		unsigned int* d_voxels,
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

	__global__ void Kernel_GetLabels(
		float* d_points,
		unsigned int numberOfPoints,
		unsigned int* d_voxels,
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

		d_labels[threadid] = voxel;
	}

	std::vector<unsigned int> GetLabels(
		float* d_points,
		unsigned int numberOfPoints,
		unsigned int* d_voxels,
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

	struct ClusteringCacheInfo
	{
		float voxelSize = 0.1f;
		dim3 cacheDimensions = dim3(200, 300, 400);
		unsigned int numberOfVoxels;
		float3 cacheMin = make_float3(0.0f, 0.0f, 0.0f);

		cudaArray* cacheData3D = nullptr;
		cudaSurfaceObject_t surfaceObject3D;

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

		void UploadPoints(const std::vector<float3>& hostPoints);
		void RunClustering(int iterations = 1);
		std::vector<unsigned int> DownloadLabels();

		void ConnectedComponentLabeling(
			unsigned int* d_voxels,
			uint3* occupiedVoxelIndices,
			unsigned int numberOfOccupiedVoxelIndices,
			dim3 volumeDimensions);

	private:
		void AllocateDeviceMemory(size_t numPoints);
		void FreeDeviceMemory();

	private:
		ClusteringCacheInfo info;

		float* d_points = nullptr;
		unsigned int numberOfPoints = 0;

		unsigned int* d_voxels = nullptr;

		unsigned int* occupiedPointIndices = nullptr;
		unsigned int* numberOfOccupiedPointIndices = nullptr;

		unsigned int* d_labels = nullptr;
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
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaExtent extent = make_cudaExtent(info.cacheDimensions.x, info.cacheDimensions.y, info.cacheDimensions.z);

		cudaMalloc3DArray(&info.cacheData3D, &channelDesc, extent, cudaArraySurfaceLoadStore);

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = info.cacheData3D;

		cudaCreateSurfaceObject(&info.surfaceObject3D, &resDesc);

		// Allocate buffers for occupied voxel indices
		size_t maxVoxels = info.numberOfVoxels;
		cudaMalloc(&info.occupiedVoxelIndices, maxVoxels * sizeof(uint3));
		cudaMalloc(&info.numberOfOccupiedVoxelIndices, sizeof(unsigned int));
		cudaMemset(info.numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));
	}

	void ClusteringCache::Terminate()
	{
		if (info.surfaceObject3D)
			cudaDestroySurfaceObject(info.surfaceObject3D);

		if (info.cacheData3D)
			cudaFreeArray(info.cacheData3D);

		if (info.occupiedVoxelIndices)
			cudaFree(info.occupiedVoxelIndices);

		if (info.numberOfOccupiedVoxelIndices)
			cudaFree(info.numberOfOccupiedVoxelIndices);
	}

	void ClusteringCache::UploadPoints(const std::vector<float3>& hostPoints)
	{
		numberOfPoints = static_cast<unsigned int>(hostPoints.size());
		AllocateDeviceMemory(numberOfPoints);

		cudaMemcpy(d_points, hostPoints.data(), sizeof(float3) * numberOfPoints, cudaMemcpyHostToDevice);
		cudaMemset(info.numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));
		cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));
	}

	void ClusteringCache::RunClustering(int iterations)
	{
		using namespace Clustering;

		float3 volumeCenter = make_float3(
			info.cacheMin.x + (info.cacheDimensions.x * info.voxelSize) / 2.0f,
			info.cacheMin.y + (info.cacheDimensions.y * info.voxelSize) / 2.0f,
			info.cacheMin.z + (info.cacheDimensions.z * info.voxelSize) / 2.0f);

		for (int i = 0; i < iterations; ++i)
		{
			cudaMemset(info.numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));
			cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));

			ClearVoxels(d_voxels, info.numberOfVoxels, info.cacheDimensions, info.voxelSize, info.cacheMin, volumeCenter);

			OccupyVoxels(
				d_points,
				numberOfPoints,
				d_voxels,
				info.numberOfVoxels,
				info.cacheDimensions,
				info.voxelSize,
				info.cacheMin,
				volumeCenter,
				info.occupiedVoxelIndices,
				info.numberOfOccupiedVoxelIndices,
				occupiedPointIndices,
				numberOfOccupiedPointIndices);

			unsigned int h_occupiedCount = 0;
			cudaMemcpy(&h_occupiedCount, info.numberOfOccupiedVoxelIndices, sizeof(unsigned int), cudaMemcpyDeviceToHost);

			nvtxRangePushA("CCL");

			ConnectedComponentLabeling(d_voxels, info.occupiedVoxelIndices, h_occupiedCount, info.cacheDimensions);

			nvtxRangePop();
		}
	}

	std::vector<unsigned int> ClusteringCache::DownloadLabels()
	{
		float3 center = make_float3(
			info.cacheMin.x + (info.cacheDimensions.x * info.voxelSize) / 2.0f,
			info.cacheMin.y + (info.cacheDimensions.y * info.voxelSize) / 2.0f,
			info.cacheMin.z + (info.cacheDimensions.z * info.voxelSize) / 2.0f);

		return Clustering::GetLabels(
			d_points,
			numberOfPoints,
			d_voxels,
			info.numberOfVoxels,
			info.cacheDimensions,
			info.voxelSize,
			info.cacheMin,
			center);
	}

	void ClusteringCache::AllocateDeviceMemory(size_t numPoints)
	{
		cudaMalloc(&d_points, sizeof(float3) * numPoints);
		cudaMalloc(&d_voxels, sizeof(unsigned int) * info.numberOfVoxels);

		cudaMalloc(&occupiedPointIndices, sizeof(unsigned int) * numPoints);
		cudaMalloc(&numberOfOccupiedPointIndices, sizeof(unsigned int));
		cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));
	}

	void ClusteringCache::FreeDeviceMemory()
	{
		if (d_points) cudaFree(d_points);
		if (d_voxels) cudaFree(d_voxels);
		if (occupiedPointIndices) cudaFree(occupiedPointIndices);
		if (numberOfOccupiedPointIndices) cudaFree(numberOfOccupiedPointIndices);
	}

	__device__ __forceinline__ unsigned int FindRoot(unsigned int* voxels, unsigned int idx) {
		while (true) {
			unsigned int parent = voxels[idx];
			unsigned int grand = voxels[parent];
			if (parent == idx) break;
			if (parent != grand) voxels[idx] = grand;
			idx = parent;
		}
		return idx;
	}

	__device__ __forceinline__ void Union(unsigned int* voxels, unsigned int a, unsigned int b) {
		unsigned int rootA = FindRoot(voxels, a);
		unsigned int rootB = FindRoot(voxels, b);
		if (rootA != rootB) {
			if (rootA < rootB)
				atomicMin(&voxels[rootB], rootA);
			else
				atomicMin(&voxels[rootA], rootB);
		}
	}

	__global__ void Kernel_InterBlockMerge26Way(
		unsigned int* voxels,
		uint3* occupiedIndices,
		unsigned int numOccupied,
		dim3 dims)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= numOccupied) return;

		uint3 idx = occupiedIndices[tid];
		unsigned int center = idx.z * dims.y * dims.x + idx.y * dims.x + idx.x;
		//if (voxels[center].position.x == FLT_MAX) return;
		if (0 == voxels[center]) return;

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
					if (0 != voxels[neighbor]) {
						Union(voxels, center, neighbor);
					}
				}
			}
		}
	}

	__global__ void Kernel_InterBlockMerge6Way(
		unsigned int* voxels,
		uint3* occupiedIndices,
		unsigned int numOccupied,
		dim3 dims)
	{
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= numOccupied) return;

		uint3 idx = occupiedIndices[tid];
		unsigned int center = idx.z * dims.y * dims.x + idx.y * dims.x + idx.x;
		//if (voxels[center].position.x == FLT_MAX) return;
		if (0 == voxels[center]) return;

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
			if (0 != voxels[neighbor]) {
				Union(voxels, center, neighbor);
			}
		}
	}

	__global__ void Kernel_CompressLabels(unsigned int* voxels, unsigned int N) {
		unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= N) return;

		//if (voxels[tid].position.x != FLT_MAX) {
		if (0 != voxels[tid]) {
			voxels[tid] = FindRoot(voxels, tid);
		}
	}

	void ClusteringCache::ConnectedComponentLabeling(
		unsigned int* d_voxels,
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

		//Kernel_InterBlockMergeP << <gridOccupied, blockSize >> > (d_voxels, occupiedVoxelIndices, numberOfOccupiedVoxelIndices, volumeDimensions);
		//cudaDeviceSynchronize();

		//Kernel_InterBlockMergeN << <gridOccupied, blockSize >> > (d_voxels, occupiedVoxelIndices, numberOfOccupiedVoxelIndices, volumeDimensions);
		//cudaDeviceSynchronize();

		Kernel_CompressLabels << <gridVoxels, blockSize >> > (d_voxels, totalVoxels);
		cudaDeviceSynchronize();
	}
}


std::vector<unsigned int> cuMain(const std::vector<float3>& host_points, float3 center)
{
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
		cache.UploadPoints(host_points);
		cache.RunClustering(10);
		std::vector<unsigned int> labels = cache.DownloadLabels();
		cache.Terminate();

		return labels;
	}

	//nvtxRangePush("TestClustering");

	//float* d_points = nullptr;
	//cudaMalloc(&d_points, sizeof(float) * host_points.size() * 3);
	//cudaMemcpy(d_points, host_points.data(), sizeof(float) * host_points.size() * 3, cudaMemcpyHostToDevice);

	//unsigned int numberOfPoints = host_points.size();
	//dim3 volumeDimensions(200, 300, 400);
	////dim3 volumeDimensions(400, 400, 400);
	//unsigned int numberOfVoxels = volumeDimensions.x * volumeDimensions.y * volumeDimensions.z;
	//float voxelSize = 0.1f;
	////float3 volumeCenter = make_float3(3.9904f, -15.8357f, -7.2774f);
	////float3 volumeCenter = make_float3(4.0f, -15.0f, -7.0f);
	//float3 volumeCenter = make_float3(center.x, center.y, center.z);
	//float3 volumeMin = make_float3(
	//	volumeCenter.x - (float)(volumeDimensions.x / 2) * voxelSize,
	//	volumeCenter.y - (float)(volumeDimensions.y / 2) * voxelSize,
	//	volumeCenter.z - (float)(volumeDimensions.z / 2) * voxelSize);

	//Clustering::Voxel* d_voxels = nullptr;
	//cudaMalloc(&d_voxels, sizeof(Clustering::Voxel) * numberOfVoxels);

	//uint3* occupiedVoxelIndices = nullptr;
	//cudaMalloc(&occupiedVoxelIndices, sizeof(uint3) * 5000000);
	//unsigned int* numberOfOccupiedVoxelIndices = nullptr;
	//cudaMalloc(&numberOfOccupiedVoxelIndices, sizeof(unsigned int));
	//cudaMemset(numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));

	//unsigned int* occupiedPointIndices = nullptr;
	//cudaMalloc(&occupiedPointIndices, sizeof(unsigned int) * 5000000);
	//unsigned int* numberOfOccupiedPointIndices = nullptr;
	//cudaMalloc(&numberOfOccupiedPointIndices, sizeof(unsigned int));
	//cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));

	//cudaDeviceSynchronize();

	//for (size_t i = 0; i < 10; i++)
	//{
	//	cudaMemset(numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));
	//	cudaMemset(numberOfOccupiedPointIndices, 0, sizeof(unsigned int));

	//	ClearVoxels(d_voxels, numberOfVoxels, volumeDimensions, voxelSize, volumeMin, volumeCenter);

	//	OccupyVoxels(
	//		d_points,
	//		numberOfPoints,
	//		d_voxels,
	//		numberOfVoxels,
	//		volumeDimensions,
	//		voxelSize,
	//		volumeMin,
	//		volumeCenter,
	//		occupiedVoxelIndices,
	//		numberOfOccupiedVoxelIndices,
	//		occupiedPointIndices,
	//		numberOfOccupiedPointIndices);

	//	unsigned int h_numberOfOccupiedVoxelIndices = 0;
	//	cudaMemcpy(&h_numberOfOccupiedVoxelIndices, numberOfOccupiedVoxelIndices, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//	nvtxRangePushA("CCL");

	//	ConnectedComponentLabeling(d_voxels, occupiedVoxelIndices, h_numberOfOccupiedVoxelIndices, volumeDimensions);

	//	nvtxRangePop();
	//}
	////VisualizeVoxels(
	////	d_voxels,
	////	numberOfVoxels,
	////	volumeDimensions,
	////	voxelSize,
	////	volumeMin);

	//std::vector<unsigned int> result = GetLabels(
	//	d_points,
	//	numberOfPoints,
	//	d_voxels,
	//	numberOfVoxels,
	//	volumeDimensions,
	//	voxelSize,
	//	volumeMin,
	//	volumeCenter);

	//{
	//	std::unordered_map<unsigned int, unsigned int> labelHistogram;

	//	for (auto& i : result)
	//	{
	//		if (0 == labelHistogram.count(i))
	//		{
	//			labelHistogram[i] = 1;
	//		}
	//		else
	//		{
	//			labelHistogram[i] += 1;
	//		}
	//	}

	//	unsigned int i = 0;
	//	for (auto& [label, count] : labelHistogram)
	//	{
	//		alog("[%4d] point label - %16d : count - %8d\n", i++, label, count);
	//	}
	//	alog("\n");
	//}

	//unsigned int h_numberOfOccupiedPointIndices = 0;
	//cudaMemcpy(&h_numberOfOccupiedPointIndices, numberOfOccupiedPointIndices, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//unsigned int* h_occupiedPointIndices = new unsigned int[h_numberOfOccupiedPointIndices];
	//cudaMemcpy(h_occupiedPointIndices, occupiedPointIndices, sizeof(unsigned int) * h_numberOfOccupiedPointIndices, cudaMemcpyDeviceToHost);

	//for (size_t i = 0; i < h_numberOfOccupiedPointIndices; i++)
	//{
	//	auto index = h_occupiedPointIndices[i];
	//	auto p = host_points[index];

	//	//VD::AddSphere("In Area", { x,y, z }, 0.05f, { 255, 0, 0 });
	//}

	//cudaFree(d_points);
	//cudaFree(d_voxels);
	//cudaFree(occupiedVoxelIndices);
	//cudaFree(numberOfOccupiedVoxelIndices);
	//cudaFree(occupiedPointIndices);
	//cudaFree(numberOfOccupiedPointIndices);

	//delete[] h_occupiedPointIndices;

	//cudaDeviceSynchronize();
	//nvtxRangePop();

	//return result;
}