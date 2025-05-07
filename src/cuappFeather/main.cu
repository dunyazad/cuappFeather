#pragma warning(disable : 4819)

#include <glad/glad.h>

#include "main.cuh"

#include <nvtx3/nvToolsExt.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include "nvapi510/include/nvapi.h"
#include "nvapi510/include/NvApiDriverSettings.h"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include <Serialization.hpp>

#pragma comment(lib, "nvapi64.lib")


struct HashMapVoxel
{
    unsigned int label = 0;
    Eigen::Vector3f position = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    Eigen::Vector3b color = Eigen::Vector3b(255, 255, 255);
};

__device__ __host__ inline size_t voxel_hash(int3 coord, size_t tableSize)
{
    return ((size_t)(coord.x * 73856093) ^ (coord.y * 19349663) ^ (coord.z * 83492791)) % tableSize;
}

__global__ void Kernel_InsertPoints(Eigen::Vector3f* points, Eigen::Vector3f* normals, Eigen::Vector3b* colors, int numberOfPoints, float voxelSize, HashMapVoxel* table, size_t tableSize, unsigned int maxProbe)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberOfPoints) return;

    auto p = points[idx];
    auto n = normals[idx];
    auto c = colors[idx];

    int3 coord = make_int3(floorf(p.x() / voxelSize), floorf(p.y() / voxelSize), floorf(p.z() / voxelSize));

    size_t h = voxel_hash(coord, tableSize);
    for (int i = 0; i < maxProbe; ++i) {
        size_t slot = (h + i) % tableSize;
        if (atomicCAS(&table[slot].label, 0, slot) == 0)
        {
            //alog("%d, %d, %d\n", coord.x, coord.y, coord.z);

            table[slot].label = slot;
            table[slot].position = Eigen::Vector3f((float)coord.x * voxelSize, (float)coord.y * voxelSize, (float)coord.z * voxelSize);
            table[slot].normal = Eigen::Vector3f(n.x(), n.y(), n.z());
            table[slot].color = Eigen::Vector3b(c.x(), c.y(), c.z());
            return;
        }
    }
}

__global__ void Kernel_Serialize(HashMapVoxel* d_table, size_t tableSize,
    Eigen::Vector3f* d_points, Eigen::Vector3f* d_normals, Eigen::Vector3b* d_colors,
    unsigned int* numberOfOccupiedVoxels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tableSize) return;

    auto& voxel = d_table[idx];

    if (0 != voxel.label)
    {
        //alog("%f, %f, %f\n", voxel.position.x(), voxel.position.y(), voxel.position.z());

        auto oldIndex = atomicAdd(numberOfOccupiedVoxels, 1);
        d_points[oldIndex] = voxel.position;
        d_normals[oldIndex] = voxel.normal;
        d_colors[oldIndex] = voxel.color;
    }
}

struct HashMap
{
    size_t tableSize = 10485760;
    unsigned int maxProbe = 32;
    unsigned int blockSize = 256;

    HashMapVoxel* d_table = nullptr;

    void Initialize()
    {
        cudaMalloc(&d_table, sizeof(HashMapVoxel) * tableSize);
        cudaMemset(d_table, 0, sizeof(HashMapVoxel) * tableSize);
    }

    void Terminate()
    {
        cudaFree(d_table);
    }

    void InsertHPoints(Eigen::Vector3f* h_points, Eigen::Vector3f* h_normals, Eigen::Vector3b* h_colors, unsigned numberOfPoints)
    {
        Eigen::Vector3f* d_points = nullptr;
        Eigen::Vector3f* d_normals = nullptr;
        Eigen::Vector3b* d_colors = nullptr;

        cudaMalloc(&d_points, sizeof(Eigen::Vector3f) * numberOfPoints);
        cudaMemcpy(d_points, h_points, sizeof(Eigen::Vector3f) * numberOfPoints, cudaMemcpyHostToDevice);

        cudaMalloc(&d_normals, sizeof(Eigen::Vector3f) * numberOfPoints);
        cudaMemcpy(d_normals, h_normals, sizeof(Eigen::Vector3f) * numberOfPoints, cudaMemcpyHostToDevice);

        cudaMalloc(&d_colors, sizeof(Eigen::Vector3b) * numberOfPoints);
        cudaMemcpy(d_colors, d_colors, sizeof(Eigen::Vector3b) * numberOfPoints, cudaMemcpyHostToDevice);

        InsertDPoints(d_points, d_normals, d_colors, numberOfPoints);

        cudaFree(d_points);
        cudaFree(d_normals);
        cudaFree(d_colors);
    }

    void InsertDPoints(Eigen::Vector3f* d_points, Eigen::Vector3f* d_normals, Eigen::Vector3b* d_colors, unsigned numberOfPoints)
    {
        unsigned int blockSize = 256;
        unsigned int gridOccupied = (numberOfPoints + blockSize - 1) / blockSize;

        Kernel_InsertPoints << <gridOccupied, blockSize >> > (
            d_points,
            d_normals,
            d_colors,
            numberOfPoints,
            0.1f, d_table, tableSize, maxProbe);

        cudaDeviceSynchronize();
    }

    void Serialize(const std::string& filename)
    {
        PLYFormat ply;

        unsigned int* d_numberOfOccupiedVoxels = nullptr;
        cudaMalloc(&d_numberOfOccupiedVoxels, sizeof(unsigned int));

        Eigen::Vector3f* d_points = nullptr;
        Eigen::Vector3f* d_normals = nullptr;
        Eigen::Vector3b* d_colors = nullptr;

        cudaMalloc(&d_points, sizeof(Eigen::Vector3f) * tableSize);
        cudaMalloc(&d_normals, sizeof(Eigen::Vector3f) * tableSize);
        cudaMalloc(&d_colors, sizeof(Eigen::Vector3b) * tableSize);

        unsigned int blockSize = 256;
        unsigned int gridOccupied = (tableSize + blockSize - 1) / blockSize;

        Kernel_Serialize << <gridOccupied, blockSize >> > (
            d_table,
            tableSize,
            d_points,
            d_normals,
            d_colors,
            d_numberOfOccupiedVoxels);

        cudaDeviceSynchronize();

        unsigned int h_numberOfOccupiedVoxels = 0;
        cudaMemcpy(&h_numberOfOccupiedVoxels, d_numberOfOccupiedVoxels, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        Eigen::Vector3f* h_points  = new Eigen::Vector3f[h_numberOfOccupiedVoxels];
        Eigen::Vector3f* h_normals = new Eigen::Vector3f[h_numberOfOccupiedVoxels];
        Eigen::Vector3b* h_colors  = new Eigen::Vector3b[h_numberOfOccupiedVoxels];

        cudaMemcpy(h_points, d_points, sizeof(Eigen::Vector3f) * h_numberOfOccupiedVoxels, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_normals, d_normals, sizeof(Eigen::Vector3f) * h_numberOfOccupiedVoxels, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colors, d_colors, sizeof(Eigen::Vector3b) * h_numberOfOccupiedVoxels, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < h_numberOfOccupiedVoxels; i++)
        {
            auto& p = h_points[i];
            auto& n = h_normals[i];
            auto& c = h_colors[i];

            ply.AddPoint(p.x(), p.y(), p.z());
            ply.AddNormal(n.x(), n.y(), n.z());
            ply.AddColor(c.x() / 255.0f, c.y() / 255.0f, c.z() / 255.0f);
        }

        ply.Serialize(filename);

        cudaFree(d_numberOfOccupiedVoxels);
        cudaFree(d_points);
        cudaFree(d_normals);
        cudaFree(d_colors);

        delete[] h_points;
        delete[] h_normals;
        delete[] h_colors;
    }
};

__host__ __device__
float3 rgb_to_hsv(uchar3 rgb) {
    float r = rgb.x / 255.0f;
    float g = rgb.y / 255.0f;
    float b = rgb.z / 255.0f;

    float cmax = fmaxf(r, fmaxf(g, b));
    float cmin = fminf(r, fminf(g, b));
    float delta = cmax - cmin;

    float h = 0.0f;
    if (delta > 1e-6f) {
        if (cmax == r) {
            h = fmodf((g - b) / delta, 6.0f);
        }
        else if (cmax == g) {
            h = (b - r) / delta + 2.0f;
        }
        else {
            h = (r - g) / delta + 4.0f;
        }
        h *= 60.0f;
        if (h < 0.0f) h += 360.0f;
    }

    float s = (cmax == 0.0f) ? 0.0f : delta / cmax;
    float v = cmax;

    return make_float3(h, s, v); // H in degrees, S and V in [0,1]
}

__host__ __device__
uchar3 hsv_to_rgb(float3 hsv) {
    float h = hsv.x; // [0, 360)
    float s = hsv.y; // [0, 1]
    float v = hsv.z; // [0, 1]

    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float r, g, b;
    if (h < 60.0f) {
        r = c; g = x; b = 0;
    }
    else if (h < 120.0f) {
        r = x; g = c; b = 0;
    }
    else if (h < 180.0f) {
        r = 0; g = c; b = x;
    }
    else if (h < 240.0f) {
        r = 0; g = x; b = c;
    }
    else if (h < 300.0f) {
        r = x; g = 0; b = c;
    }
    else {
        r = c; g = 0; b = x;
    }

    uchar3 rgb;
    rgb.x = static_cast<unsigned char>((r + m) * 255.0f);
    rgb.y = static_cast<unsigned char>((g + m) * 255.0f);
    rgb.z = static_cast<unsigned char>((b + m) * 255.0f);
    return rgb;
}

struct PointCloud
{
    Eigen::Vector3f* d_points = nullptr;
    Eigen::Vector3f* d_normals = nullptr;
    Eigen::Vector3b* d_colors = nullptr;
    unsigned int numberOfPoints = 0;
};

PointCloud pointCloud;

__global__ void Kernel_DetectEdge(
    const Eigen::Vector3f* d_points,
    const Eigen::Vector3f* d_normals,
    const Eigen::Vector3b* d_colors,
    uint8_t* d_is_edge,
    int numberOfPoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberOfPoints) return;

    Eigen::Vector3f pi = d_points[idx];
    Eigen::Vector3f ni = d_normals[idx];
    Eigen::Vector3b ci = d_colors[idx];

    float3 hsv_i = rgb_to_hsv(make_uchar3(ci.x(), ci.y(), ci.z()));

    int edge_count = 0;
    int total_neighbors = 0;

    for (int j = 0; j < numberOfPoints; ++j) {
        if (j == idx) continue;

        Eigen::Vector3f pj = d_points[j];
        if ((pj - pi).squaredNorm() > 0.01f) continue;

        Eigen::Vector3f nj = d_normals[j];
        Eigen::Vector3b cj = d_colors[j];
        float3 hsv_j = rgb_to_hsv(make_uchar3(cj.x(), cj.y(), cj.z()));

        float angle = acosf(fminf(fmaxf(ni.dot(nj), -1.0f), 1.0f));
        float h_diff = fmodf(fabsf(hsv_i.x - hsv_j.x), 360.0f);
        h_diff = fminf(h_diff, 360.0f - h_diff);

        if (angle > 0.3f || h_diff > 20.0f) {
            edge_count++;
        }

        total_neighbors++;
    }

    d_is_edge[idx] = (edge_count >= 2);
}

vector<uint8_t> DetectEdge()
{
    vector<uint8_t> h_is_edge(pointCloud.numberOfPoints);
    uint8_t* d_is_edge = nullptr;
    cudaMalloc(&d_is_edge, sizeof(uint8_t) * pointCloud.numberOfPoints);

    unsigned int blockSize = 256;
    unsigned int gridOccupied = (pointCloud.numberOfPoints + blockSize - 1) / blockSize;

    Kernel_DetectEdge << <gridOccupied, blockSize >> > (
        pointCloud.d_points,
        pointCloud.d_normals,
        pointCloud.d_colors,
        d_is_edge,
        pointCloud.numberOfPoints);

    cudaDeviceSynchronize();

    cudaMemcpy(h_is_edge.data(), d_is_edge, sizeof(uint8_t) * pointCloud.numberOfPoints, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    return h_is_edge;
}

void cuMain(
	float voxelSize,
	std::vector<float3>& host_points,
	std::vector<float3>& host_normals,
	std::vector<uchar3>& host_colors,
	float3 center)
{
    pointCloud.numberOfPoints = host_points.size();

    cudaMalloc(&pointCloud.d_points, sizeof(Eigen::Vector3f) * pointCloud.numberOfPoints);
    cudaMalloc(&pointCloud.d_normals, sizeof(Eigen::Vector3f) * pointCloud.numberOfPoints);
    cudaMalloc(&pointCloud.d_colors, sizeof(Eigen::Vector3b) * pointCloud.numberOfPoints);

    cudaMemcpy(pointCloud.d_points, host_points.data(), sizeof(Eigen::Vector3f) * pointCloud.numberOfPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(pointCloud.d_normals, host_normals.data(), sizeof(Eigen::Vector3f) * pointCloud.numberOfPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(pointCloud.d_colors, host_colors.data(), sizeof(Eigen::Vector3b) * pointCloud.numberOfPoints, cudaMemcpyHostToDevice);

    HashMap hm;
    hm.Initialize();

    hm.InsertDPoints(pointCloud.d_points, pointCloud.d_normals, pointCloud.d_colors, pointCloud.numberOfPoints);

    hm.Serialize("C:\\Debug\\PLY\\Set\\Voxels.ply");

    hm.Terminate();
}

bool ForceGPUPerformance()
{
    NvAPI_Status status;

    status = NvAPI_Initialize();
    if (status != NVAPI_OK)
    {
        return false;
    }

    NvDRSSessionHandle hSession = 0;
    status = NvAPI_DRS_CreateSession(&hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    // (2) load all the system settings into the session
    status = NvAPI_DRS_LoadSettings(hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    NvDRSProfileHandle hProfile = 0;
    status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
    if (status != NVAPI_OK)
    {
        return false;
    }

    NVDRS_SETTING drsGet = { 0, };
    drsGet.version = NVDRS_SETTING_VER;
    status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
    if (status != NVAPI_OK)
    {
        return false;
    }
    auto m_gpu_performance = drsGet.u32CurrentValue;

    NVDRS_SETTING drsSetting = { 0, };
    drsSetting.version = NVDRS_SETTING_VER;
    drsSetting.settingId = PREFERRED_PSTATE_ID;
    drsSetting.settingType = NVDRS_DWORD_TYPE;
    drsSetting.u32CurrentValue = PREFERRED_PSTATE_PREFER_MAX;

    status = NvAPI_DRS_SetSetting(hSession, hProfile, &drsSetting);
    if (status != NVAPI_OK)
    {
        return false;
    }

    status = NvAPI_DRS_SaveSettings(hSession);
    if (status != NVAPI_OK)
    {
        return false;
    }

    // (6) We clean up. This is analogous to doing a free()
    NvAPI_DRS_DestroySession(hSession);
    hSession = 0;

    return true;
}

#pragma region Print GPU Performance Mode
//{
//	NvAPI_Status status;

//	status = NvAPI_Initialize();
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_Initialize() status != NVAPI_OK\n");
//		return;
//	}

//	NvDRSSessionHandle hSession = 0;
//	status = NvAPI_DRS_CreateSession(&hSession);
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_DRS_CreateSession() status != NVAPI_OK\n");
//		return;
//	}

//	// (2) load all the system settings into the session
//	status = NvAPI_DRS_LoadSettings(hSession);
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_DRS_LoadSettings() status != NVAPI_OK\n");
//		return;
//	}

//	NvDRSProfileHandle hProfile = 0;
//	status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_DRS_GetBaseProfile() status != NVAPI_OK\n");
//		return;
//	}

//	NVDRS_SETTING drsGet = { 0, };
//	drsGet.version = NVDRS_SETTING_VER;
//	status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
//	if (status != NVAPI_OK)
//	{
//		printf("NvAPI_DRS_GetSetting() status != NVAPI_OK\n");
//		return;
//	}

//	auto gpu_performance = drsGet.u32CurrentValue;

//	printf("gpu_performance : %d\n", gpu_performance);

//	// (6) We clean up. This is analogous to doing a free()
//	NvAPI_DRS_DestroySession(hSession);
//	hSession = 0;
//}
#pragma endregion
