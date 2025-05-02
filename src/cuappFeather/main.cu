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

struct PointCloud
{
    Eigen::Vector3f* d_points = nullptr;
    Eigen::Vector3f* d_normals = nullptr;
    Eigen::Vector3b* d_colors = nullptr;
    unsigned int numberOfPoints = 0;
};

PointCloud pointCloud;

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

cudaSurfaceObject_t surfaceObject = 0;
cudaGraphicsResource* cudaResource = nullptr;

__global__ void fillTextureKernel(cudaSurfaceObject_t surface, int width, int height, int xOffset, int yOffset)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	uchar4 color;
	color.x = (x + xOffset) % 256;
	color.y = (y + yOffset) % 256;
	color.z = 128;
	color.w = 255;

	surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

void GenerateCUDATexture(unsigned int textureID, unsigned int width, unsigned int height, unsigned int xOffset, int yOffset)
{
    //cudaGLSetGLDevice(0);

    // (1) Register OpenGL texture to CUDA
    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsGLRegisterImage failed: " << cudaGetErrorString(err) << std::endl;
        return; // <<< 여기서 끝내야 합니다
    }

    // (2) Map Resources
    err = cudaGraphicsMapResources(1, &cudaResource, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // (3) Get mapped array
    cudaArray* textureArray = nullptr;
    err = cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaResource, 0, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // (4) Create surface object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    
    cudaCreateSurfaceObject(&surfaceObject, &resDesc);

    // (5) Kernel Launch
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    fillTextureKernel << <gridSize, blockSize >> > (surfaceObject, width, height, xOffset, yOffset);
    cudaDeviceSynchronize();

    // (6) Clean up
    //cudaDestroySurfaceObject(surfaceObject);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    cudaDeviceSynchronize();
}

void UpdateCUDATexture(unsigned int textureID, unsigned int width, unsigned int height, Eigen::Matrix4f projectionMatrix, Eigen::Matrix4f viewMatrix)
{
    //return;

    if (cudaResource == nullptr) return;

    //if (false == tick) return;

    nvtxRangePushA("UpdateCUDATexture");

    //// (1) Register OpenGL texture to CUDA
    //cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    //if (err != cudaSuccess)
    //{
    //    std::cerr << "cudaGraphicsGLRegisterImage failed: " << cudaGetErrorString(err) << std::endl;
    //    return; // <<< 여기서 끝내야 합니다
    //}

     // (2) Map Resources
    auto err = cudaGraphicsMapResources(1, &cudaResource, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // (3) Get mapped array
    cudaArray* textureArray = nullptr;
    err = cudaGraphicsSubResourceGetMappedArray(&textureArray, cudaResource, 0, 0);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // (4) Create surface object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = textureArray;

    cudaSurfaceObject_t surfaceObject = 0; // <-- 여기가 로컬 변수
    cudaCreateSurfaceObject(&surfaceObject, &resDesc);

    ClearTexture(3840, 2160);
    RenderPointCloud(textureID, width, height, projectionMatrix, viewMatrix);

    //// (5) Kernel Launch
    //dim3 blockSize(32, 32);
    //dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    //fillTextureKernel << <gridSize, blockSize >> > (surfaceObject, width, height, xOffset, yOffset);
    //cudaDeviceSynchronize();

    //CallFillTextureKernel(width, height, xOffset, yOffset);

    // (6) Clean up
    //cudaDestroySurfaceObject(surfaceObject);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    cudaDeviceSynchronize();

    //  업데이트 후 Mipmap 갱신
    glBindTexture(GL_TEXTURE_2D, textureID);
    glGenerateMipmap(GL_TEXTURE_2D);
    //glBindTexture(GL_TEXTURE_2D, 0);

    glFinish();

    nvtxRangePop();
}

void CallFillTextureKernel(unsigned int width, unsigned int height, unsigned int xOffset, int yOffset)
{
    //if (true == tick) return;

    nvtxRangePushA("CallFillTextureKernel");
    

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    fillTextureKernel << <gridSize, blockSize >> > (surfaceObject, width, height, xOffset, yOffset);
    cudaDeviceSynchronize();

    nvtxRangePop();
}

__global__ void Kernel_ClearTexture(cudaSurfaceObject_t surface, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uchar4 color;
    color.x = 0;
    color.y = 0;
    color.z = 0;
    color.w = 255;

    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

void ClearTexture(unsigned int width, unsigned int height)
{
    //if (true == tick) return;

    nvtxRangePushA("ClearTexture");


    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    Kernel_ClearTexture << <gridSize, blockSize >> > (surfaceObject, width, height);
    cudaDeviceSynchronize();

    nvtxRangePop();
}

__global__ void Kernel_RenderPointCloud(cudaSurfaceObject_t surface, int width, int height, Eigen::Matrix4f projectionMatrix, Eigen::Matrix4f viewMatrix, PointCloud pointCloud)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= pointCloud.numberOfPoints) return;

    Eigen::Vector3f gp = pointCloud.d_points[threadid];
    //Eigen::Vector3f lp = (projectionMatrix * viewMatrix * Eigen::Vector4f(gp.x() * 10.0f, gp.y() * 10.0f, gp.z() * 10.0f, 1.0f)).head<3>();
    Eigen::Vector3f lp = (viewMatrix * Eigen::Vector4f(gp.x() * 100.0f, gp.y() * 100.0f, gp.z() * 100.0f, 1.0f)).head<3>();
    Eigen::Vector3f gn = pointCloud.d_normals[threadid];
    Eigen::Vector3b gc = pointCloud.d_colors[threadid];

    Eigen::Vector3f tp = Eigen::Vector3f(lp.x() + (float)width * 0.5f, lp.y() + (float)height * 0.5f, lp.z());
    unsigned int ix = floorf(tp.x());
    unsigned int iy = floorf(tp.y());

    if (ix >= width || iy >= height) return;

    uchar4 color;
    color.x = gc.x();
    color.y = gc.y();
    color.z = gc.z();
    color.w = 255;

    surf2Dwrite(color, surface, ix * sizeof(uchar4), iy);
}

void RenderPointCloud(unsigned int textureID, unsigned int width, unsigned int height, Eigen::Matrix4f projectionMatrix, Eigen::Matrix4f viewMatrix)
{
    //if (true == tick) return;

    nvtxRangePushA("RenderPointCloud");


    unsigned int blockSize = 512;
    unsigned int gridSize = (pointCloud.numberOfPoints + blockSize - 1) / blockSize;
    Kernel_RenderPointCloud << <gridSize, blockSize >> > (surfaceObject, width, height, projectionMatrix, viewMatrix, pointCloud);
    cudaDeviceSynchronize();

    nvtxRangePop();
}
