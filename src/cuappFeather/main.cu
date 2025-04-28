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

#include <Eigen/Core>
#include <Eigen/Dense>

#pragma comment(lib, "nvapi64.lib")

Eigen::Vector3f d_points;
Eigen::Vector3f d_normals;
Eigen::Vector3f d_colors;

void cuMain(
	float voxelSize,
	std::vector<float3>& host_points,
	std::vector<float3>& host_normals,
	std::vector<float3>& host_colors,
	float3 center)
{
    
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
    cudaDestroySurfaceObject(surfaceObject);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    cudaDeviceSynchronize();
}

void UpdateCUDATexture(unsigned int textureID, unsigned int width, unsigned int height, unsigned int xOffset, int yOffset)
{
    if (cudaResource == nullptr) return;

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

    // (5) Kernel Launch
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    fillTextureKernel << <gridSize, blockSize >> > (surfaceObject, width, height, xOffset, yOffset);
    cudaDeviceSynchronize();

    // (6) Clean up
    cudaDestroySurfaceObject(surfaceObject);
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    cudaDeviceSynchronize();

    //  업데이트 후 Mipmap 갱신
    glBindTexture(GL_TEXTURE_2D, textureID);
    glGenerateMipmap(GL_TEXTURE_2D);
    //glBindTexture(GL_TEXTURE_2D, 0);

    nvtxRangePop();
}
