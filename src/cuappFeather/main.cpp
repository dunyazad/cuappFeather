#pragma warning(disable : 4819)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4244)

#include <iostream>

#include "main.cuh"

#include <iostream>
using namespace std;

#include <libFeather.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//const string resource_file_name_ply = "../../res/3D/Compound_Partial.ply";
//const string resource_file_name_alp = "../../res/3D/Compound_Partial.alp";

//const string resource_file_name_ply = "../../res/3D/ZeroCrossingPoints_Partial.ply";
//const string resource_file_name_alp = "../../res/3D/ZeroCrossingPoints_Partial.alp";

//const string resource_file_name_ply = "../../res/3D/Teeth.ply";
//const string resource_file_name_alp = "../../res/3D/Teeth.alp";

//const string resource_file_name_ply = "../../res/3D/Normal.ply";
//const string resource_file_name_alp = "../../res/3D/Normal.alp";

//const string resource_file_name = "Compound_Partial";
const string resource_file_name = "Compound";
const string resource_file_name_ply = "../../res/3D/" + resource_file_name + ".ply";
const string resource_file_name_alp = "../../res/3D/" + resource_file_name + ".alp";

const f32 voxelSize = 0.1f;

#pragma once
#include <algorithm>
#include <cmath>

class ColorSpace {
public:
	// Convert RGB [0.0, 1.0] ¡æ HSV [H: 0-360, S: 0-1, V: 0-1]
	static void RGBtoHSV(float r, float g, float b, float& h, float& s, float& v) {
		float max = std::max({ r, g, b });
		float min = std::min({ r, g, b });
		v = max;

		float delta = max - min;
		if (delta < 1e-6f) {
			h = 0.0f;
			s = 0.0f;
			return;
		}

		s = delta / max;

		if (r == max)
			h = 60.0f * fmodf((g - b) / delta, 6.0f);
		else if (g == max)
			h = 60.0f * ((b - r) / delta + 2.0f);
		else
			h = 60.0f * ((r - g) / delta + 4.0f);

		if (h < 0.0f)
			h += 360.0f;
	}

	// Convert HSV [H: 0-360, S: 0-1, V: 0-1] ¡æ RGB [0.0, 1.0]
	static void HSVtoRGB(float h, float s, float v, float& r, float& g, float& b) {
		float c = v * s;
		float h_prime = h / 60.0f;
		float x = c * (1.0f - fabsf(fmodf(h_prime, 2.0f) - 1.0f));

		float r1, g1, b1;
		if (0 <= h_prime && h_prime < 1) {
			r1 = c; g1 = x; b1 = 0;
		}
		else if (1 <= h_prime && h_prime < 2) {
			r1 = x; g1 = c; b1 = 0;
		}
		else if (2 <= h_prime && h_prime < 3) {
			r1 = 0; g1 = c; b1 = x;
		}
		else if (3 <= h_prime && h_prime < 4) {
			r1 = 0; g1 = x; b1 = c;
		}
		else if (4 <= h_prime && h_prime < 5) {
			r1 = x; g1 = 0; b1 = c;
		}
		else if (5 <= h_prime && h_prime < 6) {
			r1 = c; g1 = 0; b1 = x;
		}
		else {
			r1 = g1 = b1 = 0;
		}

		float m = v - c;
		r = r1 + m;
		g = g1 + m;
		b = b1 + m;
	}
};


int main(int argc, char** argv)
{
	cout << "AppFeather" << endl;

	Feather.Initialize(1920, 1080);

	auto w = Feather.GetFeatherWindow();

	ALPFormat<PointPNC> alp;

	bool tick = false;

#pragma region AppMain
	{
		auto appMain = Feather.CreateEntity("AppMain");
		Feather.CreateEventCallback<KeyEvent>(appMain, [&tick](Entity entity, const KeyEvent& event) {
			if (GLFW_KEY_ESCAPE == event.keyCode)
			{
				glfwSetWindowShouldClose(Feather.GetFeatherWindow()->GetGLFWwindow(), true);
			}
			else if (GLFW_KEY_TAB == event.keyCode)
			{
				if (0 == event.action)
				{
					auto o = Feather.GetEntity("Input Point Cloud _ O");
					Feather.GetComponent<Renderable>(o).SetVisible(tick);
					auto p = Feather.GetEntity("Input Point Cloud");
					Feather.GetComponent<Renderable>(p).SetVisible(!tick);

					tick = !tick;
				}
			}
			});
	}
#pragma endregion

#pragma region Camera
	{
		Entity cam = Feather.CreateEntity("Camera");
		auto& pcam = Feather.CreateComponent<PerspectiveCamera>(cam);
		auto& pcamMan = Feather.CreateComponent<CameraManipulatorTrackball>(cam);
		pcamMan.SetCamera(&pcam);

		Feather.CreateEventCallback<FrameBufferResizeEvent>(cam, [&pcam](Entity entity, const FrameBufferResizeEvent& event) {
			auto window = Feather.GetFeatherWindow();
			auto aspectRatio = (f32)window->GetWidth() / (f32)window->GetHeight();
			pcam.SetAspectRatio(aspectRatio);
			});

		Feather.CreateEventCallback<KeyEvent>(cam, [](Entity entity, const KeyEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity).OnKey(event);
			});

		Feather.CreateEventCallback<MousePositionEvent>(cam, [](Entity entity, const MousePositionEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity).OnMousePosition(event);
			});

		Feather.CreateEventCallback<MouseButtonEvent>(cam, [](Entity entity, const MouseButtonEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity).OnMouseButton(event);
			});

		Feather.CreateEventCallback<MouseWheelEvent>(cam, [](Entity entity, const MouseWheelEvent& event) {
			Feather.GetComponent<CameraManipulatorTrackball>(entity).OnMouseWheel(event);
			});
	}
#pragma endregion

#pragma region Status Panel
	{
		auto gui = Feather.CreateEntity("Status Panel");
		auto statusPanel = Feather.CreateComponent<StatusPanel>(gui);

		Feather.CreateEventCallback<MousePositionEvent>(gui, [](Entity entity, const MousePositionEvent& event) {
			auto& component = Feather.GetComponent<StatusPanel>(entity);
			component.mouseX = event.xpos;
			component.mouseY = event.ypos;
			});
	}
#pragma endregion

//#define Load Model Default
#ifdef Load Model Default
	Feather.AddOnInitializeCallback([&]() {
#pragma region Load PLY and Convert to ALP format
		{
			auto t = Time::Now();

			if (false == alp.Deserialize(resource_file_name_alp))
			{
				PLYFormat ply;
				ply.Deserialize(resource_file_name_ply);
				//ply.SwapAxisYZ();

				vector<PointPNC> points;
				for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
				{
					auto px = ply.GetPoints()[i * 3];
					auto py = ply.GetPoints()[i * 3 + 1];
					auto pz = ply.GetPoints()[i * 3 + 2];

					auto nx = ply.GetNormals()[i * 3];
					auto ny = ply.GetNormals()[i * 3 + 1];
					auto nz = ply.GetNormals()[i * 3 + 2];

					if (false == ply.GetColors().empty())
					{
						if (ply.UseAlpha())
						{
							auto cx = ply.GetColors()[i * 4];
							auto cy = ply.GetColors()[i * 4 + 1];
							auto cz = ply.GetColors()[i * 4 + 2];
							auto ca = ply.GetColors()[i * 4 + 3];

							points.push_back({ {px, py, pz}, {nx, ny, nz}, {cx, cy, cz} });
		}
						else
						{
							auto cx = ply.GetColors()[i * 3];
							auto cy = ply.GetColors()[i * 3 + 1];
							auto cz = ply.GetColors()[i * 3 + 2];

							points.push_back({ {px, py, pz}, {nx, ny, nz}, {cx, cy, cz} });
						}
		}
					else
					{
						points.push_back({ {px, py, pz}, {nx, ny, nz}, {1.0f, 1.0f, 1.0f} });
					}
}
				alog("PLY %llu points loaded\n", points.size());

				alp.AddPoints(points);
				alp.Serialize(resource_file_name_alp);
			}

			t = Time::End(t, "Loading Compound");

			auto entity = Feather.CreateEntity("Input Point Cloud _ O");
			auto& renderable = Feather.CreateComponent<Renderable>(entity);

			renderable.Initialize(Renderable::GeometryMode::Triangles);
			renderable.AddShader(Feather.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
			renderable.AddShader(Feather.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
			renderable.SetActiveShaderIndex(1);

			auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere("zero", 0.05f, 6, 6);
			//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
			renderable.AddIndices(indices);
			renderable.AddVertices(vertices);
			renderable.AddNormals(normals);
			renderable.AddColors(colors);
			renderable.AddUVs(uvs);

			vector<float3> host_points;
			vector<float3> host_normals;
			vector<float3> host_colors;

			for (auto& p : alp.GetPoints())
			{
				auto r = p.color.x;
				auto g = p.color.y;
				auto b = p.color.z;
				auto a = 1.f;

				renderable.AddInstanceColor(MiniMath::V4(r, g, b, a));
				renderable.AddInstanceNormal(p.normal);

				MiniMath::M4 model = MiniMath::M4::identity();
				model.m[0][0] = 1.5f;
				model.m[1][1] = 1.5f;
				model.m[2][2] = 1.5f;
				model = MiniMath::translate(model, p.position);
				renderable.AddInstanceTransform(model);

				host_points.push_back(make_float3(p.position.x, p.position.y, p.position.z));
				host_normals.push_back(make_float3(p.normal.x, p.normal.y, p.normal.z));
				host_colors.push_back(make_float3(r, g, b));
			}

			alog("ALP %llu points loaded\n", alp.GetPoints().size());

			renderable.EnableInstancing(alp.GetPoints().size());
			auto [x, y, z] = alp.GetAABBCenter();
			f32 cx = x;
			f32 cy = y;
			f32 cz = z;

			Feather.CreateEventCallback<KeyEvent>(entity, [cx, cy, cz](Entity entity, const KeyEvent& event) {
				auto& renderable = Feather.GetComponent<Renderable>(entity);
				if (GLFW_KEY_M == event.keyCode)
				{
					renderable.NextDrawingMode();
				}
				else if (GLFW_KEY_1 == event.keyCode)
				{
					renderable.SetActiveShaderIndex(0);
				}
				else if (GLFW_KEY_2 == event.keyCode)
				{
					renderable.SetActiveShaderIndex(1);
				}
				else if (GLFW_KEY_PAGE_UP == event.keyCode)
				{

				}
				else if (GLFW_KEY_PAGE_DOWN == event.keyCode)
				{
				}
				else if (GLFW_KEY_R == event.keyCode)
				{
					auto entities = Feather.GetRegistry().view<CameraManipulatorTrackball>();
					for (auto& entity : entities)
					{
						auto cameraManipulator = Feather.GetComponent<CameraManipulatorTrackball>(entity);
						auto camera = cameraManipulator.GetCamera();
						camera->SetEye({ cx,cy,cz + cameraManipulator.GetRadius() });
						camera->SetTarget({ cx,cy,cz });
					}
				}
				});

			t = Time::End(t, "Upload to GPU");

			auto hashToFloat = [](uint32_t seed) -> float {
				seed ^= seed >> 13;
				seed *= 0x5bd1e995;
				seed ^= seed >> 15;
				return (seed & 0xFFFFFF) / static_cast<float>(0xFFFFFF);
			};

			auto pointLabels = cuMain(host_points, host_normals, host_colors, make_float3(x, y, z));
			for (size_t i = 0; i < pointLabels.size(); i++)
			{
				auto label = pointLabels[i];
				if (label != -1)
				{
					float r = hashToFloat(label * 3 + 0);
					float g = hashToFloat(label * 3 + 1);
					float b = hashToFloat(label * 3 + 2);

					renderable.SetInstanceColor(i, MiniMath::V4(r, g, b, 1.0f));
				}
			}

			{ // AABB
				auto m = alp.GetAABBMin();
				float x = get<0>(m);
				float y = get<1>(m);
				float z = get<2>(m);
				auto M = alp.GetAABBMax();
				float X = get<0>(M);
				float Y = get<1>(M);
				float Z = get<2>(M);

				auto entity = Feather.CreateEntity("AABB");
				auto& renderable = Feather.CreateComponent<Renderable>(entity);
				renderable.Initialize(Renderable::GeometryMode::Lines);

				renderable.AddShader(Feather.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });

				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });

				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
			}

			{ // Cache Area
				auto [cx, cy, cz] = alp.GetAABBCenter();

				float x = cx + (-10.0f);
				float y = cy + (-15.0f);
				float z = cz + (-20.0f);
				float X = cx + (10.0f);
				float Y = cy + (15.0f);
				float Z = cz + (20.0f);

				auto entity = Feather.CreateEntity("CacheAABB");
				auto& renderable = Feather.CreateComponent<Renderable>(entity);
				renderable.Initialize(Renderable::GeometryMode::Lines);

				renderable.AddShader(Feather.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });

				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });

				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
			}
		}
#pragma endregion
		});
#endif

#define CLUSTERING
#ifdef CLUSTERING
	Feather.AddOnInitializeCallback([&]() {

		//#define RENDER_TRIANGLE
#ifdef RENDER_TRIANGLE
		{
			auto entity = Feather.CreateInstance<Entity>("Triangle");
			auto shader = Feather.CreateInstance<Shader>();
			shader->Initialize(File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs"));
			auto renderable = Feather.CreateInstance<Renderable>();
			renderable->Initialize(Renderable::GeometryMode::Triangles);
			renderable->SetShader(shader);

			renderable->AddVertex({ -1.0f, -1.0f, 0.0f });
			renderable->AddVertex({ 1.0f, -1.0f, 0.0f });
			renderable->AddVertex({ 0.0f, 1.0f, 0.0f });

			renderable->AddNormal({ 0.0f, 0.0f, 1.0f });
			renderable->AddNormal({ 0.0f, 0.0f, 1.0f });
			renderable->AddNormal({ 0.0f, 0.0f, 1.0f });

			renderable->AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
			renderable->AddColor({ 0.0f, 1.0f, 0.0f, 1.0f });
			renderable->AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });

			renderable->AddIndex(0);
			renderable->AddIndex(1);
			renderable->AddIndex(2);
		}
#endif // RENDER_TRIANGLE

		//#define RENDER_VOXELS_BOX
#ifdef RENDER_VOXELS_BOX
		{
			auto entity = Feather.CreateInstance<Entity>("Box");
			auto shader = Feather.CreateInstance<Shader>();
			shader->Initialize(File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs"));
			auto renderable = Feather.CreateInstance<Renderable>();
			renderable->Initialize(Renderable::GeometryMode::Triangles);
			renderable->SetShader(shader);

			auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "one");
			renderable->AddIndices(indices);
			renderable->AddVertices(vertices);
			renderable->AddNormals(normals);
			renderable->AddColors(colors);
			renderable->AddUVs(uvs);

			int xCount = 100;
			int yCount = 100;
			int zCount = 100;
			int tCount = xCount * yCount * zCount;

			for (int i = 0; i < tCount; i++)
			{
				int z = i / (xCount * yCount);
				int y = (i % (xCount * yCount)) / xCount;
				int x = (i % (xCount * yCount)) % xCount;

				MiniMath::M4 model = MiniMath::M4::identity();
				model.m[0][0] = 0.5f;
				model.m[1][1] = 0.5f;
				model.m[2][2] = 0.5f;
				model = MiniMath::translate(model, MiniMath::V3(x, y, z));
				renderable->AddInstanceTransform(model);
			}

			renderable->EnableInstancing(tCount);
		}
#endif // RENDER_VOXELS_BOX

		//#define RENDER_VOXELS_SPHERE
#ifdef RENDER_VOXELS_SPHERE
		{
			auto entity = Feather.CreateInstance<Entity>("Box");
			auto shader = Feather.CreateInstance<Shader>();
			shader->Initialize(File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs"));
			auto renderable = Feather.CreateInstance<Renderable>();
			renderable->Initialize(Renderable::GeometryMode::Triangles);
			renderable->SetShader(shader);

			auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere("zero", 0.25f, 6, 6);
			//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
			renderable->AddIndices(indices);
			renderable->AddVertices(vertices);
			renderable->AddNormals(normals);
			renderable->AddColors(colors);
			renderable->AddUVs(uvs);

			int xCount = 100;
			int yCount = 100;
			int zCount = 100;
			int tCount = xCount * yCount * zCount;

			for (int i = 0; i < tCount; i++)
			{
				int z = i / (xCount * yCount);
				int y = (i % (xCount * yCount)) / xCount;
				int x = (i % (xCount * yCount)) % xCount;

				MiniMath::M4 model = MiniMath::M4::identity();
				model.m[0][0] = 0.5f;
				model.m[1][1] = 0.5f;
				model.m[2][2] = 0.5f;
				model = MiniMath::translate(model, MiniMath::V3(x, y, z));
				renderable->AddInstanceTransform(model);
			}

			renderable->EnableInstancing(tCount);

			renderable->AddEventHandler(EventType::KeyPress, [&](const Event& event, FeatherObject* object) {
				if (GLFW_KEY_M == event.keyEvent.keyCode)
				{
					auto renderable = dynamic_cast<Renderable*>(object);
					renderable->NextDrawingMode();
				}
				});
		}
#endif // RENDER_VOXELS_BOX

		//#define LOAD_PLY
#ifdef LOAD_PLY
		{
			auto entity = Feather.CreateInstance<Entity>("Teeth");
			auto shader = Feather.CreateInstance<Shader>();
			shader->Initialize(File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs"));

			auto renderable = Feather.CreateInstance<Renderable>();
			renderable->Initialize(Renderable::GeometryMode::Points);
			renderable->SetShader(shader);

			PLYFormat ply;
			ply.Deserialize("../../res/3D/Teeth.ply");
			ply.SwapAxisYZ();

			if (false == ply.GetPoints().empty())
				renderable->AddVertices((MiniMath::V3*)ply.GetPoints().data(), ply.GetPoints().size() / 3);
			if (false == ply.GetNormals().empty())
				renderable->AddNormals((MiniMath::V3*)ply.GetNormals().data(), ply.GetNormals().size() / 3);
			if (false == ply.GetColors().empty())
			{
				if (ply.UseAlpha())
				{
					renderable->AddColors((MiniMath::V4*)ply.GetColors().data(), ply.GetColors().size() / 4);
				}
				else
				{
					renderable->AddColors((MiniMath::V3*)ply.GetColors().data(), ply.GetColors().size() / 3);
				}
			}

			auto cameraManipulator = Feather.GetFirstInstance<CameraManipulatorTrackball>();
			auto camera = cameraManipulator->SetCamera();
			auto [x, y, z] = ply.GetAABBCenter();
			camera->SetEye({ x,y,z + cameraManipulator->GetRadius() });
			camera->SetTarget({ x,y,z });
		}
#endif // LOAD_PLY

		/*
		{
			PLYFormat ply;
			ply.Deserialize("../../res/3D/Teeth.ply");
			ply.SwapAxisYZ();

			auto entity = Feather.CreateInstance<Entity>("Box");
			auto shader = Feather.CreateInstance<Shader>();
			shader->Initialize(File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs"));
			auto renderable = Feather.CreateInstance<Renderable>();
			renderable->Initialize(Renderable::GeometryMode::Triangles);
			renderable->SetShader(shader);

			auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere("zero", 0.05f, 6, 6);
			//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
			renderable->AddIndices(indices);
			renderable->AddVertices(vertices);
			renderable->AddNormals(normals);
			renderable->AddColors(colors);
			renderable->AddUVs(uvs);

			ui32 tCount = ply.GetPoints().size() / 3;

			for (int i = 0; i < tCount; i++)
			{
				auto x = ply.GetPoints()[i * 3];
				auto y = ply.GetPoints()[i * 3 + 1];
				auto z = ply.GetPoints()[i * 3 + 2];

				MiniMath::M4 model = MiniMath::M4::identity();
				model.m[0][0] = 0.5f;
				model.m[1][1] = 0.5f;
				model.m[2][2] = 0.5f;
				model = MiniMath::translate(model, MiniMath::V3(x, y, z));
				renderable->AddInstanceTransform(model);
			}

			renderable->EnableInstancing(tCount);

			renderable->AddEventHandler(EventType::KeyPress, [&](const Event& event, FeatherObject* object) {
				if (GLFW_KEY_M == event.keyEvent.keyCode)
				{
					auto renderable = dynamic_cast<Renderable*>(object);
					renderable->NextDrawingMode();
				}
				});
		}
		*/

#pragma region Load PLY and Convert to ALP format
		{
			auto t = Time::Now();

			if (false == alp.Deserialize(resource_file_name_alp))
			{
				bool foundZero = false;

				PLYFormat ply;
				ply.Deserialize(resource_file_name_ply);
				//ply.SwapAxisYZ();

				vector<PointPNC> points;
				for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
				{
					auto px = ply.GetPoints()[i * 3];
					auto py = ply.GetPoints()[i * 3 + 1];
					auto pz = ply.GetPoints()[i * 3 + 2];

					if (0 == px && 0 == py && 0 == pz)
					{
						if (false == foundZero)
						{
							foundZero = true;
						}
						else
						{
							continue;
						}
					}

					auto nx = ply.GetNormals()[i * 3];
					auto ny = ply.GetNormals()[i * 3 + 1];
					auto nz = ply.GetNormals()[i * 3 + 2];

					if (false == ply.GetColors().empty())
					{
						if (ply.UseAlpha())
						{
							auto cx = ply.GetColors()[i * 4];
							auto cy = ply.GetColors()[i * 4 + 1];
							auto cz = ply.GetColors()[i * 4 + 2];
							auto ca = ply.GetColors()[i * 4 + 3];

							points.push_back({ {px, py, pz}, {nx, ny, nz}, {cx, cy, cz} });
						}
						else
						{
							auto cx = ply.GetColors()[i * 3];
							auto cy = ply.GetColors()[i * 3 + 1];
							auto cz = ply.GetColors()[i * 3 + 2];

							points.push_back({ {px, py, pz}, {nx, ny, nz}, {cx, cy, cz} });
						}
					}
					else
					{
						points.push_back({ {px, py, pz}, {nx, ny, nz}, {1.0f, 1.0f, 1.0f} });
					}
				}
				alog("PLY %llu points loaded\n", points.size());

				alp.AddPoints(points);
				alp.Serialize(resource_file_name_alp);
			}

			t = Time::End(t, "Loading Compound");

			auto entity = Feather.CreateEntity("Input Point Cloud _ O");
			auto& renderable = Feather.CreateComponent<Renderable>(entity);

			renderable.Initialize(Renderable::GeometryMode::Triangles);
			renderable.AddShader(Feather.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
			renderable.AddShader(Feather.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
			renderable.SetActiveShaderIndex(1);

			auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere("zero", 0.05f, 6, 6);
			//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
			renderable.AddIndices(indices);
			renderable.AddVertices(vertices);
			renderable.AddNormals(normals);
			renderable.AddColors(colors);
			renderable.AddUVs(uvs);

			vector<float3> host_points;
			vector<float3> host_normals;
			vector<float3> host_colors;

			for (auto& p : alp.GetPoints())
			{
				auto r = p.color.x;
				auto g = p.color.y;
				auto b = p.color.z;
				auto a = 1.f;

				float h = 0.0f;
				float s = 0.0f;
				float v = 0.0f;

				renderable.AddInstanceColor(MiniMath::V4(r, g, b, a));
				renderable.AddInstanceNormal(p.normal);

				MiniMath::M4 model = MiniMath::M4::identity();
				model.m[0][0] = 1.5f;
				model.m[1][1] = 1.5f;
				model.m[2][2] = 1.5f;
				model = MiniMath::translate(model, p.position);
				renderable.AddInstanceTransform(model);

				host_points.push_back(make_float3(p.position.x, p.position.y, p.position.z));
				host_normals.push_back(make_float3(p.normal.x, p.normal.y, p.normal.z));
				host_colors.push_back(make_float3(r, g, b));
			}

			alog("ALP %llu points loaded\n", alp.GetPoints().size());

			renderable.EnableInstancing(alp.GetPoints().size());
			auto [x, y, z] = alp.GetAABBCenter();
			f32 cx = x;
			f32 cy = y;
			f32 cz = z;

			auto [mx, my, mz] = alp.GetAABBMin();
			auto [Mx, My, Mz] = alp.GetAABBMax();
			f32 lx = Mx - mx;
			f32 ly = My - my;
			f32 lz = Mz - mz;

			Entity cam = Feather.GetEntity("Camera");
			auto& pcam = Feather.GetComponent<PerspectiveCamera>(cam);
			auto& cameraManipulator = Feather.GetComponent<CameraManipulatorTrackball>(cam);
			cameraManipulator.SetRadius(lx + ly + lz);
			auto camera = cameraManipulator.GetCamera();
			camera->SetEye({ cx,cy,cz + cameraManipulator.GetRadius() });
			camera->SetTarget({ cx,cy,cz });

			cameraManipulator.MakeDefault();


			Feather.CreateEventCallback<KeyEvent>(entity, [cx, cy, cz, lx, ly, lz](Entity entity, const KeyEvent& event) {
				auto& renderable = Feather.GetComponent<Renderable>(entity);
				if (GLFW_KEY_M == event.keyCode)
				{
					renderable.NextDrawingMode();
				}
				else if (GLFW_KEY_1 == event.keyCode)
				{
					renderable.SetActiveShaderIndex(0);
				}
				else if (GLFW_KEY_2 == event.keyCode)
				{
					renderable.SetActiveShaderIndex(1);
				}
				else if (GLFW_KEY_PAGE_UP == event.keyCode)
				{

				}
				else if (GLFW_KEY_PAGE_DOWN == event.keyCode)
				{
				}
				//else if (GLFW_KEY_R == event.keyCode)
				//{
				//	auto entities = Feather.GetRegistry().view<CameraManipulatorTrackball>();
				//	for (auto& entity : entities)
				//	{
				//		auto cameraManipulator = Feather.GetComponent<CameraManipulatorTrackball>(entity);

				//		cameraManipulator.SetRadius((lx + ly + lz) * 2.0f);

				//		auto camera = cameraManipulator.GetCamera();
				//		camera->SetEye({ cx,cy,cz + cameraManipulator.GetRadius() });
				//		camera->SetTarget({ cx,cy,cz });
				//	}
				//}
				});

			t = Time::End(t, "Upload to GPU");

			auto hashToFloat = [](uint32_t seed) -> float {
				seed ^= seed >> 13;
				seed *= 0x5bd1e995;
				seed ^= seed >> 15;
				return (seed & 0xFFFFFF) / static_cast<float>(0xFFFFFF);
			};

			auto pointLabels = cuMain(voxelSize, host_points, host_normals, host_colors, make_float3(x, y, z));

			for (size_t i = 0; i < host_colors.size(); i++)
			{
				auto label = pointLabels[i];
				if (label != -1)
				{
					float r = hashToFloat(label * 3 + 0);
					float g = hashToFloat(label * 3 + 1);
					float b = hashToFloat(label * 3 + 2);

					renderable.SetInstanceColor(i, MiniMath::V4(r, g, b, 1.0f));
				}

				//auto c = host_colors[i];
				////printf("%f, %f, %f\n", c.x, c.y, c.z);
				//renderable.SetInstanceColor(i, MiniMath::V4(c.x, c.y, c.z, 1.0f));
			}

			/*
			for (size_t i = 0; i < pointLabels.size(); i++)
			{
				auto label = pointLabels[i];
				if (label != -1)
				{
					float r = hashToFloat(label * 3 + 0);
					float g = hashToFloat(label * 3 + 1);
					float b = hashToFloat(label * 3 + 2);

					renderable.SetInstanceColor(i, MiniMath::V4(r, g, b, 1.0f));
				}
			}

			{ // AABB
				auto m = alp.GetAABBMin();
				float x = get<0>(m);
				float y = get<1>(m);
				float z = get<2>(m);
				auto M = alp.GetAABBMax();
				float X = get<0>(M);
				float Y = get<1>(M);
				float Z = get<2>(M);

				auto entity = Feather.CreateEntity("AABB");
				auto& renderable = Feather.CreateComponent<Renderable>(entity);
				renderable.Initialize(Renderable::GeometryMode::Lines);

				renderable.AddShader(Feather.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });

				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });

				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 0.0f, 0.0f, 1.0f, 1.0f });
			}

			{ // Cache Area
				auto [cx, cy, cz] = alp.GetAABBCenter();

				float x = cx + (-100.0f) * voxelSize;
				float y = cy + (-150.0f) * voxelSize;
				float z = cz + (-200.0f) * voxelSize;
				float X = cx + (100.0f) * voxelSize;
				float Y = cy + (150.0f) * voxelSize;
				float Z = cz + (200.0f) * voxelSize;

				auto entity = Feather.CreateEntity("CacheAABB");
				auto& renderable = Feather.CreateComponent<Renderable>(entity);
				renderable.Initialize(Renderable::GeometryMode::Lines);

				renderable.AddShader(Feather.CreateShader("Line", File("../../res/Shaders/Line.vs"), File("../../res/Shaders/Line.fs")));

				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });

				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });

				renderable.AddVertex({ x, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ X, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
				renderable.AddVertex({ x, Y, Z }); renderable.AddColor({ 1.0f, 0.0f, 0.0f, 1.0f });
			}
			*/
		}
#pragma endregion

#pragma region Load PLY and Convert to ALP format
		{
			auto t = Time::Now();

			if (false == alp.Deserialize(resource_file_name_alp))
			{
				PLYFormat ply;
				ply.Deserialize(resource_file_name_ply);
				//ply.SwapAxisYZ();

				vector<PointPNC> points;
				for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
				{
					auto px = ply.GetPoints()[i * 3];
					auto py = ply.GetPoints()[i * 3 + 1];
					auto pz = ply.GetPoints()[i * 3 + 2];

					auto nx = ply.GetNormals()[i * 3];
					auto ny = ply.GetNormals()[i * 3 + 1];
					auto nz = ply.GetNormals()[i * 3 + 2];

					if (false == ply.GetColors().empty())
					{
						if (ply.UseAlpha())
						{
							auto cx = ply.GetColors()[i * 4];
							auto cy = ply.GetColors()[i * 4 + 1];
							auto cz = ply.GetColors()[i * 4 + 2];
							auto ca = ply.GetColors()[i * 4 + 3];

							points.push_back({ {px, py, pz}, {nx, ny, nz}, {cx, cy, cz} });
						}
						else
						{
							auto cx = ply.GetColors()[i * 3];
							auto cy = ply.GetColors()[i * 3 + 1];
							auto cz = ply.GetColors()[i * 3 + 2];

							points.push_back({ {px, py, pz}, {nx, ny, nz}, {cx, cy, cz} });
						}
					}
					else
					{
						points.push_back({ {px, py, pz}, {nx, ny, nz}, {1.0f, 1.0f, 1.0f} });
					}
				}
				alog("PLY %llu points loaded\n", points.size());

				alp.AddPoints(points);
				alp.Serialize(resource_file_name_alp);
			}

			t = Time::End(t, "Loading Compound");

			auto entity = Feather.CreateEntity("Input Point Cloud");
			auto& renderable = Feather.CreateComponent<Renderable>(entity);

			renderable.SetVisible(false);

			renderable.Initialize(Renderable::GeometryMode::Triangles);
			renderable.AddShader(Feather.CreateShader("Instancing", File("../../res/Shaders/Instancing.vs"), File("../../res/Shaders/Instancing.fs")));
			renderable.AddShader(Feather.CreateShader("InstancingWithoutNormal", File("../../res/Shaders/InstancingWithoutNormal.vs"), File("../../res/Shaders/InstancingWithoutNormal.fs")));
			renderable.SetActiveShaderIndex(1);

			auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildSphere("zero", 0.05f, 6, 6);
			//auto [indices, vertices, normals, colors, uvs] = GeometryBuilder::BuildBox("zero", "half");
			renderable.AddIndices(indices);
			renderable.AddVertices(vertices);
			renderable.AddNormals(normals);
			renderable.AddColors(colors);
			renderable.AddUVs(uvs);

			vector<float3> host_points;
			vector<float3> host_normals;
			vector<float3> host_colors;

			for (auto& p : alp.GetPoints())
			{
				auto r = p.color.x;
				auto g = p.color.y;
				auto b = p.color.z;
				auto a = 1.f;

				renderable.AddInstanceColor(MiniMath::V4(r, g, b, a));
				renderable.AddInstanceNormal(p.normal);

				MiniMath::M4 model = MiniMath::M4::identity();
				model.m[0][0] = 1.5f;
				model.m[1][1] = 1.5f;
				model.m[2][2] = 1.5f;
				model = MiniMath::translate(model, p.position);
				renderable.AddInstanceTransform(model);

				host_points.push_back(make_float3(p.position.x, p.position.y, p.position.z));
				host_normals.push_back(make_float3(p.normal.x, p.normal.y, p.normal.z));
				host_colors.push_back(make_float3(r, g, b));
			}

			alog("ALP %llu points loaded\n", alp.GetPoints().size());

			renderable.EnableInstancing(alp.GetPoints().size());
			auto [x, y, z] = alp.GetAABBCenter();
			f32 cx = x;
			f32 cy = y;
			f32 cz = z;

			Feather.CreateEventCallback<KeyEvent>(entity, [cx,cy,cz](Entity entity, const KeyEvent& event) {
				auto& renderable = Feather.GetComponent<Renderable>(entity);
				if (GLFW_KEY_M == event.keyCode)
				{
					renderable.NextDrawingMode();
				}
				else if (GLFW_KEY_1 == event.keyCode)
				{
					renderable.SetActiveShaderIndex(0);
				}
				else if (GLFW_KEY_2 == event.keyCode)
				{
					renderable.SetActiveShaderIndex(1);
				}
				else if (GLFW_KEY_PAGE_UP == event.keyCode)
				{

				}
				else if (GLFW_KEY_PAGE_DOWN == event.keyCode)
				{
				}
				//else if (GLFW_KEY_R == event.keyCode)
				//{
				//	auto entities = Feather.GetRegistry().view<CameraManipulatorTrackball>();
				//	for (auto& entity : entities)
				//	{
				//		auto cameraManipulator = Feather.GetComponent<CameraManipulatorTrackball>(entity);
				//		auto camera = cameraManipulator.GetCamera();
				//		camera->SetEye({ cx,cy,cz + cameraManipulator.GetRadius() });
				//		camera->SetTarget({ cx,cy,cz });
				//	}
				//}
				});

			//renderable->AddEventHandler(EventType::KeyPress, [renderable, &alp](const Event& event, FeatherObject* object) {
			//	if (GLFW_KEY_M == event.keyEvent.keyCode)
			//	{
			//		auto renderable = dynamic_cast<Renderable*>(object);
			//		renderable->NextDrawingMode();
			//	}
			//	else if (GLFW_KEY_1 == event.keyEvent.keyCode)
			//	{
			//		auto renderable = dynamic_cast<Renderable*>(object);
			//		renderable->SetActiveShaderIndex(0);
			//	}
			//	else if (GLFW_KEY_2 == event.keyEvent.keyCode)
			//	{
			//		auto renderable = dynamic_cast<Renderable*>(object);
			//		renderable->SetActiveShaderIndex(1);
			//	}
			//	else if (GLFW_KEY_R == event.keyEvent.keyCode)
			//	{
			//		auto cameraManipulator = Feather.GetFirstInstance<CameraManipulatorTrackball>();
			//		auto camera = cameraManipulator->SetCamera();
			//		auto [x, y, z] = alp.GetAABBCenter();
			//		camera->SetEye({ x,y,z + cameraManipulator->GetRadius() });
			//		camera->SetTarget({ x,y,z });
			//	}
			//	});
			//renderable->AddEventHandler(EventType::MouseButtonRelease, [renderable](const Event& event, FeatherObject* object) {
			//	if (GLFW_MOUSE_BUTTON_1 == event.mouseButtonEvent.button)
			//	{
			//		auto camera = Feather.GetFirstInstance<PerspectiveCamera>();
			//		auto viewMatrix = camera->GetViewMatrix();
			//		//viewMatrix.at(0, 3)
			//	}
			//	});
			
			t = Time::End(t, "Upload to GPU");

			auto hashToFloat = [](uint32_t seed) -> float {
				seed ^= seed >> 13;
				seed *= 0x5bd1e995;
				seed ^= seed >> 15;
				return (seed & 0xFFFFFF) / static_cast<float>(0xFFFFFF);
			};

			auto pointLabels = cuMain(voxelSize, host_points, host_normals, host_colors, make_float3(x,y,z));

			std::unordered_map<unsigned int, unsigned int> labelHistogram;

			for (auto& i : pointLabels)
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

			unsigned int maxLabel = 0;
			unsigned int maxLabelCount = 0;

			for (auto& [label, count] : labelHistogram)
			{
				//alog("[%8d] : %d\n", label, count);

				if (-1 == label) continue;
				if (count > maxLabelCount)
				{
					maxLabel = label;
					maxLabelCount = count;
				}
			}

			for (size_t i = 0; i < pointLabels.size(); i++)
			{
				auto label = pointLabels[i];
				if (maxLabel == label)
				{
					float r = hashToFloat(label * 3 + 0);
					float g = hashToFloat(label * 3 + 1);
					float b = hashToFloat(label * 3 + 2);

					renderable.SetInstanceColor(i, MiniMath::V4(r, g, b, 1.0f));
				}
				else
				{
					auto m = renderable.GetInstanceTransform(i);
					m.at(0, 0) *= 0.125f;
					m.at(1, 1) *= 0.125f;
					m.at(2, 2) *= 0.125f;
					renderable.SetInstanceTransform(i, m);
					renderable.SetInstanceColor(i, MiniMath::V4(1.0f, 0.0f, 0.0f, 0.0f));
				}
				//if (label != -1)
				//{
				//	float r = hashToFloat(label * 3 + 0);
				//	float g = hashToFloat(label * 3 + 1);
				//	float b = hashToFloat(label * 3 + 2);

				//	//if (index == 0)
				//	//{
				//	//	renderable->SetInstanceColor(i, MiniMath::V4(1.0f, 0.0f, 0.0f, 1.0f));
				//	//}
				//	//else
				//	//{
				//	renderable.SetInstanceColor(i, MiniMath::V4(r, g, b, 1.0f));
				//	//}
				//}

				//auto& p = alp.GetPoints()[i];
				//renderable->SetInstanceColor(i, MiniMath::V4(p.color.x, p.color.y, p.color.z, 1.0f));
			}
		}
#pragma endregion
		});
#endif

	Feather.Run();

	Feather.Terminate();

	return 0;
}
