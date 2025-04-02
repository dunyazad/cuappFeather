#include <iostream>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_set>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#define alog(...) printf("\033[38;5;1m\033[48;5;15m(^(OO)^) /V/\033[0m\t" __VA_ARGS__)
#define alogt(tag, ...) printf("\033[38;5;1m\033[48;5;15m [%d] (^(OO)^) /V/\033[0m\t" tag, __VA_ARGS__)

std::vector<unsigned int> cuMain(const std::vector<float3>& host_points, float3 center);
