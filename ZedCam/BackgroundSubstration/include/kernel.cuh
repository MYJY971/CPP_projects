#include "zed/Mat.hpp"
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <algorithm>
#include "npp.h"
#include "device_functions.h"
#include <stdint.h>

// CUDA Function :: fill an image with a checkerboard pattern
void cuCreateCheckerboard(sl::zed::Mat &image);

// CUDA Function :: keep the current pixel if its depth is below a threshold
void cuCroppImageByDepth(sl::zed::Mat &depth, sl::zed::Mat &imageLeft, sl::zed::Mat &imageCut, sl::zed::Mat &mask, float threshold);