///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////




/**********************************************************************************************************************
 ** This sample demonstrates how to use the depth to mask the current image with an other (background subtraction)  **
 ** By default a checkerboard is computed and used as mask but an other image can be given in argument				 **
 ** Some event are linked with keys																					 **
 **********************************************************************************************************************/


//standard include
#include <stdio.h>
#include <string.h>
#include <chrono>

//ZED include
#include <zed/Mat.hpp>
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

//OpenCV include
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Cuda functions include
#include "kernel.cuh"

using namespace sl::zed;
using namespace std;

//main Loop

int main(int argc, char **argv) {

    if (argc > 2) {
        std::cout << "Only the path of an image can be passed in arg" << std::endl;
        return -1;
    }

    Camera* zed = new Camera(HD720);

    bool loadImage = false;
    if (argc == 2)
        loadImage = true;

    InitParams parameters;
    parameters.unit = UNIT::METER; // this sample is designed to work in METER
    ERRCODE err = zed->init(parameters);

    // ERRCODE display
    cout << errcode2str(err) << endl;

    // Quit if an error occurred
    if (err != SUCCESS) {
        delete zed;
        return 1;
    }

    // print on screen the keys that can be used
    bool printHelp = false;
    std::string helpString = "[p] increase distance, [m] decrease distance, [q] quit";

    // get width and height of the ZED images
    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;
    // create and alloc GPU memory for the image matrix
    Mat imageCropp;
    imageCropp.data = (unsigned char*) nppiMalloc_8u_C4(width, height, &imageCropp.step);
    imageCropp.setUp(width, height, 4, sl::zed::UCHAR, GPU);

    // create and alloc GPU memory for the mask matrix
    Mat imageCheckerboard;
    imageCheckerboard.data = (unsigned char*) nppiMalloc_8u_C4(width, height, &imageCheckerboard.step);
    imageCheckerboard.setUp(width, height, 4, sl::zed::UCHAR, GPU);

    if (loadImage) { // if an image is given in argument we load it and use it as background
        string imageName = argv[1];
        cv::Mat imageBackground = cv::imread(imageName);

        if (imageBackground.empty()) { // if the image can't be load we will use a generated image
            loadImage = false;
            cout << " -> ERROR : can't load image : " << imageName << ", generating a synthetic image instead." << endl;
        } else {// adapt the size of the given image to the size of the zed image
            cv::resize(imageBackground, imageBackground, cv::Size(width, height));
            // we work with image in 4 channels for memory alignement purpose
            cv::cvtColor(imageBackground, imageBackground, CV_BGR2BGRA);
            // copy the image from the CPU to the GPU
            cudaMemcpy2D((uchar*) imageCheckerboard.data, imageCheckerboard.step, (Npp8u*) imageBackground.data, imageBackground.step, imageBackground.step, height, cudaMemcpyHostToDevice);
        }
    }

    if (!loadImage)// Compute the checkerboard only one time, it will be use to mask the invalid area
        cuCreateCheckerboard(imageCheckerboard);

    // create a CPU image for display purpose
    cv::Mat imDisplay(height, width, CV_8UC4);

    // define a distance threshold, above it the image will be replace by the mask
    float distCut = 3.; // in Meters
    bool threshAsChange = true;

    char key = ' ';

    std::cout << " Press 'p' to increase distance threshold" << std::endl;
    std::cout << " Press 'm' to decrease distance threshold" << std::endl;

    // launch a loop
    bool run = true;
    while (run) {

        // Grab the current images and compute the disparity
        // we want a full depth map for better visual effect
        bool res = zed->grab(FILL);

        // get the left image
        // !! WARNING !! this is not a copy, here we work with the data allocated by the zed object
        // this can be done ONLY if we call ONE time this methode before the next grab, make a copy if you want to get multiple IMAGE
        Mat imageLeftGPU = zed->getView_gpu(STEREO_LEFT);

        // get the depth
        // !! WARNING !! this is not a copy, here we work with the data allocated by the zed object
        // this can be done ONLY if we call ONE time this methode before the next grab, make a copy if you want to get multiple MEASURE
        Mat depthGPU = zed->retrieveMeasure_gpu(DEPTH);

        // Call the cuda function that mask the image area wich are deeper than the threshold
        cuCroppImageByDepth(depthGPU, imageLeftGPU, imageCropp, imageCheckerboard, distCut);

        // Copy the processed image frome the GPU to the CPU for display
        cudaMemcpy2D((uchar*) imDisplay.data, imDisplay.step, (Npp8u*) imageCropp.data, imageCropp.step, imageCropp.getWidthByte(), imageCropp.height, cudaMemcpyDeviceToHost);

        if (printHelp) // write help text on the image if needed
            cv::putText(imDisplay, helpString, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(111, 111, 111, 255), 2);

        // display the result
        cv::imshow("Image cropped by distance", imDisplay);
        key = cv::waitKey(20);

        switch (key) // handle the pressed key
        {
            case 'q': // close the program
            case 'Q':
                run = false;
                break;

            case 'p': // increase the distance threshold
            case 'P':
                distCut += 0.25;
                threshAsChange = true;
                break;

            case 'm': // decrease the distance threshold
            case 'M':
                distCut = (distCut > 1 ? distCut - 0.25 : 1);
                threshAsChange = true;
                break;

            case 'h': // print help
            case 'H':
                printHelp = !printHelp;
                cout << helpString << endl;
                break;
            default:
                break;
        }

        if (threshAsChange) {
            cout << "New distance threshold " << distCut << "m" << endl;
            threshAsChange = false;
        }

    }

    // free all the allocated memory before quit
    imDisplay.release();
    imageCropp.deallocate();
    imageCheckerboard.deallocate();
    delete zed;

    return 0;
}
