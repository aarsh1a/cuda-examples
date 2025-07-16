#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.cuh"

__global__ void gaussianBlurKernel(unsigned char *input, unsigned char *output, int width, int height)
{
    const float kernel[3][3] = {
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
        {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                sum += input[iy * width + ix] * kernel[ky + 1][kx + 1];
            }
        }
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

void applyGaussianBlur(const cv::Mat &inputImage, cv::Mat &outputImage)
{
    int width = inputImage.cols;
    int height = inputImage.rows;

    unsigned char *d_input, *d_output;

    cudaMalloc((void **)&d_input, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    cv::Mat inputImage = cv::imread("input_image.png", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cv::Mat outputImage(inputImage.size(), CV_8UC1);
    applyGaussianBlur(inputImage, outputImage);

    cv::imwrite("output_image.png", outputImage);
    return 0;
}