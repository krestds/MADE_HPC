#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__
void pattern_filter(uint8_t* input_I,
              uint8_t* output_I,
              int num_Rows, 
              int num_Cols,
              float* filter, 
              size_t pattern_size, 
              float sum)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;

  if (px >= num_Cols || py >= num_Rows) 
  {
    return;
  }

  float c = 0.0f;

  for (int fx = 0; fx < pattern_size; fx++) {
    for (int fy = 0; fy < pattern_size; fy++) 
    {
      int imagex = px + fx - pattern_size / 2;
      int imagey = py + fy - pattern_size / 2;
      imagex = min(max(imagex, 0), num_Cols - 1);
      imagey = min(max(imagey, 0), num_Rows - 1);
      c += (filter[fy * pattern_size + fx] * input_I[imagey * num_Cols + imagex]);
    }
  }

  output_I[py * num_Cols + px] = c / sum;
}


int main() {

    int height;
    int width;
    int bpp;
    int filterSize = 3;
    double sum = 0;
    float* filter = (float *)malloc(filterSize * filterSize * sizeof(float));


    for (int i = 0; i < filterSize * filterSize; i++) {
        filter[i] = 0.1;
    }

    for (int i = 0; i < filterSize * filterSize; i++) {
        sum += filter[i];
    }


    uint8_t* image = stbi_load("rabbit_gray.jpg", &width, &height, &bpp, 1);
    uint8_t* out_Image = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    uint8_t* dev_Image;
    uint8_t* dev_Out_Image;
    float* dev_Filter;

    cudaMalloc(&dev_Image, width * height * sizeof(uint8_t));
    cudaMalloc(&dev_Out_Image, width * height * sizeof(uint8_t));
    cudaMalloc(&dev_Filter, filterSize * filterSize * sizeof(float));

    cudaMemcpy(dev_Image, image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Out_Image, out_Image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Filter, filter, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice);

    pattern_filter<<<dim3(100, 100, 1), dim3(3, 3, 1)>>>(dev_Image, dev_Out_Image, height, width, dev_Filter, filterSize, sum);

    cudaDeviceSynchronize();
    cudaMemcpy(out_Image, dev_Out_Image, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    stbi_write_png("rabbit_3x3.png", width, height, 1, out_Image, width * 1);

    free(out_Image);
    free(filter);
    cudaFree(dev_Filter);
    cudaFree(dev_Image);
    cudaFree(dev_Out_Image);
    stbi_image_free(image);

    return 0;
}