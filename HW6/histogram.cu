#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__
void create_hist(uint8_t* input_I,
              int* hist,
              int num_Rows, 
              int num_Cols)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= num_Cols || py >= num_Rows) {
      return;
  }
 
  if (input_I[py * num_Cols + px] > 255)
    printf("Err: out of range\n");
  
  atomicAdd((int *)(hist + input_I[py * num_Cols + px]), 1);  
}


int main() {

    int height;
    int width;
    int bpp;
    uint8_t* image = stbi_load("rabbit_noise.jpg", &width, &height, &bpp, 1);
    int* hist = (int *) malloc(256 * sizeof(int));
    uint8_t* dev_Image;
    int* dev_hist;

    cudaMalloc(&dev_Image, width * height * sizeof(uint8_t));
    cudaMalloc(&dev_hist, 256 * sizeof(int));

    cudaMemcpy(dev_Image, image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    create_hist<<<dim3(100, 100, 1), dim3(3, 3, 1)>>>(dev_Image, dev_hist, height, width);
    
    cudaDeviceSynchronize();
    cudaMemcpy(hist, dev_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *f = fopen("histogram.csv", "w"); 
    if (f == NULL) return -1;
    for (int i = 0; i < 256; i++) {
      fprintf(f, "%d\n", hist[i]);
    }

    fclose(f);
    free(hist);
    cudaFree(dev_Image);
    cudaFree(dev_hist);
    stbi_image_free(image);

    return 0;
}