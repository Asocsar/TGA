#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

char *fileIN, *fileOUT;
unsigned char *image;
int width, height, pixelWidth; //meta info de la imagen

#ifndef SIZE
#define SIZE 1024
#endif

#ifndef PINNED
#define PINNED 0
#endif


__global__ void MaxPooling (int height, int width, int channels, int kernel_X, int kernel_Y, int *I, int *I_out)
{
  int x = (blockIdx.x * blockDim.x + threadIdx.x) * kernel_X;
  int y = (blockIdx.y * blockDim.y + threadIdx.y) * kernel_Y;

  int max_sum = 0;

  if (x < height and y < width)
  {
    for (int i = x; (i < height) and (i < i + kernel_X); ++i)
    {
      for (int j = y; (j < width) and (j < j + kernel_Y); ++j)
      {
        int sum_chanels = I[i*width + j + 0] + I[i*width + j + 1] + I[i*width + j + 2];
        if (sum_chanels > max_sum)
        {
          for (int c = 0; c < channels; ++c)
          {
            *I_out[(i/kernel_X)*width + j/kernel_Y + c] = I[i*width + j + c];
          }
        }
      }
    }
  }
}


int main(int argc, char** argv)
{
  // Ficheros de entrada, de salida, tamaño filtrado y num GPUs
  if (argc == 3) { fileIN = argv[1]; fileOUT = argv[2]; kernelSize = argv[3]; numgpu = argv[4]}
  else { printf("Usage: ./exe fileIN fileOUT\n"); exit(0); }


  printf("Reading image...\n");
  image = stbi_load(fileIN, &width, &height, &pixelWidth, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
     return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

  int count;
  cudaGetDeviceCount(&count);

  if (count < numgpu) { printf("No hay suficientes GPUs\n"); exit(0); }
  
  /*
  nThreads_X = width/kernelSize; // 320 / 2 = 160
  nThreads_Y = height/kernelSize; // 640 / 2 = 320
  nBlocks_X = width/nThreads; //
  nBlocks_Y = height/nThreads;
  */

  nThreads_X = width%kernelSize == 0? width/kernelSize : width/kernelSize + 1;
  nThreads_Y = height%kernelSize == 0? height/kernelSize : height/kernelSize + 1;
  nThreads = nThreads_X * nThreads_Y;
  nBlocks = nThreads/SIZE;

  dim3 dimGrid(nBlocks, nBlocks, 1);
  dim3 dimBlock(nThreads, nThreads, 1);

  cudaEvent_t E0, E1, E2, E3;
  float TiempoTotal, TiempoKernel;

  //imagen original en el device
  int *d_image;
  //imagen modificada en el device
  int *d_image_out;

  //imagen modificada en el host
  int *I_out;

  cudaEventCreate(&E0);
  cudaEventCreate(&E1);
  cudaEventCreate(&E2);
  cudaEventCreate(&E3);

  // Obtener Memoria en el host de la imagen resultante
  I_out = (int*) malloc(width*height/kernelSize);
  //I_out = (int*) malloc(width*height/kernelSize*numgpu);

  // Obtiene Memoria [pinned] en el host
  //cudaMallocHost((float**)&I_out, numBytes);
  //cudaMallocHost((float**)&H_y, numBytes);   // Solo se usa para comprobar el resultado

 
  cudaEventRecord(E0, 0);
  cudaEventSynchronize(E0);
 
  // Obtener Memoria en el device de la imagen original y la resultante
  cudaMalloc((int**)&d_image, width*height);
  cudaMalloc((int**)&d_image_out, width*height/kernelSize);
  CheckCudaError((char *) "Obtener Memoria en el device", __LINE__); 

  // Copiar datos desde el host en el device 
  cudaMemcpy(d_image, image, width*height, cudaMemcpyHostToDevice);
  CheckCudaError((char *) "Copiar Datos Host --> Device", __LINE__);

  cudaEventRecord(E1, 0);
  cudaEventSynchronize(E1);

  // Ejecutar el kernel 
  MaxPooling<<<nBlocks, nThreads>>>(height, width, 3, kernelSize, kernelSize, d_image, d_image_out);
  CheckCudaError((char *) "Invocar Kernel", __LINE__);

  cudaEventRecord(E2, 0);
  cudaEventSynchronize(E2);

  // Obtener el resultado desde el host 
  // Guardamos el resultado en I_out para poder comprobar el resultado
  cudaMemcpy(I_out, d_image_out, width*height/kernelSize, cudaMemcpyDeviceToHost); 
  CheckCudaError((char *) "Copiar Datos Device --> Host", __LINE__);

  // Liberar Memoria del device 
  cudaFree(d_image); cudaFree(d_image_out);

  cudaDeviceSynchronize();

  cudaEventRecord(E3, 0);
  cudaEventSynchronize(E3);

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
 
  printf("nThreads: %d\n", nThreads);
  printf("nBlocks: %d\n", nBlocks);

  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  printf("Filtrando\n");
	//SECUENCIAL BLANCO Y NEGRO:
	/*for(int i=0;i<width*height*3;i=i+3){
		image[i]=image[i];
		image[i+1]=image[i+1];
		image[i+2]=0;
	}*/



  printf("Escribiendo\n");
  //ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT,width,height,pixelWidth,image,0);

}

