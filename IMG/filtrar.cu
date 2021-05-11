#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

char *fileIN, *fileOUT;
unsigned char *image;
int width, height, pixelWidth; //meta info de la imagen


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
            *I_out[i*(width/kernel_X) + j/kernel_Y + c] = I[i*width + j + c];
          }
        }
      }
    }
  }
}


int main(int argc, char** argv)
{
  // Ficheros de entrada y de salida 
  if (argc == 3) { fileIN = argv[1]; fileOUT = argv[2]; }
  else { printf("Usage: ./exe fileIN fileOUT\n"); exit(0); }


  printf("Reading image...\n");
  image = stbi_load(fileIN, &width, &height, &pixelWidth, 0);
  if (!image) {
    fprintf(stderr, "Couldn't load image.\n");
     return (-1);
  }
  printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

  printf("Filtrando\n");
	//SECUENCIAL BLANCO Y NEGRO:
	for(int i=0;i<width*height*3;i=i+3){
		image[i]=image[i];
		image[i+1]=image[i+1];
		image[i+2]=0;
	}
  printf("Escribiendo\n");
  //ESCRITURA DE LA IMAGEN EN SECUENCIAL
  stbi_write_png(fileOUT,width,height,pixelWidth,image,0);

}

