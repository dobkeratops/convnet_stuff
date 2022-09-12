#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

#include "SDL2/SDL.h"
#include "SDL_image.h"


int init_opencl() {
    cl_uint num_devices, i;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    cl_device_id* devices = calloc(sizeof(cl_device_id), num_devices);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    char buf[128];
    for (i = 0; i < num_devices; i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, buf, NULL);
        fprintf(stdout, "Device %s supports ", buf);

        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 128, buf, NULL);
        fprintf(stdout, "%s\n", buf);
    }

    free(devices);
}

int SCREEN_HEIGHT = 800;
int SCREEN_WIDTH = 600;
int main() {
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window *window = SDL_CreateWindow("SDL Game", 0, 0, 
  SCREEN_HEIGHT, SCREEN_WIDTH, SDL_WINDOW_HIDDEN);
  SDL_ShowWindow(window);
  SDL_Event event;
  init_opencl();
  int running = 1;
  while(running) {
    while(SDL_PollEvent(&event)) {
      if(event.type == SDL_QUIT) {
        running = 0;
      }
    }
    SDL_Delay( 32 );
  }
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}