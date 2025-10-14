# CS5990 Renderer
A cross-platform Vulkan and C++ based modern real time GPU based 3D rendering engine that we are building under the supervision of Dr.Peng Jiang as an independent study.

## Compilation:
Run the makefile after installing the [dependencies](#dependencies) to compile the shaders as well as the executable. This has been tested on both Arch Linux (NVidia silicon) and macOS (Apple silicon).

## Current Features:
- Fragment/vertex shader support.
- Window resizing.
- Texture mapping.
- Discrete graphics device selection.
- GPU/CPU memory management (staging buffer).
- Swap chain and frame buffer management.
- CPU/GPU synchronization.
- Error checking.

## Dependencies:
- [Vulkan](https://www.vulkan.org)
- [GLM](https://github.com/g-truc/glm)
- [STB](https://github.com/nothings/stb)
- [GLFW](https://www.glfw.org)

