<p align="center">
  <img src="../assets/logo.svg" width="1000" alt="AcceleRender Logo"/>
</p>

**<code>AcceleRender</code>** is a modern, Vulkan-driven real-time rendering engine written in C++20 using full RAII abstractions. <br> <br>
It manages the *entire GPU rendering pipeline* — from device selection and swap chain orchestration to shader compilation, texture streaming, and multi-frame synchronization — while exposing a clean, high-level API for rendering 3D scenes efficiently.

## 🎥 Live Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=OC2P10T2mi0" target="_blank">
    <img src="https://img.youtube.com/vi/OC2P10T2mi0/hqdefault.jpg" width="70%" alt="AcceleRender Demo Video"/>
  </a>
</p>

<p align="center">
  <strong>▶ Click to watch the real-time rendering demo!</strong>
</p>

## 💡 Technical Context

This work is part of an **independent graduate research project** (CS:5990 - Individualized Research / Programming Project) focusing on high-performance GPU rendering and real-time profiling. The project was self-proposed and approved for graduate research credit, conducted under the supervision of **Dr. Peng Jiang** within the **Iowa High Performance Computing (IOWA-HPC)** research group, where progress is presented monthly.

## 🚀 Current Features

- Real-time Vulkan renderer (RAII-managed, no manual `vkDestroy*`)
- Vertex/index buffers with staging + texture loading w/ mipmaps
- Depth buffering & MSAA (anti-aliasing)
- Automatic GPU/device selection & memory allocation
- Swap chain + framebuffer management w/ safe resize handling
- Efficient command buffer recording & CPU/GPU synchronization
- Integrated real-time profiler (frame timing)

## ⏱️ CPU Profiling

**<code>ChronoProfiler</code>** is a lightweight CPU profiler built entirely from scratch using the C++ standard library.  
It tracks **frame-by-frame CPU usage**, measures execution time for code zones, and safely handles **multi-threaded workloads** — all without external dependencies.

### Key Features
- **Scoped RAII zones:** Wrap code with `ScopedZone` or `ScopedFrame` to profile automatically.  
- **Thread-safe:** Uses thread-local storage and mutexes to merge events per frame.  
- **Aggregated stats:** Reports average, max, and total time per zone.  
- **JSON export:** Save profiling sessions for offline analysis.  

### ProfilerUI — Console Visualizer
- Displays rolling frame history as **ASCII bars**.  
- Shows **aggregated statistics** for all tracked zones.  
- Updates safely in real-time alongside your multi-threaded application.

## 🖥️ Compilation
Run the makefile after installing the [dependencies](#dependencies).  
This builds both the shaders and the executable.

Compatible with:
- Arch Linux (RTX 3050 Ti, Titan Xp)
- macOS (Apple M1)

### ⚡ Common Build Commands

```bash
# 🧹 Clean old build artifacts
make clean

# 🎨 Compile shaders only
make shaders

# 🏗️ Build the project
make

# 📊 Build with profiling enabled
make PROFILING=1

# 📚 Generate documentation
make docs
```

## 📦 Dependencies
- **[Vulkan SDK](https://www.vulkan.org)** — core rendering backend
- **[GLFW](https://www.glfw.org)** — windowing + Vulkan surface creation
- **[GLM](https://github.com/g-truc/glm)** — math library (matrices, vectors, transforms)
- **[STB Image](https://github.com/nothings/stb)** — texture loading
- **[TinyOBJLoader](https://github.com/tinyobjloader/tinyobjloader)** — mesh loading
- **[nlohmann/json](https://github.com/nlohmann/json)** — config / profiling output

## 📖 Documentation & Design

### Documentation
All engine code is fully documented using **Doxygen**, with every class, method, and data structure described in detail. Check out the live docs here:  
🌐 [AcceleRender Doxygen Documentation](https://findeevy.github.io/AcceleRender/index.html) — includes our **custom SVG logo** and complete class references.

### Design Documents
For high-level architecture and implementation overviews:  
- [VulkanRenderer Design](design/VulkanRenderer) — GPU rendering pipeline, RAII abstractions, and command buffer management  
- [ChronoProfiler Design](design/ChronoProfiler) — profiler architecture, RAII zones, multi-threaded event handling, and JSON export

## 🗿 Sources

- Statue model from **Morgan McGuire’s Computer Graphics Archive**  
  https://casual-effects.com/data  
  Used under terms specified by the archive.
