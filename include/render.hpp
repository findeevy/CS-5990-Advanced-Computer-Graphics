/**
 * @file render.h
 * @brief VulkanRenderer class declaration for Accelerender - a Vulkan-based
 * renderer.
 *
 * This file contains the VulkanRenderer class declaration, which encapsulates:
 * - Vulkan instance and device creation
 * - Physical and logical GPU selection
 * - Swapchain, image views, and depth/color resources
 * - Graphics pipeline creation with shaders
 * - Command buffers and synchronization objects
 * - Rendering loop and cleanup routines
 *
 * @authors Finley Deevy, Eric Newton
 * @date 2025-11-11 (Updated)
 */

#pragma once

// ============================ //
// Includes: Standard Libraries //
// ============================ //
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// =============================== //
// Includes: Third-Party Libraries //
// =============================== //
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_beta.h>
#include <vulkan/vulkan_raii.hpp>

// =============== //
// Project Headers //
// =============== //
#include "ChronoProfiler.hpp"
#include "ProfilerUI.hpp"
#include "UniformBufferObject.hpp"
#include "Vertex.hpp"
#include "VertexHash.hpp"
#include "VulkanUtils.hpp"

// ========= //
// Constants //
// ========= //
/** @brief Initial window width in pixels. */
constexpr uint32_t WIDTH = 720;

/** @brief Initial window height in pixels. */
constexpr uint32_t HEIGHT = 540;

/** @brief File path to the 3D model used in the scene. */
const std::string MODEL_PATH = "models/statue.obj";

/** @brief File path to the texture image for the model. */
const std::string TEXTURE_PATH = "textures/statue.png";

/** @brief Vulkan validation layers enabled for debugging. */
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

/** @brief Enables validation layers if not in release mode. */
#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

/** @brief Maximum number of frames processed concurrently in the swap chain. */
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

/**
 * @class VulkanRenderer
 * @brief Encapsulates a Vulkan-based rendering engine using RAII wrappers.
 *
 * This class manages the complete lifecycle of a Vulkan renderer, including:
 * - Window creation using GLFW
 * - Vulkan instance and device setup
 * - Swap chain creation and image view management
 * - Graphics pipeline creation
 * - Command buffer recording
 * - Synchronization primitives (semaphores and fences)
 * - Resource management for buffers, textures, and uniforms
 *
 * The class uses RAII-style Vulkan handles (vk::raii) for automatic cleanup.
 *
 * @note This class assumes a single-window context.
 * @note Handles multi-frame in-flight synchronization with
 * MAX_FRAMES_IN_FLIGHT.
 *
 * @authors Finley Deevy, Eric Newton
 * @version 1.0
 * @date 2025-11-10 (Updated)
 */
class VulkanRenderer {
public:
  /**
   * @brief Runs the Vulkan renderer.
   *
   * This is the primary entry point for the renderer. It performs the following
   * steps:
   * 1. Initializes the GLFW window.
   * 2. Initializes Vulkan, including instance, device, swap chain, and
   * pipelines.
   * 3. Enters the main render loop.
   * 4. Cleans up all Vulkan and GLFW resources when finished.
   *
   * @throws std::runtime_error if any Vulkan or GLFW initialization fails.
   */
  void run();

private:
  ProfilerUI profilerUI; // Initialize here with default history size

  /** @brief RAII context for Vulkan initialization */
  vk::raii::Context context;

  /** @brief Vulkan instance handle */
  vk::raii::Instance instance = nullptr;

  /** @brief Debug messenger for validation layers */
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

  /** @brief GLFW window pointer */
  GLFWwindow *window = nullptr;

  /** @brief Selected physical GPU for rendering */
  vk::raii::PhysicalDevice physicalGPU = nullptr;

  /** @brief Logical Vulkan device */
  vk::raii::Device device = nullptr;

  /** @brief Graphics queue */
  vk::raii::Queue graphicsQueue = nullptr;

  /** @brief Features enabled on the physical device */
  vk::PhysicalDeviceFeatures GPUFeatures;

  /** @brief Presentation queue */
  vk::raii::Queue presentQueue = nullptr;

  /** @brief Surface associated with the window */
  vk::raii::SurfaceKHR surface = nullptr;

  /** @brief Swap chain object */
  vk::raii::SwapchainKHR swapChain = nullptr;

  /** @brief Swap chain images */
  std::vector<vk::raii::Image> swapChainImages;

  /** @brief Format of swap chain images */
  vk::Format swapChainImageFormat = vk::Format::eUndefined;

  /** @brief Swap chain image dimensions */
  vk::Extent2D swapChainExtent;

  /** @brief Image views for the swap chain images */
  std::vector<vk::raii::ImageView> swapChainImageViews;

  /** @brief Pipeline layout object */
  vk::raii::PipelineLayout pipelineLayout = nullptr;

  /** @brief Graphics pipeline object */
  vk::raii::Pipeline graphicsPipeline = nullptr;

  /** @brief Format of the swap chain surface */
  vk::SurfaceFormatKHR swapChainSurfaceFormat;

  /** @brief Vertex input state for the pipeline */
  vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

  /** @brief Index of the graphics queue family */
  uint32_t graphicsQueueFamilyIndex;

  /** @brief Command pool for allocating command buffers */
  vk::raii::CommandPool commandPool = nullptr;

  /** @brief Command buffers for rendering */
  std::vector<vk::raii::CommandBuffer> commandBuffers;

  /** @brief Number of samples for MSAA */
  vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;

  /** @brief Color image used for multisampling */
  vk::raii::Image colorImage = nullptr;

  /** @brief Memory backing the color image */
  vk::raii::DeviceMemory colorImageMemory = nullptr;

  /** @brief Image view for the color image */
  vk::raii::ImageView colorImageView = nullptr;

  /** @brief Number of mipmap levels for textures */
  uint32_t mipLevels;

  /** @brief Semaphores indicating image availability */
  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;

  /** @brief Semaphores signaling render completion */
  std::vector<vk::raii::Semaphore> renderFinishedSemaphores;

  /** @brief Fences to synchronize CPU and GPU */
  std::vector<vk::raii::Fence> inFlightFences;

  /** @brief Vertex buffer */
  vk::raii::Buffer vertexBuffer = nullptr;

  /** @brief Memory backing the vertex buffer */
  vk::raii::DeviceMemory vertexBufferMemory = nullptr;

  /** @brief Index buffer */
  vk::raii::Buffer indexBuffer = nullptr;

  /** @brief Memory backing the index buffer */
  vk::raii::DeviceMemory indexBufferMemory = nullptr;

  /** @brief Descriptor set layout */
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;

  /** @brief Uniform buffers for the swap chain */
  std::vector<vk::raii::Buffer> uniformBuffers;

  /** @brief Memory backing uniform buffers */
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;

  /** @brief Mapped pointers to uniform buffers */
  std::vector<void *> uniformBuffersMapped;

  /** @brief Descriptor pool */
  vk::raii::DescriptorPool descriptorPool = nullptr;

  /** @brief Descriptor sets */
  std::vector<vk::raii::DescriptorSet> descriptorSets;

  /** @brief Texture image */
  vk::raii::Image textureImage = nullptr;

  /** @brief Memory backing the texture image */
  vk::raii::DeviceMemory textureImageMemory = nullptr;

  /** @brief Image view for the texture */
  vk::raii::ImageView textureImageView = nullptr;

  /** @brief Texture sampler */
  vk::raii::Sampler textureSampler = nullptr;

  /** @brief Depth image */
  vk::raii::Image depthImage = nullptr;

  /** @brief Memory backing the depth image */
  vk::raii::DeviceMemory depthImageMemory = nullptr;

  /** @brief Image view for the depth image */
  vk::raii::ImageView depthImageView = nullptr;

  /** @brief Vertices loaded from the model */
  std::vector<Vertex> vertices;

  /** @brief Indices loaded from the model */
  std::vector<uint32_t> indices;

  /** @brief Current frame index for multi-frame rendering */
  uint32_t currentFrame = 0;

  /** @brief Flag for framebuffer resizing */
  bool framebufferResized = false;

  /** @brief Required GPU extensions */
  std::vector<const char *> gpuExtensions = {"VK_KHR_swapchain"};

  /**
   * @brief Finds a compatible GPU memory type.
   *
   * @param typeFilter Bitmask of memory types allowed by Vulkan.
   * @param properties Required memory flags (e.g., host-visible, device-local).
   * @return Index of the compatible memory type.
   */
  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties);

  /**
   * @brief Determines highest supported multi-sample count (MSAA).
   *
   * @return Highest supported sample count for color and depth buffers.
   */
  vk::SampleCountFlagBits getMaxUsableSampleCount();

  /**
   * @brief Loads a 3D model from MODEL_PATH into `vertices` and `indices`.
   *
   * @throws std::runtime_error on file I/O failure or invalid model format.
   */
  void loadModel();

  /**
   * @brief Creates depth image, allocates memory, and generates depth image
   * view.
   *
   * @throws std::runtime_error on Vulkan allocation failure.
   */
  void createDepthResources();

  /**
   * @brief Selects the first format supported by GPU based on given rules.
   *
   * @param candidates Formats to test.
   * @param tiling GPU tiling mode (optimal vs linear).
   * @param features Feature bits required (sampling/usage).
   * @return Supported format.
   */
  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features);

  /**
   * @brief Selects depth format usable as depth/stencil attachment.
   *
   * @return Depth format.
   */
  vk::Format findDepthFormat();

  /**
   * @brief Checks whether a depth format includes stencil.
   *
   * @param format Format to inspect.
   * @return true if format includes stencil, false otherwise.
   */
  bool hasStencilComponent(vk::Format format);

  /**
   * @brief Creates an image view for the texture image.
   */
  void createTextureImageView();

  /**
   * @brief Creates a texture sampler (filtering + addressing).
   */
  void createTextureSampler();

  /**
   * @brief Begins a short-lived command buffer.
   *
   * @return Unique pointer to command buffer to be manually submitted.
   */
  std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands();

  /**
   * @brief Submits one-time command buffer and frees it.
   *
   * @param commandBuffer Command buffer created via beginSingleTimeCommands().
   */
  void endSingleTimeCommands(vk::raii::CommandBuffer &commandBuffer);

  /**
   * @brief Loads texture image, uploads pixels to Vulkan image, and builds
   * mipmaps.
   */
  void createTextureImage();

  /**
   * @brief Creates MSAA color buffer + image view.
   */
  void createColorResources();

  /**
   * @brief Creates an image + GPU allocation + binds memory.
   *
   * @param width Width in pixels
   * @param height Height in pixels
   * @param mipLevels Number of mipmap levels
   * @param numSamples Sample count (MSAA)
   * @param format Pixel format (e.g. RGBA)
   * @param tiling Image tiling mode
   * @param usage Image usage flags
   * @param properties Memory requirements
   * @param image Output Vulkan image handle
   * @param imageMemory Backing device memory
   */
  void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                   vk::SampleCountFlagBits numSamples, vk::Format format,
                   vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                   vk::MemoryPropertyFlags properties, vk::raii::Image &image,
                   vk::raii::DeviceMemory &imageMemory);

  /**
   * @brief Transition GPU image layout (required for texture creation/staging).
   *
   * @param image Target image
   * @param oldLayout Original layout
   * @param newLayout Target layout
   * @param mipLevels Number of mipmap levels
   */
  void transitionImageLayout(const vk::raii::Image &image,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout, uint32_t mipLevels);

  /**
   * @brief Generates mipmaps across all mipmap levels for a texture.
   *
   * @param image Vulkan image
   * @param imageFormat Format
   * @param texWidth Width in pixels
   * @param texHeight Height in pixels
   * @param mipLevels Total mip levels
   */
  void generateMipmaps(vk::raii::Image &image, vk::Format imageFormat,
                       int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

  /**
   * @brief Copies pixel data from buffer → image.
   *
   * @param buffer Staging buffer
   * @param image Target image
   * @param width Width in pixels
   * @param height Height in pixels
   */
  void copyBufferToImage(const vk::raii::Buffer &buffer, vk::raii::Image &image,
                         uint32_t width, uint32_t height);

  /**
   * @brief Creates Vulkan descriptor pool.
   */
  void createDescriptorPool();

  /**
   * @brief Allocates and initializes descriptor sets.
   */
  void createDescriptorSets();

  /**
   * @brief Updates UBO for the current frame (camera matrices).
   *
   * @param currentImage Swapchain image index.
   */
  void updateUniformBuffer(uint32_t currentImage);

  /**
   * @brief Creates one uniform buffer per swapchain frame-in-flight.
   */
  void createUniformBuffers();

  /**
   * @brief Creates descriptor set layout (UBO + texture sampler).
   */
  void createDescriptorSetLayout();

  /**
   * @brief Copies GPU buffer → GPU buffer (e.g. staging → device-local).
   *
   * @param srcBuffer Source buffer
   * @param dstBuffer Destination buffer
   * @param size Size (bytes)
   */
  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer,
                  vk::DeviceSize size);

  /**
   * @brief Creates GPU buffer (vertex/index/uniform).
   *
   * @param size Size in bytes
   * @param usage Usage flags
   * @param properties Memory properties
   * @param buffer Output RAII VkBuffer
   * @param bufferMemory Output device memory
   */
  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties,
                    vk::raii::Buffer &buffer,
                    vk::raii::DeviceMemory &bufferMemory);

  /**
   * @brief Creates index buffer on GPU.
   */
  void createIndexBuffer();

  /**
   * @brief Creates vertex buffer on GPU.
   */
  void createVertexBuffer();

  /**
   * @brief GLFW callback for framebuffer resize (window resizing).
   *
   * @param window GLFW window
   * @param width New framebuffer width
   * @param height New framebuffer height
   */
  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height);

  /**
   * @brief Allocates command buffers from command pool.
   */
  void createCommandBuffers();

  /**
   * @brief Creates command pool (graphics queue).
   */
  void createCommandPool();

  /**
   * @brief Creates semaphores + fences to sync CPU ↔ GPU rendering.
   */
  void createSyncObjects();

  /**
   * @brief Submits command buffer and presents render image.
   *
   * @throws std::runtime_error on fence timeout (GPU hang).
   */
  void drawFrame();

  /**
   * @brief Barrier helper to transition swapchain image layout.
   *
   * @param imageIndex Image index
   * @param oldLayout Previous layout
   * @param newLayout New layout
   * @param srcAccessMask Access before
   * @param dstAccessMask Access after
   * @param srcStageMask Pipeline stage before
   * @param dstStageMask Pipeline stage after
   */
  void transition_image_layout(uint32_t imageIndex, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout,
                               vk::AccessFlags2 srcAccessMask,
                               vk::AccessFlags2 dstAccessMask,
                               vk::PipelineStageFlags2 srcStageMask,
                               vk::PipelineStageFlags2 dstStageMask);

  /**
   * @brief Records all Vulkan draw commands into a command buffer.
   *
   * @param imageIndex Swapchain image index.
   */
  void recordCommandBuffer(uint32_t imageIndex);

  /**
   * @brief Creates graphics pipeline (shaders, rasterizer, MSAA, layouts).
   */
  void createGraphicsPipeline();

  /**
   * @brief Creates a Vulkan shader module from a SPIR-V file.
   *
   * @param code Binary SPIR-V shader bytecode
   * @return ShaderModule RAII handle
   */
  vk::raii::ShaderModule createShaderModule(const std::vector<char> &code);

  /**
   * @brief Creates a Vulkan surface from GLFW window.
   */
  void createSurface();

  /**
   * @brief Sets up debug messenger if validation layers are enabled.
   */
  void setupDebugMessenger();

  /**
   * @brief Vulkan validation message callback.
   */
  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
      vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
      vk::DebugUtilsMessageTypeFlagsEXT type,
      const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *);

  /**
   * @brief Fetches required extensions (GLFW + validation layers).
   *
   * @return List of Vulkan extension names.
   */
  std::vector<const char *> getRequiredExtensions();

  /**
   * @brief Creates Vulkan instance (configures extensions, layers).
   */
  void createInstance();

  /**
   * @brief Creates image views for all swapchain images.
   */
  void createImageViews();

  /**
   * @brief Creates the swapchain (images, views, formats).
   */
  void createSwapChain();

  /**
   * @brief Chooses swapchain resolution.
   *
   * @param capabilities Surface capabilities
   * @return Swapchain extent
   */
  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities);

  /**
   * @brief Selects best present mode (FIFO override if mailbox available).
   *
   * @param availablePresentModes Supported present modes
   * @return Selected present mode
   */
  vk::PresentModeKHR chooseSwapPresentMode(
      const std::vector<vk::PresentModeKHR> &availablePresentModes);

  /**
   * @brief Selects best swapchain format (prefers SRGB).
   *
   * @param availableFormats Supported formats
   * @return Selected format
   */
  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<vk::SurfaceFormatKHR> &availableFormats);

  /**
   * @brief Selects logical GPU + queues.
   *
   * @throws std::runtime_error if GPU doesn't support swapchain extension.
   */
  void pickLogicalGPU();

  /**
   * @brief Selects physical GPU.
   *
   * @throws std::runtime_error if no suitable GPU found.
   */
  void pickPhysicalGPU();

  /**
   * @brief Recreates swapchain on window resize.
   */
  void recreateSwapChain();

  /**
   * @brief Cleans up swapchain objects (preserves pipeline / vertex buffers).
   */
  void cleanupSwapChain();

  /**
   * @brief Initializes GLFW + window.
   */
  void initWindow();

  /**
   * @brief Performs complete Vulkan initialization.
   */
  void initVulkan();

  /**
   * @brief Renders until window closes.
   */
  void mainLoop();

  /**
   * @brief Cleans up ALL Vulkan resources + GLFW.
   */
  void cleanup();
};
