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
 * @date 2025-11-10 (Updated)
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
// Project Headers
// =============== //
#include "ChronoProfiler.hpp"
#include "ProfilerUI.hpp"
#include "UniformBufferObject.hpp"
#include "Vertex.hpp"
#include "VertexHash.hpp"
#include "VulkanUtils.hpp"

// ========= //
// Constants
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

  // Private methods
  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties);
  vk::SampleCountFlagBits getMaxUsableSampleCount();
  void loadModel();
  void createDepthResources();
  vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlags features);
  vk::Format findDepthFormat();
  bool hasStencilComponent(vk::Format format);
  void createTextureImageView();
  void createTextureSampler();
  std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands();
  void endSingleTimeCommands(vk::raii::CommandBuffer &commandBuffer);
  void createTextureImage();
  void createColorResources();
  void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                   vk::SampleCountFlagBits numSamples, vk::Format format,
                   vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                   vk::MemoryPropertyFlags properties, vk::raii::Image &image,
                   vk::raii::DeviceMemory &imageMemory);
  void transitionImageLayout(const vk::raii::Image &image,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout, uint32_t mipLevels);
  void generateMipmaps(vk::raii::Image &image, vk::Format imageFormat,
                       int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
  void copyBufferToImage(const vk::raii::Buffer &buffer, vk::raii::Image &image,
                         uint32_t width, uint32_t height);
  void createDescriptorPool();
  void createDescriptorSets();
  void updateUniformBuffer(uint32_t currentImage);
  void createUniformBuffers();
  void createDescriptorSetLayout();
  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer,
                  vk::DeviceSize size);
  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties,
                    vk::raii::Buffer &buffer,
                    vk::raii::DeviceMemory &bufferMemory);
  void createIndexBuffer();
  void createVertexBuffer();
  static void framebufferResizeCallback(GLFWwindow *window, int, int);
  void createCommandBuffers();
  void createCommandPool();
  void createSyncObjects();
  void drawFrame();
  void transition_image_layout(uint32_t imageIndex, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout,
                               vk::AccessFlags2 srcAccessMask,
                               vk::AccessFlags2 dstAccessMask,
                               vk::PipelineStageFlags2 srcStageMask,
                               vk::PipelineStageFlags2 dstStageMask);
  void recordCommandBuffer(uint32_t imageIndex);
  void createGraphicsPipeline();
  vk::raii::ShaderModule createShaderModule(const std::vector<char> &code);
  void createSurface();
  void setupDebugMessenger();
  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
      vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
      vk::DebugUtilsMessageTypeFlagsEXT type,
      const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *);
  std::vector<const char *> getRequiredExtensions();
  void createInstance();
  void createImageViews();
  void createSwapChain();
  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities);
  vk::PresentModeKHR chooseSwapPresentMode(
      const std::vector<vk::PresentModeKHR> &availablePresentModes);
  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<vk::SurfaceFormatKHR> &availableFormats);
  void pickLogicalGPU();
  void pickPhysicalGPU();
  void recreateSwapChain();
  void cleanupSwapChain();
  void initWindow();
  void initVulkan();
  void mainLoop();
  void cleanup();
};
