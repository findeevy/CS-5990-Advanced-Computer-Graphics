/**
 * @file render.cpp
 * @brief VulkanRenderer class implementation for Accelerender.
 *
 * This file contains the implementation of the VulkanRenderer class methods.
 *
 * @authors Finley Deevy, Eric Newton
 * @date 2025-11-11 (Updated)
 */

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#if __has_include(<tiny_obj_loader.h>)
#include <tiny_obj_loader.h>
#else
#include "../external/tinyobjloader/tiny_obj_loader.h"
#endif

#include "../include/render.hpp"

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
void VulkanRenderer::run() {
  initWindow(); // Create GLFW window + surface
  initVulkan(); // Initialize Vulkan instance, device, swapchain, pipelines
  mainLoop();   // Enter rendering loop until window closes
  cleanup();    // Destroy all Vulkan + GLFW resources
}

/**
 * @brief Finds a suitable memory type on the GPU for allocation.
 *
 * @param typeFilter Bitmask specifying which memory types are suitable.
 * @param properties Desired memory property flags (e.g., host-visible,
 * device-local).
 * @return uint32_t Index of the suitable memory type.
 * @throws std::runtime_error If no suitable memory type is found.
 *
 * @details
 * Vulkan requires you to manually select memory types when allocating buffers
 * or images. Each physical device exposes several memory types with different
 * properties. This function searches for a memory type that satisfies both
 * the 'typeFilter' (indicating allowed types for the resource) and the
 * desired memory properties.
 *
 * Example usage:
 * '''
 * uint32_t memoryType = findMemoryType(typeFilter,
 * vk::MemoryPropertyFlagBits::eDeviceLocal);
 * '''
 */
uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter,
                                        vk::MemoryPropertyFlags properties) {
  // Query memory properties supported by the physical GPU
  vk::PhysicalDeviceMemoryProperties memProperties =
      physicalGPU.getMemoryProperties();

  // Iterate through all memory types the GPU exposes
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    // Check if this memory type matches the bitmask AND desired properties
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i; // Return usable type index
    }
  }

  throw std::runtime_error("Failed to find suitable memory type!");
}

/**
 * @brief Determines the maximum usable sample count for MSAA (multisample
 * anti-aliasing)
 *
 * @return vk::SampleCountFlagBits Maximum number of samples supported by the
 * GPU for both color and depth.
 *
 * @details
 * MSAA improves visual quality by sampling a pixel multiple times to reduce
 * aliasing. This function computes the highest sample count supported for
 * both color and depth framebuffers. The result ensures that the selected
 * sample count is compatible with both image types.
 *
 * Common sample counts: 1 (no MSAA), 2, 4, 8, 16, 32, 64
 */
vk::SampleCountFlagBits VulkanRenderer::getMaxUsableSampleCount() {
  // Query device limits (contains sample count support info)
  vk::PhysicalDeviceProperties physicalDeviceProperties =
      physicalGPU.getProperties();

  // Intersection of supported color + depth sample counts
  vk::SampleCountFlags counts =
      physicalDeviceProperties.limits.framebufferColorSampleCounts &
      physicalDeviceProperties.limits.framebufferDepthSampleCounts;

  // Return highest supported MSAA level
  if (counts & vk::SampleCountFlagBits::e64)
    return vk::SampleCountFlagBits::e64;
  if (counts & vk::SampleCountFlagBits::e32)
    return vk::SampleCountFlagBits::e32;
  if (counts & vk::SampleCountFlagBits::e16)
    return vk::SampleCountFlagBits::e16;
  if (counts & vk::SampleCountFlagBits::e8)
    return vk::SampleCountFlagBits::e8;
  if (counts & vk::SampleCountFlagBits::e4)
    return vk::SampleCountFlagBits::e4;
  if (counts & vk::SampleCountFlagBits::e2)
    return vk::SampleCountFlagBits::e2;

  return vk::SampleCountFlagBits::e1; // No MSAA supported
}

/**
 * @brief Loads a 3D model from an OBJ file into vertex and index buffers.
 *
 * @details
 * Uses TinyOBJLoader to parse the OBJ file.
 * The function:
 * - Reads vertex positions and texture coordinates
 * - Flips the Y-axis of texture coordinates to match Vulkan convention
 * - Assigns a default vertex color
 * - Builds a map of unique vertices to avoid duplicates
 * - Fills the 'vertices' and 'indices' vectors for use in Vulkan buffers
 *
 * @throws std::runtime_error If the OBJ file cannot be loaded or parsed.
 *
 * @note Vulkan expects a single contiguous vertex buffer and an index buffer
 * for drawing, which is why duplicate vertices are eliminated.
 */
void VulkanRenderer::loadModel() {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  // Parses OBJ file from disk
  if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
    throw std::runtime_error(warn + err);
  }

  std::unordered_map<Vertex, uint32_t> uniqueVertices{}; // Avoid duplicates

  // Iterate over meshes / faces
  for (const auto &shape : shapes) {
    for (const auto &index : shape.mesh.indices) {

      Vertex vertex{};
      // Extract vertex position from indexed OBJ arrays
      vertex.position = {attrib.vertices[3 * index.vertex_index + 0],
                         attrib.vertices[3 * index.vertex_index + 1],
                         attrib.vertices[3 * index.vertex_index + 2]};

      // Extract texture coordinates (flip Y axis for Vulkan)
      vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                         1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};

      vertex.color = {1.0f, 1.0f, 1.0f}; // Default white vertex color

      // Insert vertex if it's new, otherwise reuse its index
      if (!uniqueVertices.contains(vertex)) {
        uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
      }

      // Push final vertex index
      indices.push_back(uniqueVertices[vertex]);
    }
  }
}

/**
 * @brief Creates depth buffer resources for the framebuffer.
 *
 * @details
 * Depth testing ensures that fragments closer to the camera overwrite
 * farther fragments. This function:
 * - Finds a supported depth format
 * - Allocates an image with device-local memory
 * - Creates an image view for the depth attachment
 *
 * @note Depth images are used in combination with a depth/stencil attachment
 * in the Vulkan render pass.
 */
void VulkanRenderer::createDepthResources() {
  vk::Format depthFormat = findDepthFormat(); // Pick supported depth format

  // Create depth image (device local = GPU memory only)
  createImage(swapChainExtent.width, swapChainExtent.height,
              1,           // no mipmaps for depth images
              msaaSamples, // match MSAA sample count
              depthFormat, vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eDepthStencilAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
              depthImageMemory);

  // Create an image view so shaders can access depth image
  depthImageView = vkutils::createImageView(device, depthImage, depthFormat,
                                            vk::ImageAspectFlagBits::eDepth, 1);
}

/**
 * @brief Finds the first supported image format from a list of candidates.
 *
 * @param candidates List of possible formats to check (e.g., depth formats).
 * @param tiling The tiling mode to test (linear or optimal).
 * @param features Required feature flags (e.g., depth/stencil attachment).
 * @return vk::Format The first compatible format found.
 *
 * @throws std::runtime_error If no supported format is found.
 *
 * @details
 * Vulkan devices vary in which formats and tiling modes they support.
 * This function queries the GPU's format properties for each candidate format
 * and checks if it satisfies the required features under the specified tiling
 * mode.
 */
vk::Format
VulkanRenderer::findSupportedFormat(const std::vector<vk::Format> &candidates,
                                    vk::ImageTiling tiling,
                                    vk::FormatFeatureFlags features) {
  // Search through the candidate formats and find the first one that satisfies
  // the required features
  auto formatIt = std::ranges::find_if(candidates, [&](auto const format) {
    // Query the physical device for format properties
    vk::FormatProperties props = physicalGPU.getFormatProperties(format);

    // Check if the format supports the requested features for the specified
    // tiling
    return (((tiling == vk::ImageTiling::eLinear) &&
             ((props.linearTilingFeatures & features) == features)) ||
            ((tiling == vk::ImageTiling::eOptimal) &&
             ((props.optimalTilingFeatures & features) == features)));
  });

  // If no compatible format is found, throw an error
  if (formatIt == candidates.end()) {
    throw std::runtime_error("Failed to find supported format!");
  }

  return *formatIt;
}

/**
 * @brief Finds a suitable depth format supported by the GPU.
 *
 * @return vk::Format A format that supports depth and stencil attachments.
 *
 * @details
 * This helper wraps findSupportedFormat() and chooses from a common set of
 * depth formats used in Vulkan rendering. The first supported format from the
 * list is returned and will later be used to create a depth buffer.
 */
vk::Format VulkanRenderer::findDepthFormat() {
  // Common depth/stencil formats prioritized by precision
  return findSupportedFormat(
      {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
       vk::Format::eD24UnormS8Uint},
      vk::ImageTiling::eOptimal,
      vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

/**
 * @brief Checks whether a given format includes a stencil component.
 *
 * @param format Vulkan image format to check.
 * @return true If the format includes a stencil component.
 * @return false Otherwise.
 *
 * @details
 * Used primarily when creating image views and pipelines that use
 * depth/stencil buffers.
 */
bool VulkanRenderer::hasStencilComponent(vk::Format format) {
  // Only these two formats have stencil components in Vulkan
  return format == vk::Format::eD32SfloatS8Uint ||
         format == vk::Format::eD24UnormS8Uint;
}

/**
 * @brief Creates an image view for the texture image.
 *
 * @details
 * Image views define how shaders will access image data (e.g., color, depth).
 * This function sets up a color image view for the loaded texture,
 * covering all mipmap levels.
 */
void VulkanRenderer::createTextureImageView() {
  // Create a standard RGBA image view for the texture, including all mip levels
  textureImageView =
      vkutils::createImageView(device, textureImage, vk::Format::eR8G8B8A8Srgb,
                               vk::ImageAspectFlagBits::eColor, mipLevels);
}

/**
 * @brief Creates a sampler object that defines how the GPU samples the
 * texture.
 *
 * @details
 * The sampler defines filtering, wrapping, and mipmap behavior when sampling
 * textures in shaders. This implementation enables anisotropic filtering and
 * repeat addressing on all axes.
 */
void VulkanRenderer::createTextureSampler() {
  // Query the physical device limits for anisotropy support
  vk::PhysicalDeviceProperties properties = physicalGPU.getProperties();

  vk::SamplerCreateInfo samplerInfo;
  samplerInfo.magFilter =
      vk::Filter::eLinear; // Linear filtering for magnification
  samplerInfo.minFilter =
      vk::Filter::eLinear; // Linear filtering for minification
  samplerInfo.mipmapMode =
      vk::SamplerMipmapMode::eLinear; // Smooth mipmap interpolation
  samplerInfo.addressModeU =
      vk::SamplerAddressMode::eRepeat; // Wrap texture coordinates
  samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.anisotropyEnable = VK_TRUE; // Enable anisotropic filtering
  samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = vk::CompareOp::eAlways;
  samplerInfo.minLod = 0.0f;                          // Minimum LOD
  samplerInfo.maxLod = static_cast<float>(mipLevels); // Maximum LOD
  samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
  samplerInfo.unnormalizedCoordinates = VK_FALSE; // Use normalized [0,1] UVs

  // Create the Vulkan sampler
  textureSampler = vk::raii::Sampler(device, samplerInfo);
}

/**
 * @brief Allocates and begins recording a one-time-use command buffer.
 *
 * @return std::unique_ptr<vk::raii::CommandBuffer> A command buffer ready to
 * record operations.
 *
 * @details
 * Single-time command buffers are used for transient GPU operations like
 * image layout transitions or buffer copies. This function allocates one
 * from the existing command pool and starts recording immediately.
 */
std::unique_ptr<vk::raii::CommandBuffer>
VulkanRenderer::beginSingleTimeCommands() {
  // Allocate a primary command buffer from the command pool
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = commandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = 1;

  std::unique_ptr<vk::raii::CommandBuffer> commandBuffer =
      std::make_unique<vk::raii::CommandBuffer>(
          std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

  // Begin recording commands for one-time submission
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer->begin(beginInfo);

  return commandBuffer;
}

/**
 * @brief Ends recording of a single-use command buffer and submits it for
 * execution.
 *
 * @param commandBuffer Reference to the command buffer being submitted.
 *
 * @details
 * Submits the command buffer to the graphics queue and waits for the GPU to
 * complete. This ensures that transient operations like buffer copies finish
 * before continuing.
 */
void VulkanRenderer::endSingleTimeCommands(
    vk::raii::CommandBuffer &commandBuffer) {
  // Finish recording commands
  commandBuffer.end();

  // Submit the command buffer to the graphics queue
  vk::SubmitInfo submitInfo{};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &*commandBuffer;

  graphicsQueue.submit(submitInfo, nullptr);
  graphicsQueue.waitIdle(); // Ensure execution is complete
}

/**
 * @brief Loads a texture image from disk, creates a Vulkan image, and uploads
 * the data to the GPU.
 *
 * @details
 * Steps:
 * 1. Verify texture file exists.
 * 2. Load pixel data using stb_image.
 * 3. Compute number of mipmap levels.
 * 4. Create a staging buffer in host-visible memory and copy pixels to it.
 * 5. Create the actual Vulkan image in device-local memory.
 * 6. Transition the image layout for transfer operations.
 * 7. Copy the texture data from the staging buffer to the image.
 * 8. Generate mipmaps for all levels.
 *
 * @throws std::runtime_error If the file cannot be loaded or texture creation
 * fails.
 */
void VulkanRenderer::createTextureImage() {
  // Check if the texture file exists
  std::ifstream testFile("textures/texture.png");
  if (!testFile.good()) {
    std::cerr << "ERROR: Texture file 'textures/texture.png' not found!"
              << std::endl;
    throw std::runtime_error("Texture file not found!");
  }
  testFile.close();

  // Load the texture using stb_image
  int texWidth, texHeight, texChannels;
  stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight,
                              &texChannels, STBI_rgb_alpha);

  if (!pixels) {
    std::cerr << "ERROR: Failed to load texture image: "
              << stbi_failure_reason() << std::endl;
    throw std::runtime_error("Failed to load texture image!");
  }

  // Compute mip levels for the texture
  mipLevels = static_cast<uint32_t>(
                  std::floor(std::log2(std::max(texWidth, texHeight)))) +
              1;

  vk::DeviceSize imageSize =
      texWidth * texHeight * 4; // RGBA8 = 4 bytes per pixel

  // Allocate staging buffer for the texture data
  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});

  createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  // Map the buffer memory and copy the pixel data
  void *data = stagingBufferMemory.mapMemory(0, imageSize);
  memcpy(data, pixels, static_cast<size_t>(imageSize));
  stagingBufferMemory.unmapMemory();

  stbi_image_free(pixels); // Free CPU-side image data

  // Create the Vulkan image in device-local memory
  createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1,
              vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransferSrc |
                  vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eSampled,
              vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage,
              textureImageMemory);

  // Transition image to the transfer destination layout
  transitionImageLayout(textureImage, vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eTransferDstOptimal, mipLevels);

  // Copy the data from the staging buffer to the GPU image
  copyBufferToImage(stagingBuffer, textureImage,
                    static_cast<uint32_t>(texWidth),
                    static_cast<uint32_t>(texHeight));

  // Generate mipmaps for the texture
  generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight,
                  mipLevels);
}

/**
 * @brief Creates color resources for multisampled rendering.
 *
 * @details
 * This function sets up a color image to be used as a multisampled color
 * attachment in MSAA rendering. The image will later be resolved to the
 * swapchain image to display the final rendered output.
 */
void VulkanRenderer::createColorResources() {
  vk::Format colorFormat =
      swapChainImageFormat; // Use the same format as the swapchain

  // Create a multisampled image with the specified width, height, and sample
  // count
  createImage(
      swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
      colorFormat, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eTransientAttachment |
          vk::ImageUsageFlagBits::eColorAttachment, // Used as color attachment
      vk::MemoryPropertyFlagBits::eDeviceLocal,     // GPU-local memory for
                                                    // efficiency
      colorImage, colorImageMemory);

  // Create an image view so shaders can access the image
  colorImageView = vkutils::createImageView(device, colorImage, colorFormat,
                                            vk::ImageAspectFlagBits::eColor, 1);
}

/**
 * @brief Creates a Vulkan image and allocates memory for it.
 *
 * @param width Width of the image in pixels.
 * @param height Height of the image in pixels.
 * @param mipLevels Number of mipmap levels.
 * @param numSamples Number of samples per pixel (for MSAA).
 * @param format Image pixel format (e.g., RGBA, depth).
 * @param tiling How image data is laid out in memory (optimal vs linear).
 * @param usage Bitmask specifying intended usage (color attachment, sampled,
 * etc.).
 * @param properties Memory properties (e.g., device local, host visible).
 * @param image Reference to store the created image handle.
 * @param imageMemory Reference to store the allocated memory handle.
 *
 * @details
 * This encapsulates Vulkan's boilerplate for creating images:
 * 1. Fill in vk::ImageCreateInfo structure with image parameters.
 * 2. Allocate memory suitable for the image usage.
 * 3. Bind memory to the image handle.
 */
void VulkanRenderer::createImage(uint32_t width, uint32_t height,
                                 uint32_t mipLevels,
                                 vk::SampleCountFlagBits numSamples,
                                 vk::Format format, vk::ImageTiling tiling,
                                 vk::ImageUsageFlags usage,
                                 vk::MemoryPropertyFlags properties,
                                 vk::raii::Image &image,
                                 vk::raii::DeviceMemory &imageMemory) {
  vk::ImageCreateInfo imageInfo{};
  imageInfo.imageType = vk::ImageType::e2D;          // 2D image
  imageInfo.format = format;                         // Pixel format
  imageInfo.extent = vk::Extent3D{width, height, 1}; // Width, height, depth=1
  imageInfo.mipLevels = mipLevels;                   // Number of mip levels
  imageInfo.arrayLayers = 1;                         // Single-layer image
  imageInfo.samples = numSamples;                    // Multisampling count
  imageInfo.tiling = tiling;                         // Memory layout
  imageInfo.usage = usage;                           // Intended usage flags
  imageInfo.sharingMode =
      vk::SharingMode::eExclusive; // Exclusive access by one queue

  // Create the Vulkan image
  image = vk::raii::Image(device, imageInfo);

  // Get memory requirements for the image
  vk::MemoryRequirements memRequirements = image.getMemoryRequirements();

  // Allocate memory based on requirements
  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits,
                     properties); // Choose suitable memory type

  imageMemory = vk::raii::DeviceMemory(device, allocInfo);

  // Bind the allocated memory to the image
  image.bindMemory(imageMemory, 0);
}

/**
 * @brief Transitions an image between different layouts.
 *
 * @param image The image to transition.
 * @param oldLayout Current layout of the image.
 * @param newLayout Target layout for the image.
 * @param mipLevels Number of mipmap levels.
 *
 * @details
 * Vulkan requires explicit image layout transitions depending on usage.
 * This function inserts a pipeline barrier into a temporary command buffer
 * to perform the layout transition.
 */
void VulkanRenderer::transitionImageLayout(const vk::raii::Image &image,
                                           vk::ImageLayout oldLayout,
                                           vk::ImageLayout newLayout,
                                           uint32_t mipLevels) {
  // Begin single-use command buffer for layout transition
  auto commandBuffer = beginSingleTimeCommands();

  // Describe the image subresources affected by the transition
  vk::ImageMemoryBarrier barrier{};
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.image = image;
  barrier.subresourceRange = vk::ImageSubresourceRange{
      vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1};

  vk::PipelineStageFlags sourceStage;
  vk::PipelineStageFlags destinationStage;

  // Determine access masks and pipeline stages based on old and new layouts
  if (oldLayout == vk::ImageLayout::eUndefined &&
      newLayout == vk::ImageLayout::eTransferDstOptimal) {
    // Undefined -> Transfer destination for copying data
    barrier.srcAccessMask = {};
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    destinationStage = vk::PipelineStageFlagBits::eTransfer;

  } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
             newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
    // Transfer destination -> Shader read (for sampling)
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    sourceStage = vk::PipelineStageFlagBits::eTransfer;
    destinationStage = vk::PipelineStageFlagBits::eFragmentShader;

  } else {
    throw std::invalid_argument("Unsupported layout transition!");
  }

  // Insert the pipeline barrier
  commandBuffer->pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr,
                                 barrier);

  // Submit the command buffer and wait for it to complete
  endSingleTimeCommands(*commandBuffer);
}

/**
 * @brief Generates mipmaps for a texture image using linear filtering.
 *
 * @param image Image for which to generate mipmaps.
 * @param imageFormat Format of the image.
 * @param texWidth Width of the texture in pixels.
 * @param texHeight Height of the texture in pixels.
 * @param mipLevels Total number of mipmap levels.
 *
 * @details
 * Mipmaps improve rendering quality and performance for textures viewed
 * at a distance. This function progressively downsamples the image level
 * by level using linear filtering and performs necessary layout transitions.
 */
void VulkanRenderer::generateMipmaps(vk::raii::Image &image,
                                     vk::Format imageFormat, int32_t texWidth,
                                     int32_t texHeight, uint32_t mipLevels) {
  // Check if the GPU supports linear blitting for the image format
  vk::FormatProperties formatProperties =
      physicalGPU.getFormatProperties(imageFormat);

  if (!(formatProperties.optimalTilingFeatures &
        vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
    throw std::runtime_error(
        "Texture image format does not support linear blitting!");
  }

  // Begin single-use command buffer for mipmap generation
  auto commandBuffer = beginSingleTimeCommands();

  vk::ImageMemoryBarrier barrier{};
  barrier.image = *image;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.subresourceRange.levelCount = 1;

  int32_t mipWidth = texWidth;
  int32_t mipHeight = texHeight;

  // Loop through each mip level and downsample
  for (uint32_t i = 1; i < mipLevels; i++) {
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

    // Transition previous level to transfer source
    commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eTransfer, {}, {},
                                   {}, barrier);

    // Configure blit from previous mip level to current
    vk::ImageBlit blit{};
    blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.srcOffsets[1] = vk::Offset3D{mipWidth, mipHeight, 1};
    blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    blit.srcSubresource.mipLevel = i - 1;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;

    blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.dstOffsets[1] = vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1,
                                      mipHeight > 1 ? mipHeight / 2 : 1, 1};
    blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    blit.dstSubresource.mipLevel = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;

    // Execute the blit command
    commandBuffer->blitImage(*image, vk::ImageLayout::eTransferSrcOptimal,
                             *image, vk::ImageLayout::eTransferDstOptimal,
                             {blit}, vk::Filter::eLinear);

    // Transition the new mip level to shader read for sampling
    barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eFragmentShader,
                                   {}, {}, {}, barrier);

    if (mipWidth > 1)
      mipWidth /= 2;
    if (mipHeight > 1)
      mipHeight /= 2;
  }

  // Transition last mip level to shader read
  barrier.subresourceRange.baseMipLevel = mipLevels - 1;
  barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
  barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
  barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

  commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eFragmentShader, {},
                                 {}, {}, barrier);

  // End command buffer and submit
  endSingleTimeCommands(*commandBuffer);
}

/**
 * @brief Copies data from a Vulkan buffer to an image.
 *
 * @details
 * This function records and submits a single-use command buffer that copies
 * pixel data from a source buffer (typically a staging buffer) into a GPU
 * image (commonly a texture or framebuffer attachment). The image must already
 * be in the 'vk::ImageLayout::eTransferDstOptimal' layout for the transfer
 * to work correctly.
 *
 * The function uses a single-time command buffer via
 * @c beginSingleTimeCommands() and @c endSingleTimeCommands(), which handle
 * allocation, submission, and cleanup.
 *
 * @param[in] buffer The Vulkan buffer containing the pixel data to copy.
 * @param[in,out] image The destination Vulkan image that will receive the data.
 * @param[in] width Width of the image in pixels.
 * @param[in] height Height of the image in pixels.
 *
 * @note The layout of the image must be transitioned to
 *       @c vk::ImageLayout::eTransferDstOptimal before calling this function.
 */
void VulkanRenderer::copyBufferToImage(const vk::raii::Buffer &buffer,
                                       vk::raii::Image &image, uint32_t width,
                                       uint32_t height) {
  // Begin recording a single-use command buffer
  std::unique_ptr<vk::raii::CommandBuffer> commandBuffer =
      beginSingleTimeCommands();

  // Define the region of the buffer and image to copy
  vk::BufferImageCopy region{};
  region.bufferOffset = 0;      // Start at the beginning of the buffer
  region.bufferRowLength = 0;   // Tightly packed rows
  region.bufferImageHeight = 0; // Tightly packed rows
  region.imageSubresource =     // Specify the layers and mip level
      vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
  region.imageOffset =
      vk::Offset3D{0, 0, 0}; // Start at top-left corner of the image
  region.imageExtent =
      vk::Extent3D{width, height, 1}; // Size of the region to copy

  // Record the buffer-to-image copy command into the command buffer
  commandBuffer->copyBufferToImage(
      buffer,                               // Source buffer
      image,                                // Destination image
      vk::ImageLayout::eTransferDstOptimal, // Current layout of the image
      {region}                              // Regions to copy
  );

  // Submit the command buffer and wait for completion
  endSingleTimeCommands(*commandBuffer);
}

/**
 * @brief Creates a Vulkan descriptor pool.
 *
 * @details
 * Descriptor pools in Vulkan manage memory for descriptor sets. Descriptor
 * sets are used to bind GPU resources like uniform buffers and textures to
 * shaders for rendering. This function sets up a pool that can allocate
 * descriptor sets for each frame in flight.
 *
 * This implementation supports two types of descriptors:
 * 1. Uniform Buffers – typically used for per-frame data like transformation
 *    matrices.
 * 2. Combined Image Samplers – used for textures in shaders.
 *
 * @note The maximum number of sets allocated from this pool is limited to
 *       MAX_FRAMES_IN_FLIGHT. Each set corresponds to one frame in flight.
 * @see createDescriptorSets() for allocation of descriptor sets from this pool.
 */
void VulkanRenderer::createDescriptorPool() {
  // Define the number of descriptors of each type in the pool
  std::array<vk::DescriptorPoolSize, 2> poolSizes = {};

  // Pool for uniform buffer descriptors
  poolSizes[0] =
      vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                             MAX_FRAMES_IN_FLIGHT // One per frame in flight
      );

  // Pool for combined image sampler descriptors (textures)
  poolSizes[1] =
      vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler,
                             MAX_FRAMES_IN_FLIGHT // One per frame in flight
      );

  // Descriptor pool creation info
  vk::DescriptorPoolCreateInfo poolInfo;
  poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  // Allows individual descriptor sets to be freed
  poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT; // Max sets in pool
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data(); // Pointer to pool sizes

  // Create the Vulkan descriptor pool
  descriptorPool = device.createDescriptorPool(poolInfo);
}

/**
 * @brief Allocates and writes Vulkan descriptor sets.
 *
 * Each descriptor set binds:
 * - A uniform buffer for per-frame transformation matrices.
 * - A texture sampler for fragment shading.
 *
 * @details The descriptor sets are allocated from the descriptor pool created
 * by 'createDescriptorPool()'. One set per frame in flight is allocated to
 * support multiple frames being processed simultaneously by the GPU.
 *
 * @see createUniformBuffers()
 * @see updateUniformBuffer()
 */
void VulkanRenderer::createDescriptorSets() {
  // Create a vector of layouts, one for each frame in flight
  // Each layout references the same descriptor set layout
  std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               *descriptorSetLayout);

  // Info struct describing how to allocate descriptor sets
  vk::DescriptorSetAllocateInfo allocInfo;
  allocInfo.descriptorPool =
      *descriptorPool; // Allocate from our descriptor pool
  allocInfo.descriptorSetCount =
      static_cast<uint32_t>(layouts.size()); // One set per frame
  allocInfo.pSetLayouts = layouts.data();    // Layouts for each set

  // Allocate the descriptor sets from the device
  descriptorSets = device.allocateDescriptorSets(allocInfo);

  // Write each descriptor set
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    // ------------------- //
    // Uniform buffer info //
    // ------------------- //
    vk::DescriptorBufferInfo bufferInfo;
    bufferInfo.buffer = *uniformBuffers[i]; // GPU buffer for this frame
    bufferInfo.offset = 0;                  // Start at beginning of buffer
    bufferInfo.range = sizeof(UniformBufferObject); // Size of data to bind

    // Prepare a write descriptor for the uniform buffer (binding 0)
    vk::WriteDescriptorSet descriptorWrite;
    descriptorWrite.dstSet = *descriptorSets[i]; // Destination descriptor set
    descriptorWrite.dstBinding = 0;              // Matches binding in shader
    descriptorWrite.dstArrayElement = 0; // First element of array (if arrayed)
    descriptorWrite.descriptorCount = 1; // Single buffer
    descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptorWrite.pBufferInfo = &bufferInfo; // Reference to buffer info

    // -------------------- //
    // Texture sampler info //
    // -------------------- //
    vk::DescriptorImageInfo imageInfo;
    imageInfo.imageLayout =
        vk::ImageLayout::eShaderReadOnlyOptimal; // Image layout for shader
    imageInfo.imageView = *textureImageView;     // Image view
    imageInfo.sampler = *textureSampler;         // Sampler

    // Prepare a write descriptor for the texture sampler (binding 1)
    vk::WriteDescriptorSet samplerWrite;
    samplerWrite.dstSet = *descriptorSets[i]; // Destination set
    samplerWrite.dstBinding = 1;              // Binding 1 in shader
    samplerWrite.dstArrayElement = 0;         // First element
    samplerWrite.descriptorCount = 1;         // Single sampler
    samplerWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    samplerWrite.pImageInfo = &imageInfo; // Reference to image info

    // Submit both writes to the device
    std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {descriptorWrite,
                                                              samplerWrite};
    device.updateDescriptorSets(descriptorWrites, {}); // Perform the updates
  }
}

/**
 * @brief Updates the uniform buffer for a specific frame.
 *
 * This function recalculates transformation matrices every frame, based on
 * elapsed time. It rotates the model, positions the camera, and updates
 * projection settings.
 *
 * @param[in] currentImage The index of the current frame (used to select
 * buffer).
 *
 * @note This method uses GLM for matrix math and assumes right-handed
 * coordinates.
 * @warning The Y-coordinate of the projection matrix is inverted for Vulkan's
 * coordinate system.
 */
void VulkanRenderer::updateUniformBuffer(uint32_t currentImage) {
  // Record the start time at the first call; static keeps it persistent
  static auto startTime = std::chrono::high_resolution_clock::now();

  // Calculate elapsed time since start in seconds
  auto currentTime = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float>(currentTime - startTime).count();

  // Create a new uniform buffer object to hold transformation matrices
  UniformBufferObject ubo{};

  // Model matrix: rotate around Z-axis over time
  ubo.model = glm::rotate(glm::mat4(1.0f),              // Identity matrix
                          time * glm::radians(90.0f),   // Rotate 90°/s
                          glm::vec3(0.0f, 0.0f, 1.0f)); // Z-axis

  // View matrix: camera positioned at (2,2,2), looking at origin
  ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),  // Eye/camera position
                         glm::vec3(0.0f, 0.0f, 0.0f),  // Look-at target
                         glm::vec3(0.0f, 0.0f, 1.0f)); // Up vector (Z-up)

  // Projection matrix: perspective projection with 45° FOV
  ubo.proj = glm::perspective(
      glm::radians(45.0f),
      static_cast<float>(swapChainExtent.width) /
          static_cast<float>(swapChainExtent.height), // Aspect ratio
      0.1f, 10.0f);                                   // Near/far planes

  // Flip Y coordinate to match Vulkan's coordinate system (inverted compared to
  // OpenGL)
  ubo.proj[1][1] *= -1;

  // Copy the uniform buffer object into the mapped memory of the current frame
  // This updates the GPU-accessible buffer immediately
  memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

/**
 * @brief Allocates and maps uniform buffers for each frame.
 *
 * Each frame in flight gets its own uniform buffer to avoid race conditions
 * between CPU and GPU. These buffers store the transformation matrices
 * updated by @ref updateUniformBuffer().
 *
 * @note Uses 'createBuffer()' helper to allocate buffer and memory.
 * @see updateUniformBuffer()
 */
void VulkanRenderer::createUniformBuffers() {
  // Clear any existing buffers or memory references before allocation
  uniformBuffers.clear();
  uniformBuffersMemory.clear();
  uniformBuffersMapped.clear();

  // Loop over each frame in flight and create a separate uniform buffer
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    // Each uniform buffer holds a UniformBufferObject (model, view, proj
    // matrices)
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    // Temporary buffer and memory handles to pass to createBuffer()
    vk::raii::Buffer buffer({});
    vk::raii::DeviceMemory bufferMem({});

    // Create the buffer: host-visible and coherent for CPU writes
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 buffer, bufferMem);

    // Store the buffer and its memory in the class vectors
    uniformBuffers.emplace_back(std::move(buffer));
    uniformBuffersMemory.emplace_back(std::move(bufferMem));

    // Map the buffer memory for CPU access and store the pointer
    uniformBuffersMapped.emplace_back(
        uniformBuffersMemory[i].mapMemory(0, bufferSize));
  }
}

/**
 * @brief Creates a Vulkan descriptor set layout for uniform buffers and texture
 * samplers.
 *
 * This layout defines how shader stages access resources (uniform buffers and
 * combined image samplers). The layout has two bindings:
 * - Binding 0: Vertex shader uniform buffer (e.g., transformation matrices)
 * - Binding 1: Fragment shader texture sampler
 *
 * @note Must be created before allocating descriptor sets.
 * @see createDescriptorPool()
 * @see createDescriptorSets()
 */
void VulkanRenderer::createDescriptorSetLayout() {
  // Step 1: Prepare descriptor set layout bindings array (two bindings)
  std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {};

  // Step 2: Define binding 0 for a uniform buffer accessed by the vertex shader
  bindings[0] = vk::DescriptorSetLayoutBinding(
      0,                                  // Binding index
      vk::DescriptorType::eUniformBuffer, // Descriptor type
      1,                                // Number of descriptors in this binding
      vk::ShaderStageFlagBits::eVertex, // Shader stage visibility
      nullptr // Optional sampler (not needed for uniform buffer)
  );

  // Step 3: Define binding 1 for a combined image sampler accessed by the
  // fragment shader
  bindings[1] = vk::DescriptorSetLayoutBinding(
      1,                                         // Binding index
      vk::DescriptorType::eCombinedImageSampler, // Descriptor type
      1, // Number of descriptors in this binding
      vk::ShaderStageFlagBits::eFragment, // Shader stage visibility
      nullptr // Optional sampler (set in descriptor write)
  );

  // Step 4: Fill in descriptor set layout creation info
  vk::DescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.bindingCount =
      static_cast<uint32_t>(bindings.size()); // Number of bindings
  layoutInfo.pBindings = bindings.data();     // Pointer to bindings array

  // Step 5: Create the descriptor set layout on the device
  descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
  // Now descriptorSetLayout can be used when creating descriptor sets
}

/**
 * @brief Copies data from one Vulkan buffer to another using a command buffer.
 *
 * @param[in,out] srcBuffer The source buffer containing data.
 * @param[in,out] dstBuffer The destination buffer to receive data.
 * @param[in] size The number of bytes to copy.
 *
 * @details
 * A temporary command buffer is allocated, commands are recorded to perform
 * the copy, and then it is submitted and waited upon. This is typically used
 * to move data from a host-visible staging buffer to a device-local buffer.
 *
 * @note This function blocks until the copy finishes (uses
 * 'graphicsQueue.waitIdle()').
 * @warning This should not be used in performance-critical paths; for large
 * transfers, batch operations are preferable.
 */
void VulkanRenderer::copyBuffer(vk::raii::Buffer &srcBuffer,
                                vk::raii::Buffer &dstBuffer,
                                vk::DeviceSize size) {
  // Step 1: Set up command buffer allocation info
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = *commandPool; // Command pool to allocate from
  allocInfo.level = vk::CommandBufferLevel::ePrimary; // Primary command buffer
  allocInfo.commandBufferCount = 1; // Allocate a single command buffer

  // Step 2: Allocate the command buffer
  auto commandBuffers = device.allocateCommandBuffers(allocInfo);
  vk::CommandBuffer commandBuffer = *commandBuffers[0];

  // Step 3: Begin recording commands in a one-time submit buffer
  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags =
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit; // Optimization hint
  commandBuffer.begin(beginInfo);

  // Step 4: Define the region of memory to copy
  vk::BufferCopy copyRegion{};
  copyRegion.size = size; // Copy the full size requested

  // Step 5: Record the buffer copy command
  commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

  // Step 6: Finish recording the command buffer
  commandBuffer.end();

  // Step 7: Set up submission info for the graphics queue
  vk::SubmitInfo submitInfo{};
  submitInfo.commandBufferCount = 1;           // Only one command buffer
  submitInfo.pCommandBuffers = &commandBuffer; // Pointer to the command buffer

  // Step 8: Submit the command buffer to the graphics queue
  graphicsQueue.submit(submitInfo, nullptr);

  // Step 9: Wait for the copy operation to complete
  graphicsQueue
      .waitIdle(); // Ensures the buffer is fully copied before returning
}

/**
 * @brief Creates a Vulkan buffer and allocates memory for it.
 *
 * @param[in] size The size of the buffer in bytes.
 * @param[in] usage Flags defining buffer purpose (e.g., vertex, index,
 * uniform).
 * @param[in] properties Memory properties (e.g., host visible, device local).
 * @param[out] buffer The resulting Vulkan buffer object.
 * @param[out] bufferMemory The associated device memory for the buffer.
 *
 * @details
 * This function encapsulates Vulkan buffer creation and memory allocation.
 * It performs the following steps:
 * 1. Fills in buffer creation info (size, usage, sharing mode).
 * 2. Creates the buffer object.
 * 3. Queries the buffer's memory requirements.
 * 4. Allocates device memory of the correct type.
 * 5. Binds the allocated memory to the buffer.
 *
 * @see findMemoryType()
 * @see copyBuffer()
 */
void VulkanRenderer::createBuffer(vk::DeviceSize size,
                                  vk::BufferUsageFlags usage,
                                  vk::MemoryPropertyFlags properties,
                                  vk::raii::Buffer &buffer,
                                  vk::raii::DeviceMemory &bufferMemory) {
  // Step 1: Fill out buffer creation info
  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.size = size;   // Size in bytes
  bufferInfo.usage = usage; // Usage flags (vertex, index, etc.)
  bufferInfo.sharingMode =
      vk::SharingMode::eExclusive; // Only used by one queue family

  // Step 2: Create the Vulkan buffer object
  buffer = vk::raii::Buffer(device, bufferInfo);

  // Step 3: Retrieve memory requirements for the buffer
  vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

  // Step 4: Allocate memory for the buffer
  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size; // Required memory size
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, properties); // Find suitable memory type

  bufferMemory = vk::raii::DeviceMemory(device, allocInfo);

  // Step 5: Bind the allocated memory to the buffer
  buffer.bindMemory(*bufferMemory, 0);
}

/**
 * @brief Creates the index buffer for drawing geometry.
 *
 * @details
 * Creates a host-visible staging buffer, copies the index data into it, then
 * creates a device-local index buffer and transfers the data using
 * 'copyBuffer()'. This ensures efficient GPU access for rendering.
 *
 * @note Index buffer allows reusing vertex data for multiple primitives.
 * @see copyBuffer()
 * @see createBuffer()
 */
void VulkanRenderer::createIndexBuffer() {
  vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

  // Create a host-visible staging buffer
  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});
  createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  // Map memory and copy index data into the staging buffer
  void *data = stagingBufferMemory.mapMemory(0, bufferSize);
  memcpy(data, indices.data(), (size_t)bufferSize);
  stagingBufferMemory.unmapMemory();

  // Create a device-local buffer for efficient GPU access
  createBuffer(bufferSize,
               vk::BufferUsageFlagBits::eTransferDst |
                   vk::BufferUsageFlagBits::eIndexBuffer,
               vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer,
               indexBufferMemory);

  // Copy data from staging buffer to device-local index buffer
  copyBuffer(stagingBuffer, indexBuffer, bufferSize);
}

/**
 * @brief Creates the vertex buffer for rendering geometry.
 *
 * @details
 * Uses a staging buffer approach similar to 'createIndexBuffer()' to ensure
 * vertex data resides in device-local memory for optimal GPU performance.
 *
 * @note Device-local memory cannot be mapped directly.
 * @warning Ensure vertex structure matches the shader input layout.
 */
void VulkanRenderer::createVertexBuffer() {
  vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

  // Create a host-visible staging buffer
  vk::raii::Buffer stagingBuffer = nullptr;
  vk::raii::DeviceMemory stagingBufferMemory = nullptr;
  createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  // Map memory and copy vertex data into staging buffer
  void *data = stagingBufferMemory.mapMemory(0, bufferSize);
  memcpy(data, vertices.data(), (size_t)bufferSize);
  stagingBufferMemory.unmapMemory();

  // Create a device-local vertex buffer
  createBuffer(bufferSize,
               vk::BufferUsageFlagBits::eVertexBuffer |
                   vk::BufferUsageFlagBits::eTransferDst,
               vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer,
               vertexBufferMemory);

  // Transfer data from staging buffer to device-local vertex buffer
  copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
}

/**
 * @brief GLFW callback to mark framebuffer resize events.
 *
 * @param[in] window Pointer to the GLFW window being resized.
 * @param[in] width Unused parameter (required by GLFW signature).
 * @param[in] height Unused parameter (required by GLFW signature).
 *
 * @details
 * Sets a flag inside VulkanRenderer indicating the swapchain must be
 * recreated due to a window resize.
 *
 * @note Checked in the render loop to handle swapchain recreation.
 */
void VulkanRenderer::framebufferResizeCallback(GLFWwindow *window, int, int) {
  auto app =
      reinterpret_cast<VulkanRenderer *>(glfwGetWindowUserPointer(window));
  app->framebufferResized = true;
}

/**
 * @brief Creates command buffers for each frame in flight.
 *
 * @details
 * Command buffers store recorded GPU commands. Each frame in flight gets
 * its own buffer to allow concurrent GPU execution.
 *
 * @note The number of command buffers is determined by MAX_FRAMES_IN_FLIGHT.
 */
void VulkanRenderer::createCommandBuffers() {
  commandBuffers.clear();

  vk::CommandBufferAllocateInfo allocInfo;
  allocInfo.commandPool = *commandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

  // Allocate command buffers from the command pool
  commandBuffers = device.allocateCommandBuffers(allocInfo);
}

/**
 * @brief Creates the Vulkan command pool.
 *
 * @details
 * Command pools manage memory for command buffers. This pool is tied to
 * the graphics queue family and allows individual buffer resets.
 *
 * @note 'eResetCommandBuffer' flag enables command buffers to be rerecorded.
 */
void VulkanRenderer::createCommandPool() {
  vk::CommandPoolCreateInfo poolInfo;
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = graphicsQueueFamilyIndex;

  // Create the command pool
  commandPool = vk::raii::CommandPool(device, poolInfo);
}

/**
 * @brief Creates synchronization objects for frame rendering.
 *
 * @details
 * Each frame requires:
 * - Semaphore signaling image availability for rendering
 * - Semaphore signaling rendering completion
 * - Fence to synchronize CPU and GPU work
 *
 * @note The number of sync objects matches MAX_FRAMES_IN_FLIGHT.
 * @warning Fences are initialized as signaled to prevent deadlock on first use.
 */
void VulkanRenderer::createSyncObjects() {
  presentCompleteSemaphores.clear();
  renderFinishedSemaphores.clear();
  inFlightFences.clear();

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    // Create semaphores for presentation and rendering
    presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());

    // Create fence initialized as signaled for first-frame safety
    inFlightFences.emplace_back(
        device, vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
  }
}

/**
 * @brief Draws a single frame in the Vulkan render loop.
 *
 * @details
 * Handles GPU-CPU synchronization, acquires swapchain image, records
 * commands, submits them, and presents the rendered image.
 *
 * Steps:
 * 1. Wait for previous frame fence
 * 2. Acquire next swapchain image
 * 3. Update uniform buffer
 * 4. Reset fence and command buffer
 * 5. Record command buffer
 * 6. Submit draw commands and signal semaphores
 * 7. Present the image
 *
 * @note Automatically recreates the swapchain if needed.
 */
void VulkanRenderer::drawFrame() {
  // Wait for the current frame fence to ensure GPU has finished
  while (
      vk::Result::eTimeout ==
      device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
    ;

  // Acquire next available swapchain image
  auto [result, imageIndex] = swapChain.acquireNextImage(
      UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);

  // Handle out-of-date swapchain
  if (result == vk::Result::eErrorOutOfDateKHR) {
    recreateSwapChain();
    return;
  }

  // Throw error on unexpected acquisition failure
  if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  // Update per-frame uniform buffer
  updateUniformBuffer(currentFrame);

  // Reset fence and command buffer for recording
  device.resetFences(*inFlightFences[currentFrame]);
  commandBuffers[currentFrame].reset();

  // Record rendering commands for this frame
  recordCommandBuffer(imageIndex);

  // Prepare submission info for graphics queue
  vk::PipelineStageFlags waitDestinationStageMask(
      vk::PipelineStageFlagBits::eColorAttachmentOutput);

  vk::SubmitInfo submitInfo;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &*presentCompleteSemaphores[currentFrame];
  submitInfo.pWaitDstStageMask = &waitDestinationStageMask;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &*commandBuffers[currentFrame];
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &*renderFinishedSemaphores[currentFrame];

  // Submit command buffer to graphics queue
  graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

  // Prepare presentation info
  vk::PresentInfoKHR presentInfoKHR;
  presentInfoKHR.waitSemaphoreCount = 1;
  presentInfoKHR.pWaitSemaphores = &*renderFinishedSemaphores[currentFrame];
  presentInfoKHR.swapchainCount = 1;
  presentInfoKHR.pSwapchains = &*swapChain;
  presentInfoKHR.pImageIndices = &imageIndex;

  // Present rendered image to the swapchain
  result = presentQueue.presentKHR(presentInfoKHR);

  // Recreate swapchain if necessary
  if (result == vk::Result::eErrorOutOfDateKHR ||
      result == vk::Result::eSuboptimalKHR || framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
  } else if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  // Advance to the next frame in flight
  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

/**
 * @brief Transitions a swapchain image from one layout to another.
 *
 * @param[in] imageIndex Index of the swapchain image
 * @param[in] oldLayout Current layout of the image
 * @param[in] newLayout Desired layout of the image
 * @param[in] srcAccessMask Memory access before barrier
 * @param[in] dstAccessMask Memory access after barrier
 * @param[in] srcStageMask Pipeline stage before barrier
 * @param[in] dstStageMask Pipeline stage after barrier
 *
 * @details
 * Inserts a pipeline barrier to synchronize memory and change image layout.
 * Necessary for proper rendering and presentation in Vulkan.
 *
 * @note Incorrect barrier settings may cause undefined behavior.
 */
void VulkanRenderer::transition_image_layout(
    uint32_t imageIndex, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
    vk::AccessFlags2 srcAccessMask, vk::AccessFlags2 dstAccessMask,
    vk::PipelineStageFlags2 srcStageMask,
    vk::PipelineStageFlags2 dstStageMask) {
  // Configure image memory barrier
  vk::ImageMemoryBarrier2 barrier;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = swapChainImages[imageIndex];
  barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = srcAccessMask;
  barrier.dstAccessMask = dstAccessMask;
  barrier.srcStageMask = srcStageMask;
  barrier.dstStageMask = dstStageMask;

  // Submit barrier to command buffer
  vk::DependencyInfo dependencyInfo;
  dependencyInfo.imageMemoryBarrierCount = 1;
  dependencyInfo.pImageMemoryBarriers = &barrier;

  commandBuffers[currentFrame].pipelineBarrier2(dependencyInfo);
}

/**
 * @brief Records rendering commands for the current frame into a command
 * buffer.
 *
 * @param[in] imageIndex Index of the swapchain image being rendered into.
 *
 * @details
 * This method performs all setup for rendering:
 *  - Inserts pipeline barriers for color/depth transitions.
 *  - Begins dynamic rendering with multiple attachments.
 *  - Binds the graphics pipeline, vertex/index buffers, and descriptor sets.
 *  - Issues the draw command.
 *  - Transitions the final image layout to present source.
 *
 * @note Uses Vulkan 1.3 dynamic rendering (no render pass object required).
 * @see vk::RenderingInfo
 * @see vk::RenderingAttachmentInfo
 */
void VulkanRenderer::recordCommandBuffer(uint32_t imageIndex) {
  // Begin recording commands for the current frame's command buffer
  commandBuffers[currentFrame].begin({});

  // --- COLOR IMAGE BARRIER ---
  // Prepare the multisampled color image for rendering output.
  vk::ImageMemoryBarrier2 colorBarrier;
  colorBarrier.srcStageMask =
      vk::PipelineStageFlagBits2::eTopOfPipe; // No previous stages
  colorBarrier.srcAccessMask = {};            // No previous access
  colorBarrier.dstStageMask = vk::PipelineStageFlagBits2::
      eColorAttachmentOutput; // Wait until color attachment output stage
  colorBarrier.dstAccessMask =
      vk::AccessFlagBits2::eColorAttachmentWrite; // Allow writing
  colorBarrier.oldLayout =
      vk::ImageLayout::eUndefined; // Previous layout unknown
  colorBarrier.newLayout =
      vk::ImageLayout::eColorAttachmentOptimal; // Ready for color attachment
  colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  colorBarrier.image = *colorImage;
  colorBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  colorBarrier.subresourceRange.baseMipLevel = 0;
  colorBarrier.subresourceRange.levelCount = 1;
  colorBarrier.subresourceRange.baseArrayLayer = 0;
  colorBarrier.subresourceRange.layerCount = 1;

  // Submit the barrier to the GPU
  vk::DependencyInfo colorDependencyInfo;
  colorDependencyInfo.imageMemoryBarrierCount = 1;
  colorDependencyInfo.pImageMemoryBarriers = &colorBarrier;
  commandBuffers[currentFrame].pipelineBarrier2(colorDependencyInfo);

  // --- SWAPCHAIN IMAGE BARRIER ---
  // Transition the swapchain image so it can be written as a color attachment
  vk::ImageMemoryBarrier2 swapchainBarrier;
  swapchainBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
  swapchainBarrier.srcAccessMask = {};
  swapchainBarrier.dstStageMask =
      vk::PipelineStageFlagBits2::eColorAttachmentOutput;
  swapchainBarrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
  swapchainBarrier.oldLayout = vk::ImageLayout::eUndefined;
  swapchainBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
  swapchainBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  swapchainBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  swapchainBarrier.image = *swapChainImages[imageIndex];
  swapchainBarrier.subresourceRange.aspectMask =
      vk::ImageAspectFlagBits::eColor;
  swapchainBarrier.subresourceRange.baseMipLevel = 0;
  swapchainBarrier.subresourceRange.levelCount = 1;
  swapchainBarrier.subresourceRange.baseArrayLayer = 0;
  swapchainBarrier.subresourceRange.layerCount = 1;

  vk::DependencyInfo swapchainDependencyInfo;
  swapchainDependencyInfo.imageMemoryBarrierCount = 1;
  swapchainDependencyInfo.pImageMemoryBarriers = &swapchainBarrier;
  commandBuffers[currentFrame].pipelineBarrier2(swapchainDependencyInfo);

  // --- DEPTH IMAGE BARRIER ---
  // Transition the depth buffer for depth testing during rendering
  vk::ImageMemoryBarrier2 depthBarrier;
  depthBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
  depthBarrier.srcAccessMask = {};
  depthBarrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                              vk::PipelineStageFlagBits2::eLateFragmentTests;
  depthBarrier.dstAccessMask =
      vk::AccessFlagBits2::eDepthStencilAttachmentRead |
      vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
  depthBarrier.oldLayout = vk::ImageLayout::eUndefined;
  depthBarrier.newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
  depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  depthBarrier.image = *depthImage;
  depthBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
  depthBarrier.subresourceRange.baseMipLevel = 0;
  depthBarrier.subresourceRange.levelCount = 1;
  depthBarrier.subresourceRange.baseArrayLayer = 0;
  depthBarrier.subresourceRange.layerCount = 1;

  vk::DependencyInfo depthDependencyInfo;
  depthDependencyInfo.imageMemoryBarrierCount = 1;
  depthDependencyInfo.pImageMemoryBarriers = &depthBarrier;
  commandBuffers[currentFrame].pipelineBarrier2(depthDependencyInfo);

  // --- CLEAR AND ATTACHMENT SETUP ---
  // Define clear values for color and depth attachments
  vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  vk::ClearValue depthClearValue = vk::ClearDepthStencilValue(1.0f, 0);

  // Setup the multisampled color attachment
  vk::RenderingAttachmentInfo colorAttachmentInfo;
  colorAttachmentInfo.imageView = *colorImageView;
  colorAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
  colorAttachmentInfo.loadOp =
      vk::AttachmentLoadOp::eClear; // Clear before rendering
  colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore; // Store result
  colorAttachmentInfo.clearValue = clearColor;

  // Define resolve attachment to handle MSAA and write to swapchain
  vk::RenderingAttachmentInfo resolveAttachmentInfo;
  resolveAttachmentInfo.imageView = *swapChainImageViews[imageIndex];
  resolveAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
  resolveAttachmentInfo.loadOp = vk::AttachmentLoadOp::eDontCare;
  resolveAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
  resolveAttachmentInfo.clearValue = clearColor;

  // Link MSAA resolve attachment to the main color attachment
  colorAttachmentInfo.resolveMode = vk::ResolveModeFlagBits::eAverage;
  colorAttachmentInfo.resolveImageView = resolveAttachmentInfo.imageView;
  colorAttachmentInfo.resolveImageLayout = resolveAttachmentInfo.imageLayout;

  // Setup depth attachment
  vk::RenderingAttachmentInfo depthAttachmentInfo;
  depthAttachmentInfo.imageView = *depthImageView;
  depthAttachmentInfo.imageLayout =
      vk::ImageLayout::eDepthStencilAttachmentOptimal;
  depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
  depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
  depthAttachmentInfo.clearValue = depthClearValue;

  // Define the rendering area and attachments for dynamic rendering
  vk::RenderingInfo renderingInfo;
  renderingInfo.renderArea = vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent);
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments = &colorAttachmentInfo;
  renderingInfo.pDepthAttachment = &depthAttachmentInfo;
  renderingInfo.pStencilAttachment = nullptr;

  // Start dynamic rendering
  commandBuffers[currentFrame].beginRendering(renderingInfo);

  // Bind the graphics pipeline to the command buffer
  commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics,
                                            *graphicsPipeline);

  // Bind vertex and index buffers
  vk::DeviceSize offsets[] = {0};
  commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, offsets);
  commandBuffers[currentFrame].bindIndexBuffer(*indexBuffer, 0,
                                               vk::IndexType::eUint32);

  // Bind descriptor sets for uniform data and textures
  commandBuffers[currentFrame].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
      *descriptorSets[currentFrame], nullptr);

  // Set dynamic viewport and scissor
  commandBuffers[currentFrame].setViewport(
      0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                      static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
  commandBuffers[currentFrame].setScissor(
      0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

  // Issue indexed draw command
  commandBuffers[currentFrame].drawIndexed(
      static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

  // End dynamic rendering
  commandBuffers[currentFrame].endRendering();

  // --- TRANSITION TO PRESENT ---
  // Transition swapchain image to presentable layout
  vk::ImageMemoryBarrier2 presentBarrier;
  presentBarrier.srcStageMask =
      vk::PipelineStageFlagBits2::eColorAttachmentOutput;
  presentBarrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
  presentBarrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
  presentBarrier.dstAccessMask = {};
  presentBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
  presentBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
  presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  presentBarrier.image = *swapChainImages[imageIndex];
  presentBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  presentBarrier.subresourceRange.baseMipLevel = 0;
  presentBarrier.subresourceRange.levelCount = 1;
  presentBarrier.subresourceRange.baseArrayLayer = 0;
  presentBarrier.subresourceRange.layerCount = 1;

  vk::DependencyInfo presentDependencyInfo;
  presentDependencyInfo.imageMemoryBarrierCount = 1;
  presentDependencyInfo.pImageMemoryBarriers = &presentBarrier;

  commandBuffers[currentFrame].pipelineBarrier2(presentDependencyInfo);

  // Finish recording the command buffer
  commandBuffers[currentFrame].end();
}

/**
 * @brief Creates the Vulkan graphics pipeline, which defines how rendering
 * operations are executed.
 *
 * @details
 * The graphics pipeline encapsulates various stages of the rendering process
 * such as:
 * - Vertex and fragment shaders
 * - Input assembly
 * - Rasterization
 * - Multisampling
 * - Depth and stencil testing
 * - Color blending
 *
 * This function reads SPIR-V shader binaries, creates shader modules, sets up
 * all pipeline states, and finally constructs a single graphics pipeline
 * using 'vk::raii::Pipeline'.
 *
 * @note The pipeline uses dynamic viewport and scissor states, meaning they
 * can be updated at draw time.
 * @throws std::runtime_error if shader files cannot be read or pipeline
 * creation fails.
 * @see createShaderModule()
 */
void VulkanRenderer::createGraphicsPipeline() {
  // Reset vertex input state
  vertexInputInfo = vk::PipelineVertexInputStateCreateInfo{};

  // Read SPIR-V shader binaries from disk
  std::vector<char> vertShaderCode = vkutils::readFile("shaders/vert.spv");
  std::vector<char> fragShaderCode = vkutils::readFile("shaders/frag.spv");

  // Create Vulkan shader modules
  vk::raii::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
  vk::raii::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

  // Setup shader stages
  vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
  vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
  vertShaderStageInfo.module = *vertShaderModule;
  vertShaderStageInfo.pName = "main";

  vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
  fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
  fragShaderStageInfo.module = *fragShaderModule;
  fragShaderStageInfo.pName = "main";

  vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

  // Setup input assembly for triangles
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
  inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

  // Get vertex input descriptions
  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();

  vertexInputInfo = vk::PipelineVertexInputStateCreateInfo(
      vk::PipelineVertexInputStateCreateFlags(), 1, &bindingDescription,
      static_cast<uint32_t>(attributeDescriptions.size()),
      attributeDescriptions.data());

  // Configure viewport and scissor
  vk::PipelineViewportStateCreateInfo viewportState;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  // Configure rasterization (triangle fill, back-face culling, CCW front)
  vk::PipelineRasterizationStateCreateInfo rasterizer;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = vk::PolygonMode::eFill;
  rasterizer.cullMode = vk::CullModeFlagBits::eBack;
  rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.lineWidth = 1.0f;

  // Enable sample shading if supported
  vk::PhysicalDeviceFeatures supportedFeatures = physicalGPU.getFeatures();
  bool sampleRateShadingSupported = supportedFeatures.sampleRateShading;

  vk::PipelineMultisampleStateCreateInfo multisampling;
  multisampling.rasterizationSamples = msaaSamples;
  multisampling.sampleShadingEnable =
      sampleRateShadingSupported ? VK_TRUE : VK_FALSE;
  multisampling.minSampleShading = sampleRateShadingSupported ? 0.2f : 1.0f;

  // Configure depth/stencil testing
  vk::PipelineDepthStencilStateCreateInfo depthStencil;
  depthStencil.depthTestEnable = vk::True;
  depthStencil.depthWriteEnable = vk::True;
  depthStencil.depthCompareOp = vk::CompareOp::eLess;
  depthStencil.depthBoundsTestEnable = vk::False;
  depthStencil.stencilTestEnable = VK_FALSE;

  // Configure color blending (no blending)
  vk::PipelineColorBlendAttachmentState colorBlendAttachment;
  colorBlendAttachment.blendEnable = VK_FALSE;
  colorBlendAttachment.colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

  vk::PipelineColorBlendStateCreateInfo colorBlending;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;

  // Dynamic states: viewport and scissor
  std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                 vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicState;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  // Create pipeline layout (descriptor sets)
  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

  // Specify formats for dynamic rendering
  vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo;
  pipelineRenderingCreateInfo.colorAttachmentCount = 1;
  pipelineRenderingCreateInfo.pColorAttachmentFormats =
      &swapChainSurfaceFormat.format;
  pipelineRenderingCreateInfo.depthAttachmentFormat = findDepthFormat();

  // Assemble full graphics pipeline create info
  vk::GraphicsPipelineCreateInfo pipelineInfo;
  pipelineInfo.pNext = &pipelineRenderingCreateInfo;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = *pipelineLayout;
  pipelineInfo.renderPass = nullptr; // Dynamic rendering, no render pass

  // Create the graphics pipeline
  graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}

/**
 * @brief Creates a Vulkan shader module from SPIR-V bytecode.
 *
 * @param code Binary vector containing the SPIR-V shader code.
 * @return vk::raii::ShaderModule Handle to the created shader module.
 * @throws std::runtime_error If shader creation fails.
 * @note Shader modules must be destroyed before the device is destroyed.
 */
vk::raii::ShaderModule
VulkanRenderer::createShaderModule(const std::vector<char> &code) {
  // Setup creation info for Vulkan shader module
  vk::ShaderModuleCreateInfo createInfo;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
  return vk::raii::ShaderModule(device, createInfo);
}

/**
 * @brief Creates a Vulkan surface for rendering to a GLFW window.
 *
 * @details
 * This function integrates GLFW with Vulkan by creating a VkSurfaceKHR
 * object. The surface acts as the bridge between Vulkan and the windowing
 * system.
 *
 * @throws std::runtime_error If surface creation fails.
 * @note Required before swapchain creation.
 */
void VulkanRenderer::createSurface() {
  VkSurfaceKHR _surface;
  // GLFW helper creates the platform-specific surface
  if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
    throw std::runtime_error("Failed to create window surface!");
  }
  // Wrap raw Vulkan surface with RAII handle
  surface = vk::raii::SurfaceKHR(instance, _surface);
}

/**
 * @brief Sets up a Vulkan debug messenger for validation layer output.
 *
 * @details
 * The debug messenger captures Vulkan validation layer messages such as:
 * - General information
 * - Performance warnings
 * - Validation errors
 *
 * It prints relevant warnings and errors to standard error.
 *
 * @note Only active if 'enableValidationLayers' is true.
 */
void VulkanRenderer::setupDebugMessenger() {
  if (!enableValidationLayers) {
    return; // Skip creation entirely if validation layers are disabled
  }

  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
  // Specify which severity levels we want to receive (everything except info)

  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  // Specify what type of messages we want to receive (general, perf,
  // validation)

  vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT;
  debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags;
  debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags;
  debugUtilsMessengerCreateInfoEXT.pfnUserCallback = debugCallback;
  // Assign callback function that will handle validation messages

  debugMessenger =
      instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
  // Create debug messenger using RAII wrapper, no manual destruction needed
}

/**
 * @brief Callback function used by Vulkan's debug messenger to log messages.
 *
 * @param severity The severity level of the message (Error, Warning, Info,
 * Verbose).
 * @param type The type of Vulkan message (General, Performance, Validation).
 * @param pCallbackData Contains the actual message text and identifiers.
 * @param pUserData Optional user data (unused here).
 *
 * @return Always returns 'vk::False' to indicate no interception of the call.
 *
 * @note Prints only warnings and errors to standard error.
 */
VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanRenderer::debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *) {
  if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
      severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    std::cerr << "validation layer: type " << vk::to_string(type)
              << " msg: " << pCallbackData->pMessage << std::endl;
    // Print only warnings and errors to stderr
  }
  return vk::False; // Tell Vulkan: "don't stop execution, just log it"
}

/**
 * @brief Retrieves all required Vulkan instance extensions.
 *
 * @details
 * - GLFW provides a list of extensions needed for window-surface interaction.
 * - Adds Vulkan extensions for validation and physical device querying.
 *
 * @return A vector of C-style strings containing required extension names.
 * @see setupDebugMessenger()
 * @note The 'VK_KHR_get_physical_device_properties2' extension is always
 * added.
 */
std::vector<const char *> VulkanRenderer::getRequiredExtensions() {
  uint32_t glfwExtensionCount = 0;
  auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  // GLFW returns platform-specific instance extensions needed for surfaces

  std::vector<const char *> extensions(glfwExtensions,
                                       glfwExtensions + glfwExtensionCount);
  // Copy GLFW extensions into our vector

  if (enableValidationLayers) {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
    // Enable debug utils extension if validation is active
  }

  extensions.push_back(vk::KHRGetPhysicalDeviceProperties2ExtensionName);
  // Always add extension required for physical device querying (Vulkan 1.1+)

  return extensions; // Return full list for instance creation
}

/**
 * @brief Creates the Vulkan instance used to initialize the rendering
 * context.
 *
 * This function configures and initializes the Vulkan instance, which is the
 * foundation of any Vulkan application. It sets up the application
 * information, validation layers, and required extensions, then creates the
 * Vulkan instance handle using 'vk::raii::Instance'.
 *
 * @details
 * The process consists of:
 * - Defining application information ('vk::ApplicationInfo').
 * - Checking and enabling validation layers if required.
 * - Checking and enabling instance extensions (including Apple/MoltenVK
 * specifics).
 * - Creating the Vulkan instance with all validated layers and extensions.
 *
 * @throws std::runtime_error if a required layer or extension is not
 * supported.
 *
 * @see vk::ApplicationInfo
 * @see vk::InstanceCreateInfo
 * @see vk::raii::Instance
 * @see getRequiredExtensions()
 */
void VulkanRenderer::createInstance() {
  vk::ApplicationInfo appInfo("CS-5990 Renderer", VK_MAKE_VERSION(1, 0, 0),
                              "No Engine", VK_MAKE_VERSION(1, 0, 0),
                              VK_API_VERSION_1_3);
  // Define app metadata + requested Vulkan API version

  std::vector<const char *> requiredLayers;
  if (enableValidationLayers) {
    requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    // Only request validation layer if debugging is enabled
  }

  auto layerProperties = context.enumerateInstanceLayerProperties();
  // Query available validation layers on the system

  for (auto const &requiredLayer : requiredLayers) {
    if (std::none_of(layerProperties.begin(), layerProperties.end(),
                     [requiredLayer](auto const &layerProperty) {
                       return strcmp(layerProperty.layerName, requiredLayer) ==
                              0;
                     })) {
      throw std::runtime_error("Required layer not supported: " +
                               std::string(requiredLayer));
      // Error out if a requested validation layer doesn't exist
    }
  }

  auto requiredExtensions = getRequiredExtensions();
  // Retrieve required instance extensions (GLFW + debug + device props)

#if defined(__APPLE__)
  requiredExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  // MoltenVK requires portability extension due to non-conformance
#endif

  auto extensionProperties = context.enumerateInstanceExtensionProperties();
  // Query system-supported extensions for validation

  for (auto const &requiredExtension : requiredExtensions) {
    if (std::none_of(extensionProperties.begin(), extensionProperties.end(),
                     [requiredExtension](auto const &extensionProperty) {
                       return strcmp(extensionProperty.extensionName,
                                     requiredExtension) == 0;
                     })) {
      throw std::runtime_error("Required extension not supported: " +
                               std::string(requiredExtension));
      // Throw explicit error if extension missing (prevents silent failure)
    }
  }

  vk::InstanceCreateInfo createInfo;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
  createInfo.ppEnabledLayerNames = requiredLayers.data();
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(requiredExtensions.size());
  createInfo.ppEnabledExtensionNames = requiredExtensions.data();
  // Populate instance creation info with layers + extensions

#if defined(__APPLE__)
  createInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
  // Required by MoltenVK to allow portable GPU selection
#endif

  instance = vk::raii::Instance(context, createInfo);
  // Create actual Vulkan instance (RAII handles destruction)
}

/**
 * @brief Creates image views for each image in the swap chain.
 *
 * Each image view wraps a raw swap chain image, allowing it to be accessed
 * in the pipeline. This function clears existing views and rebuilds them
 * based on the current swap chain configuration.
 *
 * @note Image views are used by framebuffers and render passes.
 *
 * @see vkutils::createImageView
 */
void VulkanRenderer::createImageViews() {
  swapChainImageViews.clear(); // Remove old views
  swapChainImageViews.reserve(swapChainImages.size());
  // Reserve space to avoid reallocations

  for (auto &image : swapChainImages) {
    swapChainImageViews.emplace_back(
        vkutils::createImageView(device, image, swapChainImageFormat,
                                 vk::ImageAspectFlagBits::eColor, 1));
    // Create an image view for each swapchain image (used by render passes)
  }
}

/**
 * @brief Creates the Vulkan swap chain used for presenting images to the
 * screen.
 *
 * This function queries the surface capabilities and formats, chooses an
 * optimal configuration, and creates the swap chain with appropriate image
 * count, format, extent, and presentation mode.
 *
 * @details
 * - Queries available formats and presentation modes.
 * - Chooses a suitable surface format and extent.
 * - Creates the swap chain using 'vk::raii::SwapchainKHR'.
 * - Retrieves and wraps all swap chain images in RAII handles.
 *
 * @throws std::runtime_error if Vulkan swap chain creation fails.
 *
 * @see vk::SwapchainCreateInfoKHR
 * @see chooseSwapSurfaceFormat()
 * @see chooseSwapPresentMode()
 * @see chooseSwapExtent()
 */
void VulkanRenderer::createSwapChain() {
  auto surfaceCapabilities = physicalGPU.getSurfaceCapabilitiesKHR(*surface);
  // Query surface capabilities (image limits, transform, usage, etc.)

  auto chosenSurfaceFormat =
      chooseSwapSurfaceFormat(physicalGPU.getSurfaceFormatsKHR(*surface));
  swapChainImageFormat = chosenSurfaceFormat.format;
  swapChainSurfaceFormat = chosenSurfaceFormat;
  auto swapChainColorSpace = chosenSurfaceFormat.colorSpace;
  // Store chosen format + color space

  swapChainExtent = chooseSwapExtent(surfaceCapabilities);
  // Determine swapchain resolution (framebuffer size)

  auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
  // Request triple buffering (if allowed)

  minImageCount = (surfaceCapabilities.maxImageCount > 0 &&
                   minImageCount > surfaceCapabilities.maxImageCount)
                      ? surfaceCapabilities.maxImageCount
                      : minImageCount;
  // Clamp to maximum supported image count

  auto presentMode =
      chooseSwapPresentMode(physicalGPU.getSurfacePresentModesKHR(*surface));
  // Choose present mode (Mailbox preferred)

  vk::SwapchainCreateInfoKHR swapChainCreateInfo;
  swapChainCreateInfo.surface = *surface;
  swapChainCreateInfo.minImageCount = minImageCount;
  swapChainCreateInfo.imageFormat = swapChainImageFormat;
  swapChainCreateInfo.imageColorSpace = swapChainColorSpace;
  swapChainCreateInfo.imageExtent = swapChainExtent;
  swapChainCreateInfo.imageArrayLayers = 1;
  swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
  swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
  swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
  swapChainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
  swapChainCreateInfo.presentMode = presentMode;
  swapChainCreateInfo.clipped = true;
  // Fill out swapchain creation config

  swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
  // Create swap chain using RAII wrapper

  swapChainImages.clear();
  auto images = swapChain.getImages();
  swapChainImages.reserve(images.size());
  // Retrieve created swapchain images

  for (auto &image : images) {
    swapChainImages.emplace_back(device, image);
    // Wrap images in RAII handles for lifetime safety
  }
}

/**
 * @brief Chooses the swap chain image extent (resolution).
 *
 * @param capabilities Surface capabilities including min/max extents.
 * @return vk::Extent2D The chosen swap chain resolution.
 *
 * @details
 * If the surface defines a current extent (i.e., not resizable), it returns
 * that. Otherwise, it queries the framebuffer size from GLFW and clamps it
 * between the allowed min and max extents.
 *
 * @see vk::SurfaceCapabilitiesKHR
 * @see glfwGetFramebufferSize
 */
vk::Extent2D VulkanRenderer::chooseSwapExtent(
    const vk::SurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities
        .currentExtent; // If fixed (non-resizable surface), use it
  }

  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  // Query pixel size of the framebuffer (not window size)

  return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width),
          std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height)};
  // Clamp between allowed min/max constraints
}

/**
 * @brief Chooses the best available present mode for the swap chain.
 *
 * @param availablePresentModes List of presentation modes supported by the
 * device.
 * @return vk::PresentModeKHR The selected presentation mode.
 *
 * @details
 * Prefers 'eMailbox' (triple buffering, low latency) if available.
 * Falls back to 'eFifo' (VSync) as a guaranteed fallback.
 *
 * @see vk::PresentModeKHR
 */
vk::PresentModeKHR VulkanRenderer::chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  for (const auto &availablePresentMode : availablePresentModes) {
    if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
      return availablePresentMode; // Low latency triple-buffering
    }
  }
  return vk::PresentModeKHR::eFifo; // Guaranteed available (VSync)
}

/**
 * @brief Chooses the most appropriate surface format for rendering.
 *
 * @param availableFormats List of formats supported by the surface.
 * @return vk::SurfaceFormatKHR The chosen surface format.
 *
 * @details
 * Prefers BGRA8 with SRGB color space (standard for color accuracy).
 * Falls back to the first available format if preference is unavailable.
 *
 * @see vk::SurfaceFormatKHR
 */
vk::SurfaceFormatKHR VulkanRenderer::chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
  for (const auto &availableFormat : availableFormats) {
    if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
        availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return availableFormat; // Preferred color format
    }
  }
  return availableFormats[0]; // Otherwise fallback to first supported
}

/**
 * @brief Selects and creates a logical GPU device for Vulkan rendering.
 *
 * This function identifies queue families that support graphics and
 * presentation, checks for feature support (including sample rate shading and
 * dynamic state features), and creates a logical device ('vk::raii::Device')
 * with the necessary queues.
 *
 * @details
 * The procedure includes:
 * - Enumerating all queue families from the selected physical GPU.
 * - Finding indices for graphics and present queue families.
 * - Handling cases where graphics and present capabilities exist in separate
 * queues.
 * - Setting up required device queues and enabling Vulkan 1.3 and extended
 * dynamic state features.
 * - Creating logical device and retrieving graphics/present queue handles.
 *
 * @throws std::runtime_error if no suitable graphics or present queue
 * families are found.
 *
 * @see vk::PhysicalDevice
 * @see vk::DeviceQueueCreateInfo
 * @see vk::DeviceCreateInfo
 * @see vk::StructureChain
 */
void VulkanRenderer::pickLogicalGPU() {
  std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
      physicalGPU.getQueueFamilyProperties();
  // Retrieve all queue families supported by the physical GPU

  uint32_t graphicsIndex = static_cast<uint32_t>(queueFamilyProperties.size());
  uint32_t presentIndex = static_cast<uint32_t>(queueFamilyProperties.size());
  // Use invalid sentinel (size) to detect later if nothing is found

  for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
    if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
      // Queue supports graphics operations

      if (graphicsIndex == queueFamilyProperties.size()) {
        graphicsIndex = i; // First graphics-capable queue found
      }
      if (physicalGPU.getSurfaceSupportKHR(i, *surface)) {
        // If same queue also supports presentation, we’re good — stop searching
        graphicsIndex = i;
        presentIndex = i;
        break;
      }
    }
  }

  if (presentIndex == queueFamilyProperties.size()) {
    // If no combined graphics+present queue, search separately for a present
    // queue
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
      if (physicalGPU.getSurfaceSupportKHR(i, *surface)) {
        presentIndex = i;
        break;
      }
    }
  }

  if (graphicsIndex == queueFamilyProperties.size()) {
    // No graphics queue found earlier? Try again without presentation
    // requirement
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
      if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        graphicsIndex = i;
        break;
      }
    }
  }

  if ((graphicsIndex == queueFamilyProperties.size()) ||
      (presentIndex == queueFamilyProperties.size())) {
    // If still invalid, Vulkan cannot render — stop execution
    throw std::runtime_error("No graphics or present queue family found!");
  }

  graphicsQueueFamilyIndex = graphicsIndex;
  std::set<uint32_t> uniqueQueueFamilies = {graphicsIndex, presentIndex};
  // Use a set so graphics/present queue isn't duplicated if they are the same

  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
  float queuePriority = 1.0f; // Highest priority for device queues

  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCreateInfo;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1; // Only request one queue from this family
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  vk::PhysicalDeviceFeatures supportedFeatures = physicalGPU.getFeatures();
  bool sampleRateShadingSupported = supportedFeatures.sampleRateShading;
  // Check if MSAA sample shading exists but only enable if available

  vk::StructureChain<vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan13Features,
                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
      featureChain;
  // Structure chain allows enabling multiple feature structs

  if (sampleRateShadingSupported) {
    featureChain.get<vk::PhysicalDeviceFeatures2>().features.sampleRateShading =
        VK_TRUE; // Only enable MSAA shading if supported
  }

  featureChain.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering =
      true;
  featureChain.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 =
      true;
  // Enable Vulkan 1.3 advanced rendering & sync model

  featureChain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
      .extendedDynamicState = true;
  // Allows dynamic state changes without recreating pipeline

  vk::DeviceCreateInfo deviceCreateInfo;
  deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
  deviceCreateInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
  deviceCreateInfo.enabledExtensionCount =
      static_cast<uint32_t>(gpuExtensions.size());
  deviceCreateInfo.ppEnabledExtensionNames = gpuExtensions.data();
  // Setup the logical device configuration

  device = vk::raii::Device(physicalGPU, deviceCreateInfo);
  // Logical device creation — now Vulkan can submit work

  graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
  presentQueue = vk::raii::Queue(device, presentIndex, 0);
  // Acquire queue handles (0 = first queue of that family)
}

/**
 * @brief Selects a suitable physical GPU that supports Vulkan 1.3 and
 * required extensions.
 *
 * This function enumerates all available physical devices and chooses the
 * first one meeting the following criteria:
 * - Supports Vulkan API version 1.3 or higher.
 * - Has at least one queue family that supports graphics operations.
 * - Supports all required device extensions ('gpuExtensions').
 *
 * Once a suitable GPU is found, it sets 'physicalGPU' and determines the
 * maximum usable MSAA sample count via 'getMaxUsableSampleCount()'.
 *
 * @throws std::runtime_error if no GPU meets the Vulkan 1.3 and extension
 * requirements.
 *
 * @see vk::raii::PhysicalDevice
 * @see vk::PhysicalDeviceProperties
 * @see vk::QueueFamilyProperties
 * @see getMaxUsableSampleCount()
 */
void VulkanRenderer::pickPhysicalGPU() {
  std::vector<vk::raii::PhysicalDevice> gpus =
      instance.enumeratePhysicalDevices();
  // Enumerate all Vulkan-capable GPUs on the system

  for (auto const &gpu : gpus) {
    auto queueFamilies = gpu.getQueueFamilyProperties();
    // Check what queues this GPU supports

    bool isSuitable = gpu.getProperties().apiVersion >= VK_API_VERSION_1_3;
    // Require Vulkan 1.3 minimum

    const auto qfpIter =
        std::find_if(queueFamilies.begin(), queueFamilies.end(),
                     [](vk::QueueFamilyProperties const &qfp) {
                       return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) !=
                              static_cast<vk::QueueFlags>(0);
                     });
    isSuitable = isSuitable && (qfpIter != queueFamilies.end());
    // Must have at least one graphics-capable queue

    auto extensions = gpu.enumerateDeviceExtensionProperties();
    bool found = true;
    for (auto const &extension : gpuExtensions) {
      auto extensionIter = std::find_if(
          extensions.begin(), extensions.end(), [extension](auto const &ext) {
            return strcmp(ext.extensionName, extension) == 0;
          });
      found = found && extensionIter != extensions.end();
    }
    // Check required device extensions (e.g., swapchain support)

    isSuitable = isSuitable && found;

    if (isSuitable) {
      physicalGPU = gpu;                       // Save selection
      msaaSamples = getMaxUsableSampleCount(); // Compute maximum MSAA level
      return;                                  // Exit after first valid GPU
    }
  }

  throw std::runtime_error("Failed to find a GPU that supports Vulkan 1.3!");
  // No GPU met the criteria — terminate program
}

/**
 * @brief Recreates the Vulkan swap chain when the window is resized or
 * invalidated.
 *
 * This function handles cases where the swap chain must be rebuilt — for
 * instance, when the window is resized, minimized, or when the swap chain
 * becomes out-of-date. It waits for valid framebuffer dimensions, waits for
 * the device to be idle, cleans up old swap chain resources, and then
 * recreates all resources dependent on the swap chain.
 *
 * @note The swap chain is central to Vulkan rendering, as it manages frame
 * buffers used for presentation to the screen. Rebuilding it ensures correct
 * synchronization and image presentation.
 *
 * @see cleanupSwapChain()
 * @see createSwapChain()
 * @see createImageViews()
 * @see createColorResources()
 * @see createDepthResources()
 * @see createCommandBuffers()
 * @see createSyncObjects()
 */
void VulkanRenderer::recreateSwapChain() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  // Ask GLFW for the framebuffer (actual drawable area), not window size

  while (width < 1 || height < 1) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents(); // Pause until window regains a valid resolution
  }

  device.waitIdle();  // Ensure GPU is not using old swapchain resources
  cleanupSwapChain(); // Release old swap chain resources

  createSwapChain();      // Make new swap chain
  createImageViews();     // Create views for each swap chain image
  createColorResources(); // Recreate MSAA color attachments
  createDepthResources(); // Recreate depth buffer
  createCommandBuffers(); // Re-record rendering command buffers
  createSyncObjects();    // Recreate semaphores/fences
}

/**
 * @brief Cleans up all Vulkan resources associated with the current swap
 * chain.
 *
 * This function safely releases all image views, color attachments, and swap
 * chain objects before the swap chain is rebuilt or destroyed. It ensures no
 * dangling references remain.
 *
 * @warning Must be called only when the device is idle to avoid undefined
 * behavior.
 *
 * @see recreateSwapChain()
 */
void VulkanRenderer::cleanupSwapChain() {
  colorImageView = nullptr;   // Destroy view first
  colorImage = nullptr;       // Destroy color attachment
  colorImageMemory = nullptr; // Free GPU memory holding the image

  swapChainImageViews.clear(); // Destroy all image views
  swapChain = nullptr;         // Destroy the swap chain itself
}

/**
 * @brief Initializes a GLFW window for Vulkan rendering.
 *
 * This sets up GLFW with Vulkan API support disabled (no OpenGL context),
 * creates a resizable window, and assigns the window's user pointer to this
 * class instance for callback handling.
 *
 * @note GLFW is used only for window management here. Vulkan handles
 * rendering.
 *
 * @see framebufferResizeCallback()
 */
void VulkanRenderer::initWindow() {
  glfwInit(); // Initialize GLFW (window system)

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  // Tell GLFW not to create an OpenGL context

  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  // Enable window resizing — important for swap chain recreation

  window = glfwCreateWindow(WIDTH, HEIGHT, "Accelerender", nullptr, nullptr);
  // Create actual window

  glfwSetWindowUserPointer(window, this);
  // Attach renderer instance pointer → used in GLFW callbacks

  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  // Register callback when window is resized
}

/**
 * @brief Initializes all Vulkan components required for rendering.
 *
 * This function follows the standard Vulkan setup sequence, from instance
 * creation to swap chain setup, pipeline creation, descriptor management, and
 * synchronization setup.
 *
 * @details
 * The sequence of function calls is critical, as each step depends on prior
 * resources:
 * - Vulkan instance and debug messenger creation.
 * - Physical and logical device selection.
 * - Swap chain and rendering resources setup.
 * - Pipeline and buffer preparation.
 * - Descriptor sets and synchronization primitives.
 *
 * @throws std::runtime_error If any Vulkan initialization step fails.
 */
void VulkanRenderer::initVulkan() {
  createInstance();            // Vulkan instance
  setupDebugMessenger();       // Validation layers callback
  createSurface();             // Create window surface (GLFW → Vulkan)
  pickPhysicalGPU();           // Select discrete GPU
  pickLogicalGPU();            // Create logical device + queues
  createSwapChain();           // Frame presentation system
  createImageViews();          // Views for each swapchain image
  createColorResources();      // MSAA render target
  createDescriptorSetLayout(); // Descriptors: UBOs + textures
  createGraphicsPipeline();    // Shader + pipeline configuration
  createCommandPool();         // Memory pool used to allocate command buffers
  createDepthResources();      // Depth buffer
  createTextureImage();        // Load texture from disk
  createTextureImageView();    // Image view for sampling
  createTextureSampler();      // Texture filtering sampler
  loadModel();                 // Load vertex/index data from model
  createVertexBuffer();        // Upload vertices to GPU
  createIndexBuffer();         // Upload indices to GPU
  createUniformBuffers();      // Allocate per-frame UBOs
  createDescriptorPool();      // Pool for descriptor sets
  createDescriptorSets();      // Allocate + write descriptor sets
  createCommandBuffers();      // Build render command buffers
  createSyncObjects();         // Semaphores/fences for frame sync
}

/**
 * @brief Runs the main application loop.
 *
 * Polls window events and continuously renders frames until the window is
 * closed. Profiles CPU time per frame and outputs live ASCII visualization
 * only on selected frames to reduce terminal/UI overload.
 *
 * @note Exports JSON at the end of the run for offline analysis.
 */
void VulkanRenderer::mainLoop() {
  int frameCounter = 0;
  const int profileEveryNFrames = 10;
  // Only profile every N frames to avoid terminal spam

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents(); // Handle input + resize events

    bool doProfile = (frameCounter % profileEveryNFrames == 0);
    // Enable profiling only for selected frames

    if (doProfile) {
      ChronoProfiler::ScopedFrame frame;
      PROFILE_SCOPE("drawFrame()");
      drawFrame(); // Render + measure CPU time
    } else {
      drawFrame(); // No profiling this frame
    }

    if (doProfile) {
      profilerUI.update(); // Process profiler data
      profilerUI.render(); // Print profiler UI
    }

    frameCounter++; // Advance frame count
  }

  device.waitIdle(); // Wait for GPU to finish processing all frames
  ChronoProfiler::exportToJSON("profile_output.json");
  // Save profiling data to a JSON file
}

/**
 * @brief Cleans up all Vulkan and GLFW resources before program termination.
 *
 * This function releases the swap chain, destroys the GLFW window, and
 * terminates GLFW properly. It should be called at program shutdown.
 *
 * @see cleanupSwapChain()
 * @see glfwDestroyWindow()
 */
void VulkanRenderer::cleanup() {
  cleanupSwapChain();        // Free swapchain and related resources
  glfwDestroyWindow(window); // Destroy window
  glfwTerminate();           // Deinitialize GLFW
}
