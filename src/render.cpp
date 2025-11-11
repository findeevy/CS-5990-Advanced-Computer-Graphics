/**
 * @file render.cpp
 * @brief VulkanRenderer class implementation for Accelerender.
 *
 * This file contains the implementation of the VulkanRenderer class methods.
 *
 * @authors Finley Deevy, Eric Newton
 * @date 2025-11-10 (Updated)
 */

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#if __has_include(<tiny_obj_loader.h>)
#include <tiny_obj_loader.h>
#else
#include "../external/tinyobjloader/tiny_obj_loader.h"
#endif

#include "render.hpp"

// Method implementations

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
  initWindow();
  initVulkan();
  mainLoop();
  cleanup();
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
  vk::PhysicalDeviceMemoryProperties memProperties =
      physicalGPU.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
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
  vk::PhysicalDeviceProperties physicalDeviceProperties =
      physicalGPU.getProperties();

  vk::SampleCountFlags counts =
      physicalDeviceProperties.limits.framebufferColorSampleCounts &
      physicalDeviceProperties.limits.framebufferDepthSampleCounts;

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

  return vk::SampleCountFlagBits::e1;
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

  if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
    throw std::runtime_error(warn + err);
  }

  std::unordered_map<Vertex, uint32_t> uniqueVertices{};

  for (const auto &shape : shapes) {
    for (const auto &index : shape.mesh.indices) {
      Vertex vertex{};

      vertex.position = {attrib.vertices[3 * index.vertex_index + 0],
                         attrib.vertices[3 * index.vertex_index + 1],
                         attrib.vertices[3 * index.vertex_index + 2]};

      vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                         1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};

      vertex.color = {1.0f, 1.0f, 1.0f};

      if (!uniqueVertices.contains(vertex)) {
        uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
        vertices.push_back(vertex);
      }

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
  vk::Format depthFormat = findDepthFormat();

  createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
              depthFormat, vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eDepthStencilAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage,
              depthImageMemory);

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
  auto formatIt = std::ranges::find_if(candidates, [&](auto const format) {
    vk::FormatProperties props = physicalGPU.getFormatProperties(format);

    return (((tiling == vk::ImageTiling::eLinear) &&
             ((props.linearTilingFeatures & features) == features)) ||
            ((tiling == vk::ImageTiling::eOptimal) &&
             ((props.optimalTilingFeatures & features) == features)));
  });

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
  vk::PhysicalDeviceProperties properties = physicalGPU.getProperties();

  vk::SamplerCreateInfo samplerInfo;
  samplerInfo.magFilter = vk::Filter::eLinear;
  samplerInfo.minFilter = vk::Filter::eLinear;
  samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
  samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
  samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.anisotropyEnable = VK_TRUE;
  samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = vk::CompareOp::eAlways;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = static_cast<float>(mipLevels);
  samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;

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
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = commandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = 1;

  std::unique_ptr<vk::raii::CommandBuffer> commandBuffer =
      std::make_unique<vk::raii::CommandBuffer>(
          std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

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
  commandBuffer.end();

  vk::SubmitInfo submitInfo{};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &*commandBuffer;

  graphicsQueue.submit(submitInfo, nullptr);
  graphicsQueue.waitIdle();
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
  std::ifstream testFile("textures/texture.png");
  if (!testFile.good()) {
    std::cerr << "ERROR: Texture file 'textures/texture.png' not found!"
              << std::endl;
    throw std::runtime_error("Texture file not found!");
  }
  testFile.close();

  int texWidth, texHeight, texChannels;
  stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight,
                              &texChannels, STBI_rgb_alpha);

  if (!pixels) {
    std::cerr << "ERROR: Failed to load texture image: "
              << stbi_failure_reason() << std::endl;
    throw std::runtime_error("Failed to load texture image!");
  }

  mipLevels = static_cast<uint32_t>(
                  std::floor(std::log2(std::max(texWidth, texHeight)))) +
              1;

  vk::DeviceSize imageSize = texWidth * texHeight * 4;

  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});

  createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  void *data = stagingBufferMemory.mapMemory(0, imageSize);
  memcpy(data, pixels, static_cast<size_t>(imageSize));
  stagingBufferMemory.unmapMemory();

  stbi_image_free(pixels);

  createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1,
              vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransferSrc |
                  vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eSampled,
              vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage,
              textureImageMemory);

  transitionImageLayout(textureImage, vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eTransferDstOptimal, mipLevels);

  copyBufferToImage(stagingBuffer, textureImage,
                    static_cast<uint32_t>(texWidth),
                    static_cast<uint32_t>(texHeight));

  generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight,
                  mipLevels);
}

/**
 * @brief Creates color resources for multisampled rendering.
 *
 * This function creates a color image used as a multisampled color
 * attachment. The image is later resolved to the swapchain image for display.
 * This is part of implementing MSAA (Multisample Anti-Aliasing) in Vulkan.
 */
void VulkanRenderer::createColorResources() {
  vk::Format colorFormat = swapChainImageFormat;

  createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
              colorFormat, vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransientAttachment |
                  vk::ImageUsageFlagBits::eColorAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage,
              colorImageMemory);

  colorImageView = vkutils::createImageView(device, colorImage, colorFormat,
                                            vk::ImageAspectFlagBits::eColor, 1);
}

/**
 * @brief Creates a Vulkan image and allocates memory for it.
 *
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param mipLevels Number of mipmap levels.
 * @param numSamples Number of samples per pixel (for MSAA).
 * @param format Image format (e.g., RGBA, depth, etc.).
 * @param tiling Specifies how image data is arranged in memory.
 * @param usage Bitmask specifying intended usage (e.g., sampled, attachment).
 * @param properties Memory properties (device local, host visible, etc.).
 * @param image Reference to store the created Vulkan image handle.
 * @param imageMemory Reference to store the allocated memory handle.
 *
 * This function encapsulates the boilerplate for creating an image in Vulkan:
 *   1. Defines the image parameters.
 *   2. Allocates the proper type of GPU memory.
 *   3. Binds the memory to the image.
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
  imageInfo.imageType = vk::ImageType::e2D;
  imageInfo.format = format;
  imageInfo.extent = vk::Extent3D{width, height, 1};
  imageInfo.mipLevels = mipLevels;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = numSamples;
  imageInfo.tiling = tiling;
  imageInfo.usage = usage;
  imageInfo.sharingMode = vk::SharingMode::eExclusive;

  image = vk::raii::Image(device, imageInfo);

  vk::MemoryRequirements memRequirements = image.getMemoryRequirements();

  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  imageMemory = vk::raii::DeviceMemory(device, allocInfo);
  image.bindMemory(imageMemory, 0);
}

/**
 * @brief Transitions an image between different layouts.
 *
 * @param image The image to transition.
 * @param oldLayout The current layout of the image.
 * @param newLayout The desired new layout.
 * @param mipLevels The number of mipmap levels in the image.
 *
 * Vulkan images must be in specific layouts depending on how they're used
 * (transfer source, shader read, color attachment, etc.). This function
 * records a pipeline barrier to transition the image between these layouts.
 */
void VulkanRenderer::transitionImageLayout(const vk::raii::Image &image,
                                           vk::ImageLayout oldLayout,
                                           vk::ImageLayout newLayout,
                                           uint32_t mipLevels) {
  auto commandBuffer = beginSingleTimeCommands();

  vk::ImageMemoryBarrier barrier{};
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.image = image;
  barrier.subresourceRange = vk::ImageSubresourceRange{
      vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1};

  vk::PipelineStageFlags sourceStage;
  vk::PipelineStageFlags destinationStage;

  if (oldLayout == vk::ImageLayout::eUndefined &&
      newLayout == vk::ImageLayout::eTransferDstOptimal) {
    barrier.srcAccessMask = {};
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
    destinationStage = vk::PipelineStageFlagBits::eTransfer;

  } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
             newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    sourceStage = vk::PipelineStageFlagBits::eTransfer;
    destinationStage = vk::PipelineStageFlagBits::eFragmentShader;

  } else {
    throw std::invalid_argument("Unsupported layout transition!");
  }

  commandBuffer->pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr,
                                 barrier);

  endSingleTimeCommands(*commandBuffer);
}

/**
 * @brief Generates mipmaps for a texture image using linear blitting.
 *
 * @param image The image for which to generate mipmaps.
 * @param imageFormat The format of the image.
 * @param texWidth Texture width in pixels.
 * @param texHeight Texture height in pixels.
 * @param mipLevels Total number of mipmap levels.
 *
 * This function progressively downsamples the texture image from the base
 * level to smaller resolutions, improving visual quality when viewed from a
 * distance. It performs layout transitions, image blits, and synchronization
 * for each mip level.
 */
void VulkanRenderer::generateMipmaps(vk::raii::Image &image,
                                     vk::Format imageFormat, int32_t texWidth,
                                     int32_t texHeight, uint32_t mipLevels) {
  vk::FormatProperties formatProperties =
      physicalGPU.getFormatProperties(imageFormat);

  if (!(formatProperties.optimalTilingFeatures &
        vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
    throw std::runtime_error(
        "Texture image format does not support linear blitting!");
  }

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

  for (uint32_t i = 1; i < mipLevels; i++) {
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

    commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eTransfer, {}, {},
                                   {}, barrier);

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

    commandBuffer->blitImage(*image, vk::ImageLayout::eTransferSrcOptimal,
                             *image, vk::ImageLayout::eTransferDstOptimal,
                             {blit}, vk::Filter::eLinear);

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

  barrier.subresourceRange.baseMipLevel = mipLevels - 1;
  barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
  barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
  barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

  commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eFragmentShader, {},
                                 {}, {}, barrier);

  endSingleTimeCommands(*commandBuffer);
}

/**
 * @brief Copies data from a Vulkan buffer to an image.
 *
 * This function records and submits a command buffer that copies pixel data
 * from a given Vulkan buffer (usually staging buffer) into a GPU image
 * (usually a texture or framebuffer attachment). The image is expected to be
 * in the 'vk::ImageLayout::eTransferDstOptimal' layout.
 *
 * @param[in] buffer The source buffer containing image data.
 * @param[in,out] image The destination Vulkan image to receive data.
 * @param[in] width The width of the image in pixels.
 * @param[in] height The height of the image in pixels.
 *
 * @note The function assumes that @c beginSingleTimeCommands() and
 *       @c endSingleTimeCommands() properly handle command buffer
 *       allocation, submission, and cleanup.
 * @warning The image layout must be transitioned to
 *          @c vk::ImageLayout::eTransferDstOptimal before calling this.
 */
void VulkanRenderer::copyBufferToImage(const vk::raii::Buffer &buffer,
                                       vk::raii::Image &image, uint32_t width,
                                       uint32_t height) {
  std::unique_ptr<vk::raii::CommandBuffer> commandBuffer =
      beginSingleTimeCommands();

  vk::BufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource =
      vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
  region.imageOffset = vk::Offset3D{0, 0, 0};
  region.imageExtent = vk::Extent3D{width, height, 1};

  commandBuffer->copyBufferToImage(
      buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});

  endSingleTimeCommands(*commandBuffer);
}

/**
 * @brief Creates a Vulkan descriptor pool.
 *
 * The descriptor pool manages allocations of descriptor sets, which are
 * used to bind resources like uniform buffers and textures to shader stages.
 *
 * @details This implementation supports two descriptor types:
 * - Uniform Buffers (for per-frame transformation data)
 * - Combined Image Samplers (for texture bindings)
 *
 * @note The number of sets is limited by @c MAX_FRAMES_IN_FLIGHT.
 * @see createDescriptorSets()
 */
void VulkanRenderer::createDescriptorPool() {
  std::array<vk::DescriptorPoolSize, 2> poolSizes = {};
  poolSizes[0] = vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                                        MAX_FRAMES_IN_FLIGHT);
  poolSizes[1] = vk::DescriptorPoolSize(
      vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT);

  vk::DescriptorPoolCreateInfo poolInfo;
  poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();

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
  std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               *descriptorSetLayout);

  vk::DescriptorSetAllocateInfo allocInfo;
  allocInfo.descriptorPool = *descriptorPool;
  allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
  allocInfo.pSetLayouts = layouts.data();

  descriptorSets = device.allocateDescriptorSets(allocInfo);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorBufferInfo bufferInfo;
    bufferInfo.buffer = *uniformBuffers[i];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    vk::WriteDescriptorSet descriptorWrite;
    descriptorWrite.dstSet = *descriptorSets[i];
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vk::DescriptorImageInfo imageInfo;
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = *textureImageView;
    imageInfo.sampler = *textureSampler;

    vk::WriteDescriptorSet samplerWrite;
    samplerWrite.dstSet = *descriptorSets[i];
    samplerWrite.dstBinding = 1;
    samplerWrite.dstArrayElement = 0;
    samplerWrite.descriptorCount = 1;
    samplerWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    samplerWrite.pImageInfo = &imageInfo;

    std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {descriptorWrite,
                                                              samplerWrite};
    device.updateDescriptorSets(descriptorWrites, {});
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
  static auto startTime = std::chrono::high_resolution_clock::now();

  auto currentTime = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float>(currentTime - startTime).count();

  UniformBufferObject ubo{};
  ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                          glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.view =
      glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                  glm::vec3(0.0f, 0.0f, 1.0f));
  ubo.proj = glm::perspective(glm::radians(45.0f),
                              static_cast<float>(swapChainExtent.width) /
                                  static_cast<float>(swapChainExtent.height),
                              0.1f, 10.0f);
  ubo.proj[1][1] *= -1;

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
  uniformBuffers.clear();
  uniformBuffersMemory.clear();
  uniformBuffersMapped.clear();

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    vk::raii::Buffer buffer({});
    vk::raii::DeviceMemory bufferMem({});

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 buffer, bufferMem);

    uniformBuffers.emplace_back(std::move(buffer));
    uniformBuffersMemory.emplace_back(std::move(bufferMem));

    uniformBuffersMapped.emplace_back(
        uniformBuffersMemory[i].mapMemory(0, bufferSize));
  }
}

/**
 * @brief Creates a Vulkan descriptor set layout for uniform buffers and
 * texture samplers.
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
  std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {};

  bindings[0] =
      vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                     vk::ShaderStageFlagBits::eVertex, nullptr);

  bindings[1] = vk::DescriptorSetLayoutBinding(
      1, vk::DescriptorType::eCombinedImageSampler, 1,
      vk::ShaderStageFlagBits::eFragment, nullptr);

  vk::DescriptorSetLayoutCreateInfo layoutInfo;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

/**
 * @brief Copies data from one Vulkan buffer to another using a command
 * buffer.
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
  vk::CommandBufferAllocateInfo allocInfo{};
  allocInfo.commandPool = *commandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = 1;

  auto commandBuffers = device.allocateCommandBuffers(allocInfo);
  vk::CommandBuffer commandBuffer = *commandBuffers[0];

  vk::CommandBufferBeginInfo beginInfo{};
  beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  commandBuffer.begin(beginInfo);

  vk::BufferCopy copyRegion{};
  copyRegion.size = size;
  commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

  commandBuffer.end();

  vk::SubmitInfo submitInfo{};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  graphicsQueue.submit(submitInfo, nullptr);
  graphicsQueue.waitIdle();
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
 * This function encapsulates buffer creation and memory allocation.
 * It uses 'findMemoryType()' to locate compatible memory based on the
 * buffer's requirements.
 *
 * @see findMemoryType()
 * @see copyBuffer()
 */
void VulkanRenderer::createBuffer(vk::DeviceSize size,
                                  vk::BufferUsageFlags usage,
                                  vk::MemoryPropertyFlags properties,
                                  vk::raii::Buffer &buffer,
                                  vk::raii::DeviceMemory &bufferMemory) {
  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = vk::SharingMode::eExclusive;

  buffer = vk::raii::Buffer(device, bufferInfo);

  vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

  vk::MemoryAllocateInfo allocInfo{};
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
  buffer.bindMemory(*bufferMemory, 0);
}

/**
 * @brief Creates the index buffer for drawing geometry.
 *
 * @details
 * The function first creates a staging buffer in host-visible memory,
 * copies the index data into it, then creates a device-local buffer
 * and transfers the data using 'copyBuffer()'.
 *
 * @note Index buffer improves efficiency by reusing vertex data.
 * @see copyBuffer()
 * @see createBuffer()
 */
void VulkanRenderer::createIndexBuffer() {
  vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});
  createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  void *data = stagingBufferMemory.mapMemory(0, bufferSize);
  memcpy(data, indices.data(), (size_t)bufferSize);
  stagingBufferMemory.unmapMemory();

  createBuffer(bufferSize,
               vk::BufferUsageFlagBits::eTransferDst |
                   vk::BufferUsageFlagBits::eIndexBuffer,
               vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer,
               indexBufferMemory);

  copyBuffer(stagingBuffer, indexBuffer, bufferSize);
}

/**
 * @brief Creates the vertex buffer for rendering geometry.
 *
 * @details
 * Similar to 'createIndexBuffer()', this function uses a staging buffer
 * to copy vertex data to device-local memory for optimal GPU access.
 *
 * @note Device-local memory is faster for the GPU but cannot be mapped
 * directly.
 * @warning Ensure vertex layout matches shader input structure.
 */
void VulkanRenderer::createVertexBuffer() {
  vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

  vk::raii::Buffer stagingBuffer = nullptr;
  vk::raii::DeviceMemory stagingBufferMemory = nullptr;
  createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer, stagingBufferMemory);

  void *data = stagingBufferMemory.mapMemory(0, bufferSize);
  memcpy(data, vertices.data(), (size_t)bufferSize);
  stagingBufferMemory.unmapMemory();

  createBuffer(bufferSize,
               vk::BufferUsageFlagBits::eVertexBuffer |
                   vk::BufferUsageFlagBits::eTransferDst,
               vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer,
               vertexBufferMemory);

  copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
}

/**
 * @brief GLFW callback to mark when the framebuffer is resized.
 *
 * @param[in] window Pointer to the GLFW window being resized.
 * @param[in] width Unused parameter (required by GLFW signature).
 * @param[in] height Unused parameter (required by GLFW signature).
 *
 * @details
 * This callback sets a flag inside the VulkanRenderer object indicating
 * that the swapchain must be recreated due to window resize.
 *
 * @note The flag 'framebufferResized' is later checked in the render loop.
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
 * These command buffers are used to record rendering commands that are
 * submitted to the GPU.
 *
 * @note Uses @c MAX_FRAMES_IN_FLIGHT to ensure each frame has its own buffer.
 */
void VulkanRenderer::createCommandBuffers() {
  commandBuffers.clear();

  vk::CommandBufferAllocateInfo allocInfo;
  allocInfo.commandPool = *commandPool;
  allocInfo.level = vk::CommandBufferLevel::ePrimary;
  allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

  commandBuffers = device.allocateCommandBuffers(allocInfo);
}

/**
 * @brief Creates the Vulkan command pool.
 *
 * @details
 * Command pools manage the memory used by command buffers.
 * This pool is tied to the graphics queue family and allows resetting
 * individual buffers.
 *
 * @note The flag 'eResetCommandBuffer' allows command buffers to be
 * rerecorded.
 */
void VulkanRenderer::createCommandPool() {
  vk::CommandPoolCreateInfo poolInfo;
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
  commandPool = vk::raii::CommandPool(device, poolInfo);
}

/**
 * @brief Creates synchronization primitives for frame rendering.
 *
 * @details
 * Each frame requires:
 * - A semaphore for presentation completion
 * - A semaphore for rendering completion
 * - A fence to synchronize CPU-GPU work
 *
 * @note The number of synchronization objects matches @c
 * MAX_FRAMES_IN_FLIGHT.
 * @warning Fences are initialized as signaled to avoid deadlock on first use.
 */
void VulkanRenderer::createSyncObjects() {
  presentCompleteSemaphores.clear();
  renderFinishedSemaphores.clear();
  inFlightFences.clear();

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    inFlightFences.emplace_back(
        device, vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
  }
}

/**
 * @brief Draws a single frame in the Vulkan rendering loop.
 *
 * This function handles the synchronization of GPU and CPU operations per
 * frame, acquires the next available swapchain image, submits rendering
 * commands to the graphics queue, and presents the rendered image to the
 * presentation queue.
 *
 * @details
 * Steps include:
 *  - Waiting for fences to ensure GPU has finished using current frame's
 * resources.
 *  - Acquiring a new swapchain image.
 *  - Recording command buffers for the frame.
 *  - Submitting draw commands and signaling semaphores for synchronization.
 *  - Presenting the final image to the swapchain.
 *
 * @note This function handles swapchain recreation when the window is resized
 * or swapchain is out of date.
 *
 * @pre Vulkan device, queues, and synchronization objects must be
 * initialized.
 * @post Submits rendering commands to the graphics queue.
 *
 * @throws std::runtime_error if swapchain image acquisition or presentation
 * fails.
 *
 * @see recreateSwapChain()
 * @see updateUniformBuffer()
 * @see recordCommandBuffer()
 */
void VulkanRenderer::drawFrame() {
  while (
      vk::Result::eTimeout ==
      device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
    ;

  auto [result, imageIndex] = swapChain.acquireNextImage(
      UINT64_MAX, *presentCompleteSemaphores[currentFrame], nullptr);

  if (result == vk::Result::eErrorOutOfDateKHR) {
    recreateSwapChain();
    return;
  }

  if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
    throw std::runtime_error("failed to acquire swap chain image!");
  }

  updateUniformBuffer(currentFrame);

  device.resetFences(*inFlightFences[currentFrame]);
  commandBuffers[currentFrame].reset();

  recordCommandBuffer(imageIndex);

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

  graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

  vk::PresentInfoKHR presentInfoKHR;
  presentInfoKHR.waitSemaphoreCount = 1;
  presentInfoKHR.pWaitSemaphores = &*renderFinishedSemaphores[currentFrame];
  presentInfoKHR.swapchainCount = 1;
  presentInfoKHR.pSwapchains = &*swapChain;
  presentInfoKHR.pImageIndices = &imageIndex;

  result = presentQueue.presentKHR(presentInfoKHR);

  if (result == vk::Result::eErrorOutOfDateKHR ||
      result == vk::Result::eSuboptimalKHR || framebufferResized) {
    framebufferResized = false;
    recreateSwapChain();
  } else if (result != vk::Result::eSuccess) {
    throw std::runtime_error("failed to present swap chain image!");
  }

  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

/**
 * @brief Transitions the layout of a swapchain image.
 *
 * @param[in] imageIndex Index of the swapchain image to transition.
 * @param[in] oldLayout The current image layout.
 * @param[in] newLayout The desired new image layout.
 * @param[in] srcAccessMask Specifies memory access before the barrier.
 * @param[in] dstAccessMask Specifies memory access after the barrier.
 * @param[in] srcStageMask Pipeline stage before the barrier.
 * @param[in] dstStageMask Pipeline stage after the barrier.
 *
 * @details
 * This function inserts a pipeline barrier to synchronize memory access
 * and change the layout of the specified swapchain image.
 *
 * @note Vulkan requires explicit layout transitions for proper rendering.
 * @warning Incorrect barrier configuration can cause undefined behavior.
 *
 * @see vk::ImageMemoryBarrier2
 * @see vk::DependencyInfo
 */
void VulkanRenderer::transition_image_layout(
    uint32_t imageIndex, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
    vk::AccessFlags2 srcAccessMask, vk::AccessFlags2 dstAccessMask,
    vk::PipelineStageFlags2 srcStageMask,
    vk::PipelineStageFlags2 dstStageMask) {
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
  commandBuffers[currentFrame].begin({});

  // --- COLOR IMAGE BARRIER ---
  // Prepare the multisampled color image for color attachment output.
  vk::ImageMemoryBarrier2 colorBarrier;
  colorBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
  colorBarrier.srcAccessMask = {};
  colorBarrier.dstStageMask =
      vk::PipelineStageFlagBits2::eColorAttachmentOutput;
  colorBarrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
  colorBarrier.oldLayout = vk::ImageLayout::eUndefined;
  colorBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
  colorBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  colorBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  colorBarrier.image = *colorImage;
  colorBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  colorBarrier.subresourceRange.baseMipLevel = 0;
  colorBarrier.subresourceRange.levelCount = 1;
  colorBarrier.subresourceRange.baseArrayLayer = 0;
  colorBarrier.subresourceRange.layerCount = 1;

  vk::DependencyInfo colorDependencyInfo;
  colorDependencyInfo.imageMemoryBarrierCount = 1;
  colorDependencyInfo.pImageMemoryBarriers = &colorBarrier;
  commandBuffers[currentFrame].pipelineBarrier2(colorDependencyInfo);

  // --- SWAPCHAIN IMAGE BARRIER ---
  // Transition the swapchain image for rendering output.
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
  // Transition the depth buffer for use in fragment depth testing.
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
  // Define clear values for color and depth buffers.
  vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  vk::ClearValue depthClearValue = vk::ClearDepthStencilValue(1.0f, 0);

  // Configure multisampled color attachment.
  vk::RenderingAttachmentInfo colorAttachmentInfo;
  colorAttachmentInfo.imageView = *colorImageView;
  colorAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
  colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
  colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
  colorAttachmentInfo.clearValue = clearColor;

  // Define resolve attachment for MSAA resolve into swapchain image.
  vk::RenderingAttachmentInfo resolveAttachmentInfo;
  resolveAttachmentInfo.imageView = *swapChainImageViews[imageIndex];
  resolveAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
  resolveAttachmentInfo.loadOp = vk::AttachmentLoadOp::eDontCare;
  resolveAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
  resolveAttachmentInfo.clearValue = clearColor;

  // Link color and resolve attachments together.
  colorAttachmentInfo.resolveMode = vk::ResolveModeFlagBits::eAverage;
  colorAttachmentInfo.resolveImageView = resolveAttachmentInfo.imageView;
  colorAttachmentInfo.resolveImageLayout = resolveAttachmentInfo.imageLayout;

  // Configure depth attachment.
  vk::RenderingAttachmentInfo depthAttachmentInfo;
  depthAttachmentInfo.imageView = *depthImageView;
  depthAttachmentInfo.imageLayout =
      vk::ImageLayout::eDepthStencilAttachmentOptimal;
  depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
  depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
  depthAttachmentInfo.clearValue = depthClearValue;

  // Define rendering region and begin dynamic rendering.
  vk::RenderingInfo renderingInfo;
  renderingInfo.renderArea = vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent);
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments = &colorAttachmentInfo;
  renderingInfo.pDepthAttachment = &depthAttachmentInfo;
  renderingInfo.pStencilAttachment = nullptr;

  // Start rendering process.
  commandBuffers[currentFrame].beginRendering(renderingInfo);

  commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics,
                                            *graphicsPipeline);

  vk::DeviceSize offsets[] = {0};
  commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, offsets);
  commandBuffers[currentFrame].bindIndexBuffer(*indexBuffer, 0,
                                               vk::IndexType::eUint32);

  commandBuffers[currentFrame].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
      *descriptorSets[currentFrame], nullptr);

  commandBuffers[currentFrame].setViewport(
      0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                      static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
  commandBuffers[currentFrame].setScissor(
      0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

  commandBuffers[currentFrame].drawIndexed(
      static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

  commandBuffers[currentFrame].endRendering();

  // --- TRANSITION TO PRESENT ---
  // Transition image layout from color attachment to presentable format.
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
  vertexInputInfo = vk::PipelineVertexInputStateCreateInfo{};

  std::vector<char> vertShaderCode = vkutils::readFile("shaders/vert.spv");
  std::vector<char> fragShaderCode = vkutils::readFile("shaders/frag.spv");

  vk::raii::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
  vk::raii::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

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

  vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
  inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();

  vertexInputInfo = vk::PipelineVertexInputStateCreateInfo(
      vk::PipelineVertexInputStateCreateFlags(), 1, &bindingDescription,
      static_cast<uint32_t>(attributeDescriptions.size()),
      attributeDescriptions.data());

  vk::PipelineViewportStateCreateInfo viewportState;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  vk::PipelineRasterizationStateCreateInfo rasterizer;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = vk::PolygonMode::eFill;
  rasterizer.cullMode = vk::CullModeFlagBits::eBack;
  rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.lineWidth = 1.0f;

  vk::PhysicalDeviceFeatures supportedFeatures = physicalGPU.getFeatures();
  bool sampleRateShadingSupported = supportedFeatures.sampleRateShading;

  vk::PipelineMultisampleStateCreateInfo multisampling;
  multisampling.rasterizationSamples = msaaSamples;
  multisampling.sampleShadingEnable =
      sampleRateShadingSupported ? VK_TRUE : VK_FALSE;
  multisampling.minSampleShading = sampleRateShadingSupported ? 0.2f : 1.0f;

  vk::PipelineDepthStencilStateCreateInfo depthStencil;
  depthStencil.depthTestEnable = vk::True;
  depthStencil.depthWriteEnable = vk::True;
  depthStencil.depthCompareOp = vk::CompareOp::eLess;
  depthStencil.depthBoundsTestEnable = vk::False;
  depthStencil.stencilTestEnable = vk::False;

  vk::PipelineColorBlendAttachmentState colorBlendAttachment;
  colorBlendAttachment.blendEnable = VK_FALSE;
  colorBlendAttachment.colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

  vk::PipelineColorBlendStateCreateInfo colorBlending;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;

  std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                 vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamicState;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

  vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo;
  pipelineRenderingCreateInfo.colorAttachmentCount = 1;
  pipelineRenderingCreateInfo.pColorAttachmentFormats =
      &swapChainSurfaceFormat.format;
  pipelineRenderingCreateInfo.depthAttachmentFormat = findDepthFormat();

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
  pipelineInfo.renderPass = nullptr;

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
  if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
    throw std::runtime_error("Failed to create window surface!");
  }
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
    return;
  }

  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

  vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT;
  debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags;
  debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags;
  debugUtilsMessengerCreateInfoEXT.pfnUserCallback = debugCallback;

  debugMessenger =
      instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
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
  }
  return vk::False;
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

  std::vector<const char *> extensions(glfwExtensions,
                                       glfwExtensions + glfwExtensionCount);

  if (enableValidationLayers) {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
  }

  extensions.push_back(vk::KHRGetPhysicalDeviceProperties2ExtensionName);
  return extensions;
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

  std::vector<const char *> requiredLayers;
  if (enableValidationLayers) {
    requiredLayers.assign(validationLayers.begin(), validationLayers.end());
  }

  auto layerProperties = context.enumerateInstanceLayerProperties();
  for (auto const &requiredLayer : requiredLayers) {
    if (std::none_of(layerProperties.begin(), layerProperties.end(),
                     [requiredLayer](auto const &layerProperty) {
                       return strcmp(layerProperty.layerName, requiredLayer) ==
                              0;
                     })) {
      throw std::runtime_error("Required layer not supported: " +
                               std::string(requiredLayer));
    }
  }

  auto requiredExtensions = getRequiredExtensions();

#if defined(__APPLE__)
  requiredExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

  auto extensionProperties = context.enumerateInstanceExtensionProperties();
  for (auto const &requiredExtension : requiredExtensions) {
    if (std::none_of(extensionProperties.begin(), extensionProperties.end(),
                     [requiredExtension](auto const &extensionProperty) {
                       return strcmp(extensionProperty.extensionName,
                                     requiredExtension) == 0;
                     })) {
      throw std::runtime_error("Required extension not supported: " +
                               std::string(requiredExtension));
    }
  }

  vk::InstanceCreateInfo createInfo;
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
  createInfo.ppEnabledLayerNames = requiredLayers.data();
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(requiredExtensions.size());
  createInfo.ppEnabledExtensionNames = requiredExtensions.data();

#if defined(__APPLE__)
  createInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

  instance = vk::raii::Instance(context, createInfo);
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
  swapChainImageViews.clear();
  swapChainImageViews.reserve(swapChainImages.size());

  for (auto &image : swapChainImages) {
    swapChainImageViews.emplace_back(
        vkutils::createImageView(device, image, swapChainImageFormat,
                                 vk::ImageAspectFlagBits::eColor, 1));
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

  auto chosenSurfaceFormat =
      chooseSwapSurfaceFormat(physicalGPU.getSurfaceFormatsKHR(*surface));
  swapChainImageFormat = chosenSurfaceFormat.format;
  swapChainSurfaceFormat = chosenSurfaceFormat;
  auto swapChainColorSpace = chosenSurfaceFormat.colorSpace;

  swapChainExtent = chooseSwapExtent(surfaceCapabilities);

  auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
  minImageCount = (surfaceCapabilities.maxImageCount > 0 &&
                   minImageCount > surfaceCapabilities.maxImageCount)
                      ? surfaceCapabilities.maxImageCount
                      : minImageCount;

  auto presentMode =
      chooseSwapPresentMode(physicalGPU.getSurfacePresentModesKHR(*surface));

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

  swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);

  swapChainImages.clear();
  auto images = swapChain.getImages();
  swapChainImages.reserve(images.size());

  for (auto &image : images) {
    swapChainImages.emplace_back(device, image);
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
    return capabilities.currentExtent;
  }

  int width, height;
  glfwGetFramebufferSize(window, &width, &height);

  return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width),
          std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height)};
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
      return availablePresentMode;
    }
  }
  return vk::PresentModeKHR::eFifo;
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
      return availableFormat;
    }
  }
  return availableFormats[0];
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

  uint32_t graphicsIndex = static_cast<uint32_t>(queueFamilyProperties.size());
  uint32_t presentIndex = static_cast<uint32_t>(queueFamilyProperties.size());

  for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
    if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
      if (graphicsIndex == queueFamilyProperties.size()) {
        graphicsIndex = i;
      }
      if (physicalGPU.getSurfaceSupportKHR(i, *surface)) {
        graphicsIndex = i;
        presentIndex = i;
        break;
      }
    }
  }

  if (presentIndex == queueFamilyProperties.size()) {
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
      if (physicalGPU.getSurfaceSupportKHR(i, *surface)) {
        presentIndex = i;
        break;
      }
    }
  }

  if (graphicsIndex == queueFamilyProperties.size()) {
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
      if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        graphicsIndex = i;
        break;
      }
    }
  }

  if ((graphicsIndex == queueFamilyProperties.size()) ||
      (presentIndex == queueFamilyProperties.size())) {
    throw std::runtime_error("No graphics or present queue family found!");
  }

  graphicsQueueFamilyIndex = graphicsIndex;
  std::set<uint32_t> uniqueQueueFamilies = {graphicsIndex, presentIndex};
  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
  float queuePriority = 1.0f;

  for (uint32_t queueFamily : uniqueQueueFamilies) {
    vk::DeviceQueueCreateInfo queueCreateInfo;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  vk::PhysicalDeviceFeatures supportedFeatures = physicalGPU.getFeatures();
  bool sampleRateShadingSupported = supportedFeatures.sampleRateShading;

  vk::StructureChain<vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan13Features,
                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
      featureChain;

  if (sampleRateShadingSupported) {
    featureChain.get<vk::PhysicalDeviceFeatures2>().features.sampleRateShading =
        VK_TRUE;
  }

  featureChain.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering =
      true;
  featureChain.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 =
      true;
  featureChain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
      .extendedDynamicState = true;

  vk::DeviceCreateInfo deviceCreateInfo;
  deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
  deviceCreateInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
  deviceCreateInfo.enabledExtensionCount =
      static_cast<uint32_t>(gpuExtensions.size());
  deviceCreateInfo.ppEnabledExtensionNames = gpuExtensions.data();

  device = vk::raii::Device(physicalGPU, deviceCreateInfo);

  graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
  presentQueue = vk::raii::Queue(device, presentIndex, 0);
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

  for (auto const &gpu : gpus) {
    auto queueFamilies = gpu.getQueueFamilyProperties();

    bool isSuitable = gpu.getProperties().apiVersion >= VK_API_VERSION_1_3;

    const auto qfpIter =
        std::find_if(queueFamilies.begin(), queueFamilies.end(),
                     [](vk::QueueFamilyProperties const &qfp) {
                       return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) !=
                              static_cast<vk::QueueFlags>(0);
                     });
    isSuitable = isSuitable && (qfpIter != queueFamilies.end());

    auto extensions = gpu.enumerateDeviceExtensionProperties();
    bool found = true;
    for (auto const &extension : gpuExtensions) {
      auto extensionIter = std::find_if(
          extensions.begin(), extensions.end(), [extension](auto const &ext) {
            return strcmp(ext.extensionName, extension) == 0;
          });
      found = found && extensionIter != extensions.end();
    }
    isSuitable = isSuitable && found;

    if (isSuitable) {
      physicalGPU = gpu;
      msaaSamples = getMaxUsableSampleCount();
      return;
    }
  }

  throw std::runtime_error("Failed to find a GPU that supports Vulkan 1.3!");
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

  while (width < 1 || height < 1) {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

  device.waitIdle();
  cleanupSwapChain();

  createSwapChain();
  createImageViews();
  createColorResources();
  createDepthResources();
  createCommandBuffers();
  createSyncObjects();
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
  colorImageView = nullptr;
  colorImage = nullptr;
  colorImageMemory = nullptr;

  swapChainImageViews.clear();
  swapChain = nullptr;
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
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  window = glfwCreateWindow(WIDTH, HEIGHT, "Accelerender", nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
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
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalGPU();
  pickLogicalGPU();
  createSwapChain();
  createImageViews();
  createColorResources();
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createCommandPool();
  createDepthResources();
  createTextureImage();
  createTextureImageView();
  createTextureSampler();
  loadModel();
  createVertexBuffer();
  createIndexBuffer();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
  createSyncObjects();
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

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    bool doProfile = (frameCounter % profileEveryNFrames == 0);

    if (doProfile) {
      ChronoProfiler::ScopedFrame frame;
      PROFILE_SCOPE("drawFrame()");
      drawFrame();
    } else {
      drawFrame();
    }

    if (doProfile) {
      profilerUI.update();
      profilerUI.render();
    }

    frameCounter++;
  }

  device.waitIdle();
  ChronoProfiler::exportToJSON("profile_output.json");
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
  cleanupSwapChain();
  glfwDestroyWindow(window);
  glfwTerminate();
}
