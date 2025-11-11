#pragma once
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

/**
 * @file VulkanUtils.hpp
 * @brief Provides utility functions for Vulkan operations (file I/O and image
 * view creation).
 *
 * The **vkutils** namespace contains helper functions used across the rendering
 * pipeline for tasks such as loading shader binaries and creating image views.
 * These abstractions simplify repetitive Vulkan boilerplate, keeping core
 * rendering classes like `SwapChain` and `Vulkan` cleaner.
 *
 * @ingroup Rendering
 *
 * @see vk::raii::Image
 * @see vk::raii::ImageView
 * @see vk::raii::Device
 *
 */
namespace vkutils {

/**
 * @brief Reads a binary file (e.g., SPIR-V shader) into a vector of bytes.
 *
 * Opens a file in binary mode, seeks to the end to determine size, and
 * reads the contents into a `std::vector<char>`.
 *
 * @param filename Path to the file to read.
 * @return std::vector<char> containing the complete binary contents.
 *
 * @throws std::runtime_error If the file cannot be opened or read.
 *
 * @note Commonly used to load SPIR-V shader bytecode for Vulkan pipeline
 * creation.
 *
 * @code
 * auto vertShaderCode = vkutils::readFile("shaders/vert.spv");
 * auto fragShaderCode = vkutils::readFile("shaders/frag.spv");
 * @endcode
 */
std::vector<char> readFile(const std::string &filename);

/**
 * @brief Creates a Vulkan image view for a given image resource.
 *
 * An image view describes how Vulkan should access an image, specifying the
 * format, subresource range, and mip levels. This function encapsulates the
 * Vulkan setup boilerplate into a single RAII-based helper.
 *
 * @param device Logical Vulkan device used to create the view.
 * @param image Target Vulkan image.
 * @param format Image format (e.g., `vk::Format::eR8G8B8A8Srgb`).
 * @param aspectFlags Specifies which aspects (color, depth, stencil) are
 * included.
 * @param mipLevels Number of mipmap levels in the image. Defaults to 1.
 *
 * @return A `vk::raii::ImageView` wrapping the created image view.
 *
 * @throws vk::SystemError If Vulkan fails to create the image view.
 *
 * @note Typically used for texture or swap chain images before rendering.
 *
 * @see vk::ImageViewCreateInfo
 * @see vk::ImageAspectFlags
 *
 * @code
 * vk::raii::ImageView imageView = vkutils::createImageView(device, image,
 * format, vk::ImageAspectFlagBits::eColor);
 * @endcode
 */
vk::raii::ImageView createImageView(const vk::raii::Device &device,
                                    const vk::raii::Image &image,
                                    vk::Format format,
                                    vk::ImageAspectFlags aspectFlags,
                                    uint32_t mipLevels = 1);

} // namespace vkutils
