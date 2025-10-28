#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vulkan/vulkan_raii.hpp>

namespace vkutils {

    /**
     * @brief Reads a binary file into a std::vector<char>.
     * @param filename Path to the file.
     * @return std::vector<char> containing the file contents.
     * @throws std::runtime_error if the file cannot be opened.
     */
    std::vector<char> readFile(const std::string &filename);

    /**
     *
     * @param device
     * @param image
     * @param format
     * @param aspectFlags
     * @return
     */
    vk::raii::ImageView createImageView(const vk::raii::Device &device,
                                        const vk::raii::Image &image,
                                        vk::Format format,
                                        vk::ImageAspectFlags aspectFlags,
                                        uint32_t mipLevels = 1);

} // namespace vkutils
