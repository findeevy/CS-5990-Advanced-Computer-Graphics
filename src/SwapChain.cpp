#include "SwapChain.hpp"
#include "VulkanUtils.hpp"
#include <algorithm>  // for std::max

/**
 * @brief Creates the Vulkan swap chain and its associated image views.
 */
void SwapChain::create() {
    // Query surface capabilities
    auto surfaceCapabilities = physicalGPU.getSurfaceCapabilitiesKHR(*surface);

    // Pick surface format
    auto chosenSurfaceFormat =
            chooseSwapSurfaceFormat(physicalGPU.getSurfaceFormatsKHR(*surface));
    swapChainImageFormat = chosenSurfaceFormat.format;
    swapChainSurfaceFormat = chosenSurfaceFormat;
    auto swapChainColorSpace = chosenSurfaceFormat.colorSpace;

    // Choose extent
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);

    // Determine number of images
    uint32_t minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) {
        minImageCount = surfaceCapabilities.maxImageCount;
    }

    // Choose present mode
    auto presentMode = chooseSwapPresentMode(physicalGPU.getSurfacePresentModesKHR(*surface));

    // Create swap chain
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

    // Get swap chain images (raw vk::Image handles)
    swapChainImages = swapChain.getImages();

    // Wrap images in RAII
    std::vector<vk::raii::Image> raiiSwapChainImages;
    raiiSwapChainImages.reserve(swapChainImages.size());
    for (auto &img : swapChainImages) {
        raiiSwapChainImages.emplace_back(device, img);
    }

    // Create image views
    swapChainImageViews.clear();
    swapChainImageViews.reserve(raiiSwapChainImages.size());
    for (auto &image : raiiSwapChainImages) {
        swapChainImageViews.emplace_back(
                vkutils::createImageView(device, image, swapChainImageFormat, vk::ImageAspectFlagBits::eColor)
        );
    }
}

/**
 * @brief Cleans up swap chain resources.
 */
void SwapChain::cleanup() {
    swapChainImageViews.clear();
    swapChain = nullptr;
}

/**
 * @brief Recreates the swap chain (for example, after a window resize).
 */
void SwapChain::recreate() {
    cleanup();
    create();
}

/**
 * @brief Chooses the swap chain image extent based on surface capabilities.
 * @param capabilities The surface capabilities.
 * @return The chosen swap chain extent.
 */
vk::Extent2D SwapChain::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        vk::Extent2D actualExtent = {800, 600};  // Default fallback
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actualExtent;
    }
}

/**
 * @brief Chooses the swap chain present mode.
 * @param availablePresentModes A list of available present modes.
 * @return The chosen present mode.
 */
vk::PresentModeKHR SwapChain::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes) {
    for (const auto &mode : availablePresentModes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
            return mode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

/**
 * @brief Chooses the swap chain surface format.
 * @param availableFormats A list of available surface formats.
 * @return The chosen surface format.
 */
vk::SurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    for (const auto &format : availableFormats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }
    return availableFormats[0];
}
