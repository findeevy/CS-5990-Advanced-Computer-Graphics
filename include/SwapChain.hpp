#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vector>

class SwapChain {
public:
    // Constructor: requires Device, PhysicalDevice, and Surface
    // Previously Vulkan::device, Vulkan::physicalGPU, Vulkan::surface
    SwapChain(vk::raii::Device &device,
              vk::raii::PhysicalDevice &gpu,
              vk::raii::SurfaceKHR &surface)
            : device(device), physicalGPU(gpu), surface(surface) {}

    // -------------------
    // Functions migrated from Vulkan
    // -------------------
    void create();   // Originally Vulkan::createSwapChain()
    void recreate(); // Originally Vulkan::recreateSwapChain()
    void cleanup();  // Originally Vulkan::cleanupSwapChain()

    // Accessors (new)
    const std::vector<vk::raii::ImageView> &getImageViews() const { return swapChainImageViews; }   // Vulkan::swapChainImageViews
    vk::Format getImageFormat() const { return swapChainImageFormat; }                              // Vulkan::swapChainImageFormat
    vk::Extent2D getExtent() const { return swapChainExtent; }                                      // Vulkan::swapChainExtent

private:
    // RAII-wrapped images to ensure lifetime
    std::vector<vk::raii::Image> raiiSwapChainImages; // Wraps swapChainImages

    // -------------------
    // References to Vulkan core objects (from Vulkan class)
    // -------------------
    vk::raii::Device &device;           // Vulkan::device
    vk::raii::PhysicalDevice &physicalGPU; // Vulkan::physicalGPU
    vk::raii::SurfaceKHR &surface;      // Vulkan::surface

    // -------------------
    // SwapChain data
    // -------------------
    vk::raii::SwapchainKHR swapChain = nullptr; // Vulkan::swapChain
    std::vector<vk::Image> swapChainImages;     // Vulkan::swapChainImages
    std::vector<vk::raii::ImageView> swapChainImageViews; // Vulkan::swapChainImageViews
    vk::Format swapChainImageFormat = vk::Format::eUndefined; // Vulkan::swapChainImageFormat
    vk::Extent2D swapChainExtent;              // Vulkan::swapChainExtent
    vk::SurfaceFormatKHR swapChainSurfaceFormat; // Vulkan::swapChainSurfaceFormat

    // -------------------
    // Internal helpers (migrated from Vulkan class)
    // -------------------
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities);               // Vulkan::chooseSwapExtent()
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes); // Vulkan::chooseSwapPresentMode()
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats); // Vulkan::chooseSwapSurfaceFormat()
};

