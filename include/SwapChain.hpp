#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vector>

/**
 * @file SwapChain.hpp
 * @brief Declares the SwapChain class for encapsulating Vulkan swap chain creation and management.
 *
 * The **SwapChain** class is designed to handle the setup, recreation, and cleanup of
 * a Vulkan swap chain using the RAII (Resource Acquisition Is Initialization) pattern.
 * It abstracts low-level Vulkan logic for presentation image handling, making the
 * rendering pipeline more modular and maintainable.
 *
 * @details
 * ⚠️ **Note:** As of this version, the `SwapChain` class is **not yet integrated**
 * into the main Vulkan rendering pipeline (`main.cpp`). The current pipeline
 * still uses legacy swap chain logic from the original `Vulkan` class. Integration
 * will occur once refactoring is complete to support modularized components.
 *
 * Key responsibilities:
 * - Create and manage the Vulkan swap chain
 * - Select optimal surface format, presentation mode, and extent
 * - Manage associated image views
 *
 * @see vk::raii::SwapchainKHR
 * @see vk::SurfaceFormatKHR
 * @see Vulkan (original implementation)
 *
 * @authors
 * Finley Deevy, Eric Newton
 * @date 2025-10-20
 * @version 0.9
 *
 * @ingroup Rendering
 */
class SwapChain {
public:
    /**
     * @brief Constructs a SwapChain object with references to the Vulkan core objects.
     *
     * @param device Logical Vulkan device.
     * @param gpu Physical device used for swap chain creation.
     * @param surface Surface associated with the swap chain.
     *
     * @note This constructor does not immediately create the swap chain;
     *       call `create()` after initialization.
     */
    SwapChain(vk::raii::Device &device,
              vk::raii::PhysicalDevice &gpu,
              vk::raii::SurfaceKHR &surface)
            : device(device), physicalGPU(gpu), surface(surface) {}

    /** @brief Creates the Vulkan swap chain and associated image views. */
    void create();   ///< Originally `Vulkan::createSwapChain()`

    /** @brief Recreates the swap chain, typically after window resizing or surface loss. */
    void recreate(); ///< Originally `Vulkan::recreateSwapChain()`

    /** @brief Cleans up swap chain resources before destruction or recreation. */
    void cleanup();  ///< Originally `Vulkan::cleanupSwapChain()`

    // -------------------
    // Accessors
    // -------------------

    /** @return Vector of swap chain image views for rendering. */
    const std::vector<vk::raii::ImageView> &getImageViews() const { return swapChainImageViews; }

    /** @return Image format used for the swap chain. */
    vk::Format getImageFormat() const { return swapChainImageFormat; }

    /** @return Swap chain image extent (width/height). */
    vk::Extent2D getExtent() const { return swapChainExtent; }

private:
    // --------------------------------------
    // References to Vulkan core objects
    // --------------------------------------

    vk::raii::Device &device;             ///< Logical device (Vulkan::device)
    vk::raii::PhysicalDevice &physicalGPU;///< Physical GPU (Vulkan::physicalGPU)
    vk::raii::SurfaceKHR &surface;        ///< Window surface (Vulkan::surface)

    // --------------------------------------
    // Swap chain resources
    // --------------------------------------

    vk::raii::SwapchainKHR swapChain = nullptr;       ///< Vulkan swap chain handle.
    std::vector<vk::Image> swapChainImages;           ///< Raw swap chain images.
    std::vector<vk::raii::ImageView> swapChainImageViews; ///< Image views for rendering.
    vk::Format swapChainImageFormat = vk::Format::eUndefined; ///< Chosen image format.
    vk::Extent2D swapChainExtent;                     ///< Dimensions of the swap chain.
    vk::SurfaceFormatKHR swapChainSurfaceFormat;      ///< Chosen surface format.

    // ---------------------------------------------------------
    // Internal helpers (ported from Vulkan class)
    // ---------------------------------------------------------

    /** @brief Chooses optimal swap extent based on surface capabilities. */
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities);

    /** @brief Selects the best available presentation mode for rendering. */
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes);

    /** @brief Selects the preferred surface format (color space and pixel format). */
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats);
};
