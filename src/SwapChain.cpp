#include "SwapChain.hpp"
#include "VulkanUtils.hpp"
#include <algorithm> // for std::max
#include <limits>    // for std::numeric_limits
#include <vector>

/**
 * @brief Creates the Vulkan swap chain and its associated image views.
 *
 * High-level steps:
 * 1. Query surface capabilities and available formats/present modes from the
 * GPU.
 * 2. Choose a surface format, present mode, and extent (resolution).
 * 3. Build a vk::SwapchainCreateInfoKHR describing how the swap chain should
 * behave.
 * 4. Construct the RAII-wrapped vk::raii::SwapchainKHR object.
 * 5. Retrieve raw swap chain images and create RAII-wrapped image objects so we
 * can create corresponding image views (one per swap chain image).
 *
 * Important notes:
 * - This function can throw on Vulkan errors (vk::SystemError) when creating
 * the swapchain or image views. Callers are responsible for handling exceptions
 * appropriately.
 * - RAII objects hold references to the device; do not destroy the device
 * before the swapchain.
 *
 * @throws vk::SystemError if swap chain or image view creation fails.
 * @see recreate()
 * @see cleanup()
 */
void SwapChain::create() {
  // Query the current surface capabilities for the given physical GPU and
  // surface. This provides supported min/max image counts, currentTransform,
  // extents, etc.
  auto surfaceCapabilities = physicalGPU.getSurfaceCapabilitiesKHR(*surface);

  // Pick the best surface format (color format + color space) available.
  // This determines pixel format and color space used when presenting images.
  auto chosenSurfaceFormat =
      chooseSwapSurfaceFormat(physicalGPU.getSurfaceFormatsKHR(*surface));
  swapChainImageFormat = chosenSurfaceFormat.format;
  swapChainSurfaceFormat = chosenSurfaceFormat;
  auto swapChainColorSpace = chosenSurfaceFormat.colorSpace;

  // Choose the resolution (extent) for swapchain images.
  // If the windowing system specifies a fixed extent we must use it.
  swapChainExtent = chooseSwapExtent(surfaceCapabilities);

  // Decide how many images the swapchain should have.
  // Using more images can increase GPU/CPU parallelism; many examples use 2
  // or 3. Here we prefer 3 for extra buffering, but respect the device's
  // maximum.
  uint32_t minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
  if (surfaceCapabilities.maxImageCount > 0 &&
      minImageCount > surfaceCapabilities.maxImageCount) {
    // Cap at the GPU's maximum if it specifies one (0 means no limit).
    minImageCount = surfaceCapabilities.maxImageCount;
  }

  // Choose a present mode based on what the platform and driver support.
  // mailbox is preferred (low latency, no tearing), fallback to FIFO which is
  // guaranteed.
  auto presentMode =
      chooseSwapPresentMode(physicalGPU.getSurfacePresentModesKHR(*surface));

  // Fill in the standard fields for swapchain creation.
  // Each field here controls a specific behavior:
  // - surface: which window/screen to present to
  // - minImageCount: number of images in swapchain
  // - imageFormat / imageColorSpace: pixel layout
  // - imageExtent: resolution
  // - imageArrayLayers: usually 1 for 2D rendering
  // - imageUsage: how we intend to use the images (color attachment for
  // rendering)
  // - imageSharingMode: exclusive is fastest for single-queue families
  // - preTransform: rotation/transform applied by the surface (use
  // currentTransform)
  // - compositeAlpha: how alpha is handled with other windows (opaque by
  // default)
  // - presentMode: vsync / mailbox / immediate behavior
  // - clipped: allow discarding pixels obscured by other windows (true)
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

  // Create the RAII swapchain object. vk::raii::SwapchainKHR will free itself
  // when it goes out of scope or is assigned nullptr (see cleanup()).
  swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);

  // Retrieve the raw vk::Image handles owned by the swapchain.
  swapChainImages = swapChain.getImages();

  // Wrap the raw images in RAII Image objects so we can safely manage their
  // lifetime when creating ImageViews. Note: these RAII images simply provide
  // RAII semantics for the handle, they do not allocate memory — swapchain
  // images are owned by the swapchain.
  std::vector<vk::raii::Image> raiiSwapChainImages;
  raiiSwapChainImages.reserve(swapChainImages.size());
  for (auto &img : swapChainImages) {
    // Construct a RAII wrapper around the raw image handle (img).
    // We pass the logical device and the VkImage handle.
    raiiSwapChainImages.emplace_back(device, img);
  }

  // Create an image view for each swapchain image. Image views describe how
  // shaders will access image data (format, subresource range, etc.). We store
  // RAII-wrapped ImageView objects in swapChainImageViews. Clearing here
  // ensures repeated create() calls (e.g., during recreation) do not leak
  // previous views.
  swapChainImageViews.clear();
  swapChainImageViews.reserve(raiiSwapChainImages.size());
  for (auto &image : raiiSwapChainImages) {
    // Use the utility helper to create a view; this centralizes consistent
    // creation flags.
    swapChainImageViews.emplace_back(
        vkutils::createImageView(device, image, swapChainImageFormat,
                                 vk::ImageAspectFlagBits::eColor, 1));
  }

  // Note: raiiSwapChainImages temporaries will be destroyed at the end of this
  // function, but the ImageView objects created above hold the necessary
  // references because they are created using the device and underlying VkImage
  // handles. Ensure the device and swapChain outlive the image views; otherwise
  // you will have dangling handles.
}

/**
 * @brief Cleans up swap chain resources.
 *
 * Using RAII containers means most cleanup is done by destructors, but we
 * still:
 * - Clear the vector of image views (explicitly destroying the RAII view
 * objects).
 * - Reset the swapChain RAII handle to release the underlying VkSwapchainKHR.
 *
 * This is called before creating a new swap chain or when shutting down the
 * renderer.
 *
 * @see recreate()
 */
void SwapChain::cleanup() {
  // Explicitly destroy image views now (clear invokes destructors of the RAII
  // objects).
  swapChainImageViews.clear();

  // Reset the swapChain RAII wrapper. Assigning nullptr or default-constructed
  // RAII wrapper releases the VkSwapchainKHR resource immediately.
  swapChain = nullptr;
}

/**
 * @brief Recreates the swap chain (e.g., after a window resize event).
 *
 * Typical usage pattern:
 * - On framebuffer resize, call recreate() to re-query surface capabilities and
 *   rebuild the swap chain and image views to match the new window size.
 *
 * Note: More robust implementations may need to wait for the device to become
 * idle or to free other resources (framebuffers, pipelines) that depend on the
 * swapchain.
 *
 * @see create()
 * @see cleanup()
 */
void SwapChain::recreate() {
  // Simple approach: destroy existing resources then create new ones.
  // In a production renderer you'd likely wait for the device to be idle and
  // free dependent objects (framebuffers, command buffers, etc.) first.
  cleanup();
  create();
}

/**
 * @brief Chooses the swap chain image extent based on surface capabilities.
 *
 * Explanation:
 * - Some window systems (e.g., Android) provide a fixed currentExtent; in that
 * case the driver requires you to use that exact extent.
 * - If currentExtent has the special value of UINT32_MAX, the surface allows
 * the app to choose an extent. We fall back to a default (800x600) and clamp it
 * to the supported min/max extents provided by the surface capabilities.
 *
 * @param capabilities The surface capabilities retrieved from the physical
 * device.
 * @return The chosen vk::Extent2D extent for swap chain images.
 */
vk::Extent2D
SwapChain::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
  // If the surface sets a fixed extent, use it.
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    // Fallback default; in a real application you should query the actual
    // window size (e.g., from GLFW) and use that instead of a hard-coded
    // 800x600.
    vk::Extent2D actualExtent = {800, 600}; // Default fallback

    // Clamp the fallback to the allowed range.
    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);
    return actualExtent;
  }
}

/**
 * @brief Chooses the preferred swap chain present mode.
 *
 * Present modes:
 * - eMailbox: low latency, no tearing (if supported); similar to triple
 * buffering.
 * - eImmediate: may present images immediately and can tear (not recommended).
 * - eFifo: guaranteed to be available on all platforms; behaves like vsync.
 *
 * We prefer mailbox when available, otherwise fallback to FIFO for portability.
 *
 * @param availablePresentModes List of supported present modes returned by the
 * driver.
 * @return The chosen vk::PresentModeKHR (mailbox preferred, otherwise FIFO).
 */
vk::PresentModeKHR SwapChain::chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> &availablePresentModes) {
  for (const auto &mode : availablePresentModes) {
    if (mode == vk::PresentModeKHR::eMailbox) {
      return mode;
    }
  }
  // FIFO is required to be supported on all Vulkan implementations.
  return vk::PresentModeKHR::eFifo;
}

/**
 * @brief Chooses the optimal swap chain surface format.
 *
 * Preferred format:
 * - VK_FORMAT_B8G8R8A8_SRGB with sRGB nonlinear color space for correct color
 *   reproduction on most platforms.
 *
 * If that specific format is not available we return the first available
 * option. In a more advanced implementation you could score formats based on
 * color depth, linear vs sRGB, and device support.
 *
 * @param availableFormats List of supported surface formats.
 * @return The chosen vk::SurfaceFormatKHR.
 */
vk::SurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
  for (const auto &format : availableFormats) {
    if (format.format == vk::Format::eB8G8R8A8Srgb &&
        format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return format;
    }
  }
  // Fallback: return whatever the driver exposes first.
  return availableFormats[0];
}
