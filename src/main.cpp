/**
 * @file main.cpp
 * @brief Entry point for the Accelerender Vulkan-based renderer.
 *
 * Handles platform-specific Vulkan loader setup (macOS), initializes
 * the VulkanRenderer, runs the rendering loop, and ensures proper
 * cleanup in case of exceptions.
 *
 * @authors Finley Deevy, Eric Newton
 * @date 2025-11-10 (Updated)
 */

#include "render.hpp"

/**
 * @brief Entry point for the Accelerender application.
 *
 * Handles platform-specific Vulkan loader configuration (macOS), initializes
 * the VulkanRenderer class, runs the main rendering loop, and performs
 * exception-safe cleanup.
 *
 * @return EXIT_SUCCESS if the application runs successfully, otherwise
 * EXIT_FAILURE.
 */
int main() {
#ifdef __APPLE__
#include <cstdlib>
#include <iostream>
#endif

#ifdef __APPLE__
  // On macOS, manually set the dynamic loader path to the Vulkan SDK dylib
  const char *sdkPath = std::getenv("VULKAN_SDK");
  if (sdkPath) {
    // Construct full path to Vulkan loader dynamic library
    std::string dylibPath =
        std::string(sdkPath) + "/macOS/lib/libvulkan.1.dylib";

    // Override DYLD_LIBRARY_PATH to point to Vulkan loader
    setenv("DYLD_LIBRARY_PATH", (std::string(sdkPath) + "/macOS/lib").c_str(),
           1);

    std::cout << "Using Vulkan loader from: " << dylibPath << std::endl;
  } else {
    // Warn user if VULKAN_SDK environment variable is missing
    std::cerr << "Warning: VULKAN_SDK not set. Vulkan may fail to load."
              << std::endl;
  }
#endif

  // Create Accelerender application object
  VulkanRenderer app;

  try {
    // Run Accelerender initialization, main loop, and rendering
    app.run();
  } catch (const std::exception &e) {
    // Catch any exceptions thrown during setup or rendering
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // Normal exit if everything ran successfully
  return EXIT_SUCCESS;
}
