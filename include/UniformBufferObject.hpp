#pragma once
#include <glm/glm.hpp>

/**
 * @file UniformBufferObject.hpp
 * @brief Defines the UniformBufferObject struct used to pass transformation matrices to shaders.
 *
 * The **UniformBufferObject (UBO)** encapsulates the transformation matrices required
 * by the graphics pipeline to convert vertex positions through model, view, and
 * projection spaces. It is typically updated once per frame and bound to a uniform
 * buffer accessible by the vertex shader.
 *
 * @struct UniformBufferObject
 * @ingroup Rendering
 *
 * @note Ensure this struct follows Vulkan's std140 alignment rules. Matrices are
 *       column-major and should match the layout qualifiers in GLSL.
 *
 * @see vk::DescriptorSet
 * @see glm::mat4
 * @see https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap15.html#interfaces-resources-standard-block-layout
 *
 * @code
 * // Example usage:
 * UniformBufferObject ubo{};
 * ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(45.0f), {0.0f, 0.0f, 1.0f});
 * ubo.view  = glm::lookAt({2.0f, 2.0f, 2.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f});
 * ubo.proj  = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 10.0f);
 * @endcode
 *
 * @authors
 * Finley Deevy, Eric Newton
 * @date 2025-10-20
 * @version 1.1
 */
struct UniformBufferObject {
    /** @brief Model matrix: transforms local object coordinates to world space. */
    glm::mat4 model;

    /** @brief View matrix: defines the camera position and orientation in the scene. */
    glm::mat4 view;

    /** @brief Projection matrix: applies perspective or orthographic projection. */
    glm::mat4 proj;
};
