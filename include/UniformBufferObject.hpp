#pragma once
#include <glm/glm.hpp>

/**
 * @struct UniformBufferObject
 * @brief Stores the model, view, and projection matrices for Vulkan shaders.
 *
 * Typically uploaded to a uniform buffer and used in vertex or fragment shaders
 * to transform vertices from model space to clip space.
 *
 * @author Finley Deevy, Eric Newton
 * @date 2025-10-20
 * @version 1.0
 *
 * @note The struct layout must match the shader’s expectations (e.g., std140 layout).
 */
struct UniformBufferObject {
    /** @brief Model transformation matrix. */
    glm::mat4 model;

    /** @brief View (camera) transformation matrix. */
    glm::mat4 view;

    /** @brief Projection matrix (perspective or orthographic). */
    glm::mat4 proj;
};
