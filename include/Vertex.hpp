#pragma once

#include <glm/glm.hpp>               // for glm::vec2, vec3
#include <vulkan/vulkan_raii.hpp>    // for vk::VertexInputBindingDescription, etc.
#include <array>                     // for std::array
#include <functional>                // for std::hash

/**
 * @struct Vertex
 * @brief Represents a single vertex in a 3D model for Vulkan rendering.
 *
 * This struct contains the essential attributes for a vertex:
 * - Position in 3D space
 * - Color
 * - Texture coordinates
 *
 * It also provides Vulkan-specific descriptions for binding and
 * attribute layout, used during pipeline creation.
 *
 * @author Finley Deevy, Eric Newton
 * @date 2025-10-20
 * @version 1.0
 *
 * @note This struct is tightly coupled with Vulkan shaders. Ensure that
 *       the shader input locations match the attribute descriptions.
 *
 * @warning Changing the order or types of fields requires updating
 *          getBindingDescription() and getAttributeDescriptions().
 *
 * @see vk::VertexInputBindingDescription
 * @see vk::VertexInputAttributeDescription
 *
 * @code
 * // Example usage:
 * std::vector<Vertex> vertices;
 * vk::VertexInputBindingDescription binding = Vertex::getBindingDescription();
 * auto attributes = Vertex::getAttributeDescriptions();
 * @endcode
 */
struct Vertex {
    /** @brief 3D position of the vertex. */
    glm::vec3 position;

    /** @brief Color of the vertex. Typically RGB values in [0,1]. */
    glm::vec3 color;

    /** @brief 2D texture coordinates (U,V) for sampling textures. */
    glm::vec2 texCoord;

    /**
     * @brief Returns the Vulkan binding description for this vertex type.
     *
     * The binding description tells Vulkan how to read vertex data
     * from a buffer. Here, the binding index is 0, stride is sizeof(Vertex),
     * and input rate is per-vertex.
     *
     * @return vk::VertexInputBindingDescription describing the vertex layout.
     *
     * @note Binding index must match the binding used in the Vulkan pipeline.
     */
    static vk::VertexInputBindingDescription getBindingDescription() {
        return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }

    /**
     * @brief Returns Vulkan attribute descriptions for the vertex fields.
     *
     * Each attribute description specifies:
     * - location (shader input)
     * - binding (buffer binding index)
     * - format (data type and size)
     * - offset (byte offset in the struct)
     *
     * @return std::array of 3 vk::VertexInputAttributeDescription objects
     *         corresponding to position, color, and texCoord.
     *
     * @note Locations must match the vertex shader inputs.
     * @see getBindingDescription()
     */
    static std::array<vk::VertexInputAttributeDescription, 3>
    getAttributeDescriptions() {
        return {vk::VertexInputAttributeDescription(
                0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)),
                vk::VertexInputAttributeDescription(
                        1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
                vk::VertexInputAttributeDescription(
                        2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))};
    }

    /**
     * @brief Equality operator to compare two vertices.
     *
     * Checks if position, color, and texture coordinates are all equal.
     *
     * @param other The vertex to compare against.
     * @return true if all fields match, false otherwise.
     *
     * @note Useful for de-duplicating vertices in vertex buffers.
     * @see std::unordered_set
     */
    bool operator==(const Vertex &other) const {
        return position == other.position && color == other.color &&
               texCoord == other.texCoord;
    }
};