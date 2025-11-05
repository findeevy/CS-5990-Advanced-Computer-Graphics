#pragma once

#include <glm/glm.hpp>               // for glm::vec2, glm::vec3
#include <vulkan/vulkan_raii.hpp>    // for vk::VertexInputBindingDescription, etc.
#include <array>                     // for std::array
#include <functional>                // for std::hash

/**
 * @file Vertex.hpp
 * @brief Defines the Vertex struct used in Vulkan rendering pipelines.
 *
 * The Vertex struct encapsulates all per-vertex attributes required
 * for rendering — position, color, and texture coordinates — and
 * provides helper functions for describing these attributes to Vulkan.
 *
 * @details
 * This struct is designed to align precisely with shader input layouts.
 * It provides static functions to retrieve binding and attribute
 * descriptions that are used during Vulkan pipeline creation.
 *
 * @warning If you modify field order or types, update both
 *          getBindingDescription() and getAttributeDescriptions()
 *          to maintain compatibility with your shaders.
 *
 * @note Shader input locations must match the attribute descriptions.
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
 *
 * @authors
 * Eric Newton, Finley Deevy
 * @date 2025-10-20
 * @version 1.0
 */
struct Vertex {
    /** @brief 3D position of the vertex in model space. */
    glm::vec3 position;

    /** @brief Vertex color, typically represented as RGB values in [0, 1]. */
    glm::vec3 color;

    /** @brief 2D texture coordinates (U, V) for sampling textures. */
    glm::vec2 texCoord;

    /**
     * @brief Returns the Vulkan binding description for this vertex layout.
     *
     * The binding description specifies how the vertex buffer is consumed
     * by the input assembly stage. This implementation assumes a single
     * vertex buffer bound at index 0, with a per-vertex input rate.
     *
     * @return A vk::VertexInputBindingDescription describing buffer layout.
     *
     * @note Binding index must match the value used in the pipeline creation.
     */
    static vk::VertexInputBindingDescription getBindingDescription() {
        return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
    }

    /**
     * @brief Returns Vulkan attribute descriptions for vertex attributes.
     *
     * Each description specifies the mapping between struct fields and
     * shader input locations. The returned array includes attributes for
     * position (location 0), color (location 1), and texCoord (location 2).
     *
     * @return std::array of vk::VertexInputAttributeDescription entries.
     *
     * @note These locations must match the layout qualifiers in your vertex shader.
     * @see getBindingDescription()
     */
    static std::array<vk::VertexInputAttributeDescription, 3>
    getAttributeDescriptions() {
        return {
                vk::VertexInputAttributeDescription(
                        0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)),
                vk::VertexInputAttributeDescription(
                        1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
                vk::VertexInputAttributeDescription(
                        2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
        };
    }

    /**
     * @brief Equality comparison operator.
     *
     * Determines whether two Vertex instances contain identical position,
     * color, and texture coordinate data.
     *
     * @param other The vertex to compare with.
     * @return True if all fields are equal; otherwise false.
     *
     * @note This operator enables vertices to be de-duplicated in
     *       hash-based containers such as std::unordered_set.
     */
    bool operator==(const Vertex &other) const {
        return position == other.position &&
               color == other.color &&
               texCoord == other.texCoord;
    }
};
