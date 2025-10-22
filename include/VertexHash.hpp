#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include "Vertex.hpp"
#include <cstddef>
#include <functional>

/**
 * @file VertexHash.hpp
 * @brief Defines a hash specialization for the Vertex struct, enabling its use in hash-based containers.
 *
 * This header provides a custom `std::hash` specialization for the `Vertex` type,
 * allowing `Vertex` instances to be stored in standard containers such as
 * `std::unordered_set` and `std::unordered_map`.
 *
 * @note The hash function must be consistent with `Vertex::operator==`.
 *       If you modify the fields or order of the `Vertex` struct, update this hash accordingly.
 *
 * @see Vertex
 * @see std::unordered_set
 * @see std::unordered_map
 *
 * @code
 * std::unordered_set<Vertex> uniqueVertices;
 * Vertex v1{{1.0f, 2.0f, 3.0f}, {0.5f, 0.5f, 0.5f}, {0.0f, 1.0f}};
 * uniqueVertices.insert(v1);
 * @endcode
 *
 * @authors
 * Finley Deevy, Eric Newton
 * @date 2025-10-20
 * @version 1.0
 */
template <> struct std::hash<Vertex> {
    size_t operator()(Vertex const &vertex) const noexcept {
        // Combine the individual attribute hashes using bitwise XOR and shifts.
        // This ensures decent distribution across buckets while remaining simple.
        size_t posHash = std::hash<glm::vec3>()(vertex.position);
        size_t colorHash = std::hash<glm::vec3>()(vertex.color);
        size_t texHash = std::hash<glm::vec2>()(vertex.texCoord);

        return ((posHash ^ (colorHash << 1)) >> 1) ^ (texHash << 1);
    }
};
