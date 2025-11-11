#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include "Vertex.hpp"
#include <cstddef>
#include <functional>

/**
 * @file VertexHash.hpp
 * @brief Provides a 'std::hash' specialization for the Vertex struct.
 *
 * This header defines a custom hash functor that enables `Vertex` objects
 * to be used as keys in unordered associative containers such as
 * 'std::unordered_set` and 'std::unordered_map'.
 *
 * The hash function combines position, color, and texture coordinate data
 * to generate a unique hash value consistent with `Vertex::operator==`.
 *
 * @note If you modify the layout or members of `Vertex`, this hash
 *       function must be updated to maintain consistency.
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
 */
template <> struct std::hash<Vertex> {
    /**
    * @brief Computes a combined hash for a Vertex object.
    *
    * Each attribute (position, color, and texCoord) is hashed individually
    * using `std::hash` and then combined via bitwise XOR and shifts
    * to produce a well-distributed composite hash.
    *
    * @param vertex The Vertex instance to hash.
    * @return A size_t representing the hash value.
    */
    size_t operator()(Vertex const &vertex) const noexcept {
        // Hash each vector field individually.
        size_t posHash = std::hash<glm::vec3>()(vertex.position);
        size_t colorHash = std::hash<glm::vec3>()(vertex.color);
        size_t texHash = std::hash<glm::vec2>()(vertex.texCoord);

        // Combine using XOR and bit shifts to minimize collisions.
        return ((posHash ^ (colorHash << 1)) >> 1) ^ (texHash << 1);
    }
};
