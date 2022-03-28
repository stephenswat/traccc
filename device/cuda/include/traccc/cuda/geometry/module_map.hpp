/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <type_traits>

// Project include(s).
#include "traccc/definitions/primitives.hpp"

#include <vecmem/memory/cuda/device_memory_resource.hpp>

#include <cuda_runtime.h>

namespace traccc::cuda {
template <typename, typename>
class module_map_view;

template <typename K = geometry_id, typename V = transform3>
class module_map {
    public:
    /**
     * @brief Construct a module map from one created by the `io` library.
     *
     * This method constructs a module map from an existing module map
     * represented as a std::map.
     *
     * @param[in] input The existing module map to convert.
     */
    module_map(const ::traccc::module_map<K, V>& input)
    : m_n_nodes(input.m_nodes.size())
    , m_n_values(input.m_values.size())
    , m_nodes(vecmem::make_unique_alloc<module_map_node[]>(m_mr, m_n_nodes))
    , m_values(vecmem::make_unique_alloc<V[]>(m_mr, m_n_values))
    {
        cudaMemcpy(m_nodes.get(), input.m_nodes.data(), m_n_nodes * sizeof(module_map_node), cudaMemcpyHostToDevice);
        cudaMemcpy(m_values.get(), input.m_values.data(), m_n_values * sizeof(V), cudaMemcpyHostToDevice);
    }

    private:
    /**
     * @brief The internal representation of nodes in our binary search tree.
     *
     * These objects carry three pieces of data. Firstly, there is the starting
     * ID. Then, there is the size. Since the node represents a stretch of
     * consecutive IDs, we know that the node ends at `start + size`. Finally,
     * there is the index in the value array. We keep indices instead of
     * pointers to make it easier to port this code to other devices.
     */
    struct module_map_node {
        K start;
        std::size_t size;
        std::size_t index;
    };

    const std::size_t m_n_nodes;
    const std::size_t m_n_values;

    vecmem::cuda::device_memory_resource m_mr;

    /**
     * @brief The internal storage of the nodes in our binary search tree.
     *
     * This follows the well-known formalism where the root node resides at
     * index 0, while for any node at position n, the left child is at index 2n
     * + 1, and the right child is at index 2n + 2.
     */
    vecmem::unique_alloc_ptr<module_map_node[]> m_nodes;

    /**
     * @brief This vector stores the values in a contiguous manner. Our nodes
     * keep indices in this array instead of pointers.
     */
    vecmem::unique_alloc_ptr<V[]> m_values;

    /*
     * Declare the view class as our friend, so that it can access our
     * pointers.
     */
    friend class module_map_view<K, V>;
};

template <typename K = geometry_id, typename V = transform3>
class module_map_view {
    public:
    using node_t = typename module_map<K, V>::module_map_node;

    TRACCC_HOST module_map_view(const module_map<K, V> & input)
    : m_n_nodes(input.m_n_nodes)
    , m_n_values(input.m_n_values)
    , m_nodes(input.m_nodes.get())
    , m_values(input.m_values.get())
    {}

    /**
     * @brief Find a given key in the map.
     *
     * @param[in] i The key to look-up.
     *
     * @return The value associated with the given key.
     *
     * @warning This method does no bounds checking, and will result in
     * undefined behaviour if the key does not exist in the map.
     */
    TRACCC_DEVICE const V* operator[](const K& i) const {
        unsigned int n = 0;

        while (true) {
            /*
             * For memory safety, if we are out of bounds we will exit.
             */
            if (n >= m_n_nodes) {
                return nullptr;
            }

            /*
             * Retrieve the current root node.
             */
            const node_t& node = m_nodes[n];

            /*
             * If the size is zero, it is essentially an invalid node (i.e. the
             * node does not exist).
             */
            if (node.size == 0) {
                return nullptr;
            }

            /*
             * If the value we are looking for is past the start of the current
             * node, there are three possibilities. Firstly, the value might be in
             * the current node. Secondly, the value might be in the right child of
             * the current node. Thirdly, the value might not be in the map at all.
             */
            if (i >= node.start) {
                /*
                 * Next, we check if the value is within the range represented by
                 * the current node.
                 */
                if (i < node.start + node.size) {
                    /*
                     * Found it! Return a pointer to the value within the
                     * contiguous range.
                     */
                    return &m_values[node.index + (i - node.start)];
                } else {
                    /*
                     * Two possibilties remain, we need to check the right subtree.
                     */
                    n = 2 * n + 2;
                }
            }
            /*
             * If the value we want to find is less then the start of this node,
             * there are only two possibilities. Firstly, the value might be in the
             * left subtree, or the value might not be in the map at all.
             */
            else {
                n = 2 * n + 1;
            }
        }
    }

    /**
     * @brief Get the total number of modules in the module map.
     *
     * This iterates over all of the nodes in the map and sums up their sizes.
     *
     * @return The total number of modules in this module map.
     */
    TRACCC_DEVICE std::size_t size(void) const {
        return m_n_values;
    }

    TRACCC_DEVICE bool contains(const K& i) const { return operator[](i) != nullptr; }

    TRACCC_DEVICE bool empty(void) const { return m_n_values == 0; }

    const std::size_t m_n_nodes;
    const std::size_t m_n_values;
    const node_t * m_nodes;
    const V * m_values;
};
}  // namespace traccc
