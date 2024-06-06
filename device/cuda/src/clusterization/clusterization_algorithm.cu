/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// CUDA Library include(s).
#include "../utils/barrier.hpp"
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"

// Project include(s)
#include "traccc/clusterization/device/ccl_kernel.hpp"

// Vecmem include(s).
#include <cstring>
#include <vecmem/utils/copy.hpp>
#include <iostream>

namespace traccc::cuda {

namespace kernels {

/// CUDA kernel for running @c traccc::device::ccl_kernel
__global__ void ccl_kernel(
    const cell_collection_types::const_view cells_view,
    const cell_module_collection_types::const_view modules_view,
    const device::details::index_t max_cells_per_partition,
    const device::details::index_t target_cells_per_partition,
    measurement_collection_types::view measurements_view,
    vecmem::data::vector_view<unsigned int> cell_links,
    vecmem::data::vector_view<device::details::index_t> f_backup_view,
    vecmem::data::vector_view<device::details::index_t> gf_backup_view,
    vecmem::data::vector_view<unsigned char> adjc_backup_view,
    vecmem::data::vector_view<device::details::index_t> adjv_backup_view,
    vecmem::device_atomic_ref<unsigned int> backup_mutex) {

    __shared__ unsigned int partition_start, partition_end;
    __shared__ unsigned int outi;
    extern __shared__ device::details::index_t shared_v[];
    vecmem::data::vector_view<device::details::index_t> f_view{
        max_cells_per_partition, shared_v};
    vecmem::data::vector_view<device::details::index_t> gf_view{
        max_cells_per_partition, shared_v + max_cells_per_partition};
    traccc::cuda::barrier barry_r;

    device::ccl_kernel(threadIdx.x, blockDim.x, blockIdx.x, cells_view,
                       modules_view, max_cells_per_partition,
                       target_cells_per_partition, partition_start,
                       partition_end, outi, f_view, gf_view, f_backup_view,
                       gf_backup_view, adjc_backup_view, adjv_backup_view,
                       backup_mutex, barry_r, measurements_view, cell_links);
}

}  // namespace kernels

clusterization_algorithm::clusterization_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, stream& str,
    const config_type& config)
    : m_mr(mr),
      m_copy(copy),
      m_stream(str),
      m_config(config),
      m_f_backup(
          m_config.backup_partition_size * m_config.target_cells_per_partition,
          m_mr.main),
      m_gf_backup(
          m_config.backup_partition_size * m_config.target_cells_per_partition,
          m_mr.main),
      m_adjc_backup(
          m_config.backup_partition_size * m_config.target_cells_per_partition,
          m_mr.main),
      m_adjv_backup(m_config.backup_partition_size * 8 *
                        m_config.target_cells_per_partition,
                    m_mr.main),
      m_backup_mutex(vecmem::make_unique_alloc<unsigned int>(m_mr.main)) {
    std::cout << "Hello: " << (m_config.backup_partition_size * 8 *
                        m_config.target_cells_per_partition) << std::endl;
    m_copy.get().setup(m_f_backup)->ignore();
    m_copy.get().setup(m_gf_backup)->ignore();
    m_copy.get().setup(m_adjc_backup)->ignore();
    m_copy.get().setup(m_adjv_backup)->ignore();
    TRACCC_CUDA_ERROR_CHECK(cudaMemset(
        m_backup_mutex.get(), 0,
        sizeof(std::remove_extent_t<decltype(m_backup_mutex)::element_type>)));
}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const cell_collection_types::const_view& cells,
    const cell_module_collection_types::const_view& modules) const {

    // Get a convenience variable for the stream that we'll be using.
    cudaStream_t stream = details::get_stream(m_stream);

    // Get the number of cells
    const cell_collection_types::view::size_type num_cells =
        m_copy.get().get_size(cells);

    // Create the result object, overestimating the number of measurements.
    measurement_collection_types::buffer measurements{
        num_cells, m_mr.main, vecmem::data::buffer_type::resizable};
    m_copy.get().setup(measurements)->ignore();

    // If there are no cells, return right away.
    if (num_cells == 0) {
        return measurements;
    }

    // Create buffer for linking cells to their measurements.
    //
    // @todo Construct cell clusters on demand in a member function for
    // debugging.
    //
    vecmem::data::vector_buffer<unsigned int> cell_links(num_cells, m_mr.main);
    m_copy.get().setup(cell_links)->ignore();

    // Launch ccl kernel. Each thread will handle a single cell.
    const device::details::ccl_kernel_helper helper{
        static_cast<device::details::index_t>(
            m_config.target_cells_per_partition),
        static_cast<device::details::index_t>(m_config.target_cells_per_thread),
        num_cells};
    kernels::ccl_kernel<<<helper.num_partitions, helper.threads_per_partition,
                          2 * helper.max_cells_per_partition *
                              sizeof(device::details::index_t),
                          stream>>>(
        cells, modules, helper.max_cells_per_partition,
        static_cast<device::details::index_t>(
            m_config.target_cells_per_partition),
        measurements, cell_links, m_f_backup, m_gf_backup, m_adjc_backup,
        m_adjv_backup, *m_backup_mutex);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());

    // Return the reconstructed measurements.
    return measurements;
}

}  // namespace traccc::cuda
