/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc {
struct clustering_config {
    clustering_config(unsigned int _target_cells_per_partition = 2048,
                      unsigned int _target_cells_per_thread = 8,
                      unsigned int _backup_partition_size = 256 * 4096)
        : target_cells_per_partition(_target_cells_per_partition),
          target_cells_per_thread(_target_cells_per_thread),
          backup_partition_size(_backup_partition_size) {}

    unsigned int target_cells_per_partition;
    unsigned int target_cells_per_thread;
    unsigned int backup_partition_size;
};
}  // namespace traccc
