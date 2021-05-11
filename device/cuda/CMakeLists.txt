enable_language(CUDA)

set( TRACCC_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set( TRACCC_CUDA_SEEDING_SOURCE_DIR ${TRACCC_CUDA_DIR}/src/seeding )
set( TRACCC_CUDA_UTILS_SOURCE_DIR ${TRACCC_CUDA_DIR}/src/utils )

add_subdirectory(include/cuda/utils)
add_subdirectory(include/cuda/seeding)

add_library(
    traccc_cuda
    SHARED
    src/details/sparse_ccl.cu
    src/component_connection.cpp
)
target_link_libraries(traccc_cuda INTERFACE
  traccc::cuda::utils
  traccc::cuda::seeding
    PUBLIC
    traccc::core
    vecmem::core
    vecmem::cuda
)
add_library(traccc::cuda ALIAS traccc_cuda)


set_property(
    TARGET traccc_cuda
    PROPERTY
    CUDA_ARCHITECTURES 60 72 75
)

target_compile_options(
    traccc_cuda
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
)

target_include_directories(
    traccc_cuda
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
