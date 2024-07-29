# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()

# Tell the user what's happening.
message( STATUS "Building CCCL as part of the TRACCC project" )

# Declare where to get Thrust from.
set( TRACCC_CCCL_SOURCE
   "URL;https://github.com/NVIDIA/cccl/archive/refs/tags/v2.5.0.tar.gz;URL_MD5;7d74b60294545f6543d80483bf10b4d8"
   CACHE STRING "Source for CCCL, when built as part of this project" )
mark_as_advanced( TRACCC_CCCL_SOURCE )
FetchContent_Declare( CCCL ${TRACCC_CCCL_SOURCE} )

# Get it into the current directory.
FetchContent_MakeAvailable( CCCL )
