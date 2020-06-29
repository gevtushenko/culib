set(NCCL_INC_PATHS
        /usr/include
        /usr/local/include)

set(NCCL_LIB_PATHS
        /lib
        /lib64
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64)

find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS ${NCCL_INC_PATHS})
find_library(NCCL_LIBRARIES NAMES nccl PATHS ${NCCL_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARIES)