cmake_minimum_required(VERSION 3.17)
project(sdk_demo)

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")


set(TVM_ROOT /home/leo/workspace/dev/tvm)
set(DMLC_CORE ${TVM_ROOT}/3rdparty/dmlc-core)

include_directories(${TVM_ROOT}/include)
include_directories(${DMLC_CORE}/include)
include_directories(${TVM_ROOT}/3rdparty/dlpack/include)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(CUDA libs: ${CUDA_LIBRARIES})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(centerface_demo cpp_deploy.cc tvm_runtime_pack.cc)
target_link_libraries(centerface_demo ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES} rt cuda)