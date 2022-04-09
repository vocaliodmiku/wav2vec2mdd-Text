cmake_minimum_required(VERSION 3.0.0)
project(viterbi VERSION 0.1.0)

include(FetchContent)
include(CTest)
enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
# third_party: glog
FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/v0.4.0.zip
  URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

add_subdirectory(pybind11)

# third_party: libtorch use FetchContent_Declare to download, and
# use find_package to find since libtorch is not a standard cmake project

pybind11_add_module(viterbi viterbi.cpp)

target_link_libraries(viterbi PUBLIC glog)
