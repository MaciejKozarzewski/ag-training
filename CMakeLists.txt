cmake_minimum_required (VERSION 3.9)
project(dataset_utils LANGUAGES CXX)


set(MINML_BACKEND_PATH "/home/maciek/cpp_workspace/minml/" CACHE STRING "Path to ml backend library")
set(AG_PATH "/home/maciek/cpp_workspace/AlphaGomoku/" CACHE STRING "Path to main AG library")
set(CMAKE_DEBUG_POSTFIX "_d" CACHE STRING "Choose debug postfix")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type")


find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

# List all your code files here
add_library(dataset_utils SHARED csrc/dataset_utils.cpp)
# add_executable(dataset_utils csrc/dataset_utils.cpp)

set_target_properties(dataset_utils PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(dataset_utils PUBLIC "${AG_PATH}/include/")
target_include_directories(dataset_utils PUBLIC "${MINML_BACKEND_PATH}/include/")

target_link_libraries(dataset_utils PUBLIC "${AG_PATH}/build/Release/lib/libAlphaGomoku_cpu.a")
target_link_libraries(dataset_utils PUBLIC "${MINML_BACKEND_PATH}/build/cpu/Release/lib/libMinML_cpu.a")
target_link_libraries(dataset_utils PUBLIC ${TORCH_LIBRARIES} Python::Python)
target_compile_options(dataset_utils PUBLIC -fPIC)

if (WIN32)
	target_link_libraries(dataset_utils PUBLIC pthread)
	target_link_libraries(dataset_utils PUBLIC "${PROJECT_SOURCE_DIR}/extern/libzlibstatic.a")
else()
	target_link_libraries(dataset_utils PUBLIC z pthread)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(dataset_utils PUBLIC OpenMP::OpenMP_CXX)
else()
	message(FATAL_ERROR "no OpenMP")
endif()