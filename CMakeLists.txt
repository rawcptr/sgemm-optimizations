cmake_minimum_required(VERSION 3.24)
project(sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
include(FetchContent)


add_executable(${PROJECT_NAME}
  src/main.cu
  src/runner.cu
)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        12.1.0)
FetchContent_MakeAvailable(fmt)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
find_package(CUDAToolkit REQUIRED)
target_compile_options(${PROJECT_NAME} PRIVATE 
  $<$<CXX_COMPILER_ID:MSVC>:/O2>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2>
)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas CUDA::cudart fmt::fmt)

# file(CREATE_LINK
#     "${CMAKE_BINARY_DIR}/compile_commands.json"
#     "${CMAKE_SOURCE_DIR}/compile_commands.json"
#     SYMBOLIC
# )
