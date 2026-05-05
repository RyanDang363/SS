cmake_minimum_required(VERSION 3.18)
project(raggers_frame_extract LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio)

add_executable(raggers_frame_extract
  src/main.cpp
  src/frame_extractor.cpp
)

target_include_directories(raggers_frame_extract PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(raggers_frame_extract PRIVATE ${OpenCV_LIBS})

target_compile_options(raggers_frame_extract PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
