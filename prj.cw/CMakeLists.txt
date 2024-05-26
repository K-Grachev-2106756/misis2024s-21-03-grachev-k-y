cmake_minimum_required(VERSION 3.20)
project(Detection)

set(SOURCE_FILES BlastCellDetection.cpp main.cpp)

add_library(BlastCellDetectionLib STATIC BlastCellDetection.cpp)
target_include_directories(BlastCellDetectionLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(BlastCellDetectionLib PRIVATE ${OpenCV_LIBS})

add_executable(main main.cpp)
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(main PRIVATE BlastCellDetectionLib ${OpenCV_LIBS})