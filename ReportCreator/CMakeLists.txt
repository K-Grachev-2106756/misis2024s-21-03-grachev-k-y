cmake_minimum_required(VERSION 3.10)
project(ReportCreator)

add_library(ReportCreator STATIC ReportCreator.cpp)
target_include_directories(ReportCreator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})