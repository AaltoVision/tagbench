project(tagbench)
cmake_minimum_required(VERSION 3.5.0)

# Compile options etc.
set(CPP_STANDARD cxx_std_17)

# Dependencies
find_package(OpenCV REQUIRED)

add_subdirectory(libs/apriltags-cpp-win/)
set(APRILTAGS_INCLUDE_DIRS "libs/apriltags-cpp-win/src")

# Data preprocessor
add_executable(input_data_preprocessor input_data_preprocessor.cpp)
target_include_directories(input_data_preprocessor PUBLIC "libs/" ${APRILTAGS_INCLUDE_DIRS})
target_link_libraries(input_data_preprocessor ${OpenCV_LIBS} apriltags)
target_compile_features(input_data_preprocessor PUBLIC ${CPP_STANDARD})

# tagbench (the main app)
set(SRC tagbench.cpp)
add_executable(tagbench ${SRC})
target_include_directories(tagbench PUBLIC "libs/" ${APRILTAGS_INCLUDE_DIRS})
target_link_libraries(tagbench ${OpenCV_LIBS} apriltags)
target_compile_features(tagbench PUBLIC ${CPP_STANDARD})
target_compile_definitions(tagbench PRIVATE _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
