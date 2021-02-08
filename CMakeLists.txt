project(tagbench)
cmake_minimum_required(VERSION 3.5.0)

# Compile options etc.
# set(CPP_STANDARD cxx_std_14)
set(CPP_STANDARD cxx_std_17)

# Dependencies
find_package(OpenCV)

add_subdirectory(libs/apriltags-cpp-win/)
set(APRILTAGS_INCLUDE_DIRS "libs/apriltags-cpp-win/src")

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(libs/json)

# Data preprocessor
add_executable(input_data_preprocessor input_data_preprocessor.cpp)
target_include_directories(input_data_preprocessor PUBLIC ${APRILTAGS_INCLUDE_DIRS})
target_link_libraries(input_data_preprocessor PRIVATE
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    apriltags
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>)
target_compile_features(input_data_preprocessor PUBLIC ${CPP_STANDARD})

# tagbench (the main app)
set(SRC tagbench.cpp pose_optimizer.cpp pose_optimizer.h)
add_executable(tagbench ${SRC})
target_include_directories(tagbench PUBLIC "libs/eigen/" ${APRILTAGS_INCLUDE_DIRS})
target_link_libraries(tagbench ${OpenCV_LIBS} nlohmann_json::nlohmann_json apriltags)

target_compile_features(tagbench PUBLIC ${CPP_STANDARD})
target_compile_definitions(tagbench PRIVATE _CRT_SECURE_NO_WARNINGS _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING NOMINMAX _USE_MATH_DEFINES)