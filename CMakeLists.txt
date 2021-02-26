cmake_minimum_required(VERSION 3.5.0)
project(tagbench)

# Compile options etc.
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV)

# Simplified manual build for the apriltags library due to problems
# in the original library's CMakeLists.txt
set(APRILTAG_SRC
    "libs/apriltag/apriltag.c"
    "libs/apriltag/apriltag_pose.c"
    "libs/apriltag/apriltag_quad_thresh.c"
    "libs/apriltag/tag36h11.c")
file(GLOB APRILTAG_COMMON_SRC
    "libs/apriltag/common/*.c"
    "libs/apriltag/common/*.cpp")
add_library(apriltags ${APRILTAG_SRC} ${APRILTAG_COMMON_SRC})
target_include_directories(apriltags PUBLIC "libs/apriltag/")
if (NOT MSVC)
    find_package(Threads REQUIRED)
    target_link_libraries(apriltags PUBLIC Threads::Threads m)
endif()

# dependency: nlohmann_json
option(JSON_BuildTests OFF)
add_subdirectory(libs/json)

# dependency: cxxopts
option(CXXOPTS_BUILD_EXAMPLES OFF)
option(CXXOPTS_BUILD_TESTS OFF)
option(CXXOPTS_ENABLE_INSTALL OFF)
add_subdirectory(libs/cxxopts)

# Input data preprocessor
add_executable(input_data_preprocessor
    src/input_data_preprocessor.cpp
    src/detect_markers.cpp
    src/detect_markers.h
    )
target_link_libraries(input_data_preprocessor PRIVATE
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    apriltags
    cxxopts
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>)
target_compile_definitions(input_data_preprocessor PRIVATE _CRT_SECURE_NO_WARNINGS)

# tagbench (the main app)
add_executable(tagbench
    src/tagbench.cpp
    src/pose_optimizer.cpp
    src/pose_optimizer.h
    src/detect_markers.cpp
    src/detect_markers.h
    src/cv_helpers.cpp
    src/cv_helpers.h
    )
target_include_directories(tagbench PUBLIC "libs/eigen/" ${APRILTAGS_INCLUDE_DIRS})
target_link_libraries(tagbench
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    apriltags
    cxxopts)
target_compile_definitions(tagbench PRIVATE _CRT_SECURE_NO_WARNINGS _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING NOMINMAX _USE_MATH_DEFINES)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16.0")
    target_precompile_headers(tagbench PRIVATE src/pch.h)
endif()

# synthetic_test (simple synthetic test app for validating the optimizer)
add_executable(synthetic_test
    src/synthetic_test.cpp
    src/pose_optimizer.cpp
    src/pose_optimizer.h
    src/cv_helpers.cpp
    src/cv_helpers.h
    )
target_include_directories(synthetic_test PUBLIC "libs/eigen/" ${APRILTAGS_INCLUDE_DIRS})
target_link_libraries(synthetic_test ${OpenCV_LIBS})
target_compile_definitions(synthetic_test PRIVATE _CRT_SECURE_NO_WARNINGS _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING NOMINMAX _USE_MATH_DEFINES)
