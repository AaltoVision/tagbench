
project(tagbench)
cmake_minimum_required(VERSION 3.5.0)

# Compile options etc.
set(CPP_STANDARD cxx_std_17)

# Dependencies
find_package(OpenCV REQUIRED)
# set(APRILTAGS_LIBS "../../../refs/apriltags-cpp-win/build/Debug/apriltags")
set(APRILTAGS_LIBS_DEBUG "../../../refs/apriltags-cpp-win/build/Debug/apriltags")
set(APRILTAGS_LIBS "../../../refs/apriltags-cpp-win/build/Release/apriltags")
set(APRILTAGS_INCLUDE_DIRS ../../refs/apriltags-cpp-win/src)

# set(APRILTAGS_LIBS "../../../refs/apriltags-cpp-win/build/Release/apriltags")
# add_library(apriltags )
# target_link_libraries(apriltags ${APRILTAGS_LIBS})
# target_include_directories(apriltags PUBLIC ${APRILTAGS_INCLUDE_DIRS})


# tagbench (the main app)
set(SRC tagbench.cpp)
add_executable(tagbench ${SRC})
target_include_directories(tagbench PUBLIC "lib/" ${APRILTAGS_INCLUDE_DIRS})
target_link_libraries(tagbench ${OpenCV_LIBS} ${APRILTAGS_LIBS_DEBUG})
# target_link_libraries(tagbench ${OpenCV_LIBS} ${APRILTAGS_LIBS})
target_compile_features(tagbench PUBLIC ${CPP_STANDARD})
target_compile_definitions(tagbench PRIVATE _SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)


# Data preprocessor
add_executable(input_data_preprocessor input_data_preprocessor.cpp)
target_include_directories(input_data_preprocessor PUBLIC "lib/" ${APRILTAGS_INCLUDE_DIRS})
target_link_libraries(input_data_preprocessor ${OpenCV_LIBS} ${APRILTAGS_LIBS})
target_compile_features(input_data_preprocessor PUBLIC ${CPP_STANDARD})

# Preprocessed data
# add_custom_target(preprocessed_data
#     COMMAND $<TARGET_FILE:input_data_preprocessor> data/arcore-31-single-tag ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl
#     BYPRODUCTS ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl
# )
# add_dependencies(preprocessed_data input_data_preprocessor)
# add_custom_command(
# # add_custom_target(
#     # preprocessed_data
#     OUTPUT ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl
#     COMMAND $<TARGET_FILE:input_data_preprocessor> data/arcore-31-single-tag 
#     # BYPRODUCTS ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl
#     DEPENDS input_data_preprocessor data/arcore-31-single-tag
# )
# # add_dependencies(preprocessed_data input_data_preprocessor)

# Benchmark results
# add_custom_target(benchmark_results
#     COMMAND $<TARGET_FILE:tagbench> ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl > ${CMAKE_SOURCE_DIR}/dist/benchmark_results.jsonl
#     BYPRODUCTS ${CMAKE_SOURCE_DIR}/dist/benchmark_results.jsonl
# )
# add_dependencies(benchmark_results preprocessed_data)

# add_custom_command(
# # add_custom_target(
#     # benchmark_results
#     OUTPUT ${CMAKE_SOURCE_DIR}/dist/benchmark_results.jsonl
#     # COMMAND $<TARGET_FILE:tagbench> ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl > ${CMAKE_SOURCE_DIR}/dist/benchmark_results.jsonl
#     COMMAND $<TARGET_FILE:tagbench> ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl ${CMAKE_SOURCE_DIR}/dist/benchmark_results.jsonl
#     # BYPRODUCTS ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl
#     DEPENDS ${CMAKE_SOURCE_DIR}/dist/tagbench_input.jsonl
# )