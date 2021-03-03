# Project for measuring small-scale visual accuracy for VIO-estimated poses

- 'tagbench' target is the main application
- 'input_data_preprocessor' helper program turns recorded video & VIO data into input for the tagbench program
    - Currently outputs only video frames and VIO poses with matching timestamps for testing, but might be different later
- Currently using Windows port of apriltags library, will change to Linux supporting one later that also has proper license information

## Build

Tested on Linux (Ubuntu) and Windows

Instructions are in shell syntax. If you are using Windows it is recommended to use a shell such as git bash that comes with [Git for Windows](https://gitforwindows.org/)

The project depends on OpenCV, you need -DCMAKE_PREFIX_PATH="..." if you are using the [pre-built binaries](https://opencv.org/releases/). If you have it installed system-wide, you can skip it.

Note: the cmake prefix path is relative to the repository root, not the build folder.

    mkdir -p build
    cd build
    cmake .. -DCMAKE_PREFIX_PATH="path/to/opencv/build/"
    cd ..
    cmake --build build/ --config RelWithDebInfo

Note: the resulting \<build-dir\> (see <i>Usage</i>) is either <i>build</i> or <i>build/\<config\></i>, depending on your cmake generator (for example, make = build and visual studio = <i>build/\<config\></i>

## Usage

Running tagbench with the small test data set in the repo

    build/RelWithDebInfo/tagbench.exe -i data/mini-test.jsonl

## Preparing your own data

[ffmpeg](https://ffmpeg.org/) needs to be installed and in PATH

    # Change this to your data directory
    MY_INPUT_DATA_DIR=recorded_data/arcore-20201231144534

    # Extract frames from your video
    mkdir -p $MY_INPUT_DATA_DIR/frames
    ffmpeg -i $MY_INPUT_DATA_DIR/data.avi -start_number 0 $MY_INPUT_DATA_DIR/frames/%d.png -hide_banner

    # Preprocess your recorded video/VIO data
    mkdir -p data
    cmake --build build/ --target input_data_preprocessor --config RelWithDebInfo
    build/RelWithDebInfo/input_data_preprocessor.exe $MY_INPUT_DATA_DIR -o data/tagbench_input.jsonl

    # Run the benchmark with your data
    # NOTE: currently very slow for large data
    cmake --build build/ --target tagbench --config RelWithDebInfo
    build/RelWithDebInfo/tagbench.exe -i data/tagbench_input.jsonl