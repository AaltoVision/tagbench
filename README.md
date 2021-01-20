# Project for measuring small-scale visual accuracy for VIO-estimated poses

- Work in progress
- 'tagbench' target is the main application
- 'input_data_preprocessor' helper program turns recorded video & VIO data into input for the tagbench program
    - Currently outputs only video frames and VIO poses with matching timestamps for testing, but might be different later
- Currently supports just Windows, but will support Linux soon
- Currently using Windows port of apriltags library, will change to Linux supporting one later that also has proper license information

## Usage

Building and running tagbench with the small test data set in the repo

    mkdir -p build && cd build && cmake ..; cd ..
    cmake --build build/ --target tagbench --config Release
    build/Release/tagbench.exe < data/arcore-mini-test.jsonl

Preparing your own data (ffmpeg needs to be installed and in PATH)

    # Change this to your data directory
    MY_INPUT_DATA_DIR=recorded_data/arcore-20201231144534

    # Extract frames from your video
    mkdir -p $MY_INPUT_DATA_DIR/frames
    ffmpeg -i $MY_INPUT_DATA_DIR/data.avi -start_number 0 $MY_INPUT_DATA_DIR/frames/%d.png -hide_banner

    # Preprocess your recorded video/VIO data
    mkdir -p data
    cmake --build build/ --target input_data_preprocessor --config Release
    build/Release/input_data_preprocessor.exe $MY_INPUT_DATA_DIR > data/tagbench_input.jsonl

    # Run the benchmark with your data
    # NOTE: currently very slow for large data
    cmake --build build/ --target tagbench --config Release
    build/Release/tagbench.exe < data/tagbench_input.jsonl