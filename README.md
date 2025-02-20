# tagbench - Measure small-scale visual accuracy of VIO-estimated poses

The tagbench application can be used to measure small-scale visual accuracy of VIO (Visual inertial odometry)-estimated pose data.

You can use the [Android VIO-tester](https://github.com/AaltoML/android-viotester) for recording your data.
Then, use the input_data_preprocessor to prepare it into the input format that tagbench expects (see <i>Usage</i>).
However, you can also use your own data, if you prepare it in the right format (see <i>input format</i>)

Details of the small-scale visual error metric and its computation are available in [docs/apriltag-visual-error-measurement.pdf](docs/apriltag-visual-error-measurement.pdf).

## TODOs

- Make sure input_data_preprocessor outputs marker coordinates in original image size coordinates, regardless of --image_downscale_factor argument
- Make sure tagbench assumes marker coordinates are in original image size coordinates, regardless of --image_downscale_factor argument
- Currently input_data_preprocessor only outputs frames, for which there exist both a camera image, as well as extrinsic data, with matching timestamps. If there is more VIO pose data (extrinsic camera parameters) in between two sequential frames' timestamps, they are not used at all. It might make sense to output the same frame multiple times with each pose. In this case tag corner coordinates ('markers' in the .jsonl output) could also be interpolated between frames, for example linearly, instead of using the same detected marker coordinates for each in-between-frames pose.
- The current example 'mini-test' dataset in the repository might have scaled-down marker coordinates, have to check or just make a new example.
- Check that the 'Usage' section in README is up-to-date.

## Build

The application is tested on Linux (Ubuntu) and Windows.

Build and usage instructions are in shell syntax. If you are using Windows it is recommended to use a shell such as the <i>'git bash'</i> that comes with [Git for Windows](https://gitforwindows.org/).

The project depends on OpenCV, you need -DCMAKE_PREFIX_PATH="..." if you are using the [pre-built binaries](https://opencv.org/releases/). If you have it installed system-wide, you can skip it.

Note: the cmake prefix path is relative to the repository root, not the build folder.

    mkdir -p build
    cd build
    cmake .. -DCMAKE_PREFIX_PATH="path/to/opencv/build/"
    cd ..
    cmake --build build/ --config RelWithDebInfo

Note: the resulting \<build-dir\> (see <i>Usage</i>) is either <i>build</i> or <i>build/\<config\></i>, depending on your cmake generator (for example, make = build and Visual Studio = <i>build/\<config\></i>)

## Usage

Running tagbench with the small test data set in the repo

    ./<build-dir>/tagbench -i data/mini-test.jsonl -s 0.198

To see all available arguments, use the <i>--help</i> switch:

    $ ./<build-dir>/tagbench --help

    Usage:
    ./build/tagbench [OPTION...]

    -i, --input arg               Path to the input file (default: stdin)
    -o, --output arg              Output path (.jsonl file) (default: stdout)
    -s, --tag_side_length arg     Length of the tag's sides in meters in the
                                    input images
    -d, --image_downscale_factor arg
                                    Image downscaling factor (default: 1)
    -p, --plot                    Plot projected points against groundtruth
    -m, --optimizer_max_steps arg
                                    Maximum iteration count for pose optimizer
    -t, --optimizer_stop_threshold arg
                                    Threshold for projection error to stop
                                    optimization (see README)
        --preload_images arg      Preload all input frames to memory for
                                    smoother plot navigation (only when --plot is
                                    enabled) (does not affect results)
        --cache_images arg        Keep loaded input frames in memory for
                                    smoother plot navigation (only when --plot is
                                    enabled) (does not affect results)
    -v, --verbose                 Enable verbose output
    -h, --help                    Show this help message

## Preparing your own data from Android VIO-tester

For extracting frames from your input video, [ffmpeg](https://ffmpeg.org/) needs to be installed and in the environment PATH.

    # Change this to your data directory
    MY_INPUT_DATA_DIR=recorded_data/arcore-20201231144534

    # Extract frames from your video
    mkdir -p $MY_INPUT_DATA_DIR/frames
    ffmpeg -i $MY_INPUT_DATA_DIR/data.avi -start_number 0 $MY_INPUT_DATA_DIR/frames/%d.png -hide_banner

    # Preprocess your recorded video/VIO data
    mkdir -p data
    cmake --build build/ --target input_data_preprocessor --config RelWithDebInfo
    <build-dir>/input_data_preprocessor -i $MY_INPUT_DATA_DIR -o data/tagbench_input.jsonl

    # Run the benchmark with your data
    cmake --build build/ --target tagbench --config RelWithDebInfo
    <build-dir>/tagbench -i data/tagbench_input.jsonl -s 0.198

To see all options, use the <i>--help</i> switch:

    $ ./build/input_data_preprocessor --help

    Usage:
    ./build/input_data_preprocessor [OPTION...]

    -i, --input arg               Path to the input data directory
    -o, --output arg              Output path (.jsonl file) (default: stdout)
    -n, --pose_data_name arg      Name of pose data in the input
    -d, --image_downscale_factor arg
                                    Image downscaling factor (default: 1)
    -h, --help                    Show this help message
