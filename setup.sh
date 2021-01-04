# Generate build
mkdir -p build && cd build && cmake -D CMAKE_PREFIX_PATH=../../refs/opencv/build/ ..; cd ..

# Build the app
cd build && cmake --build .; cd ..

# Run the app
./build/Debug/tagbench.exe data/arcore-29-11/poses_frames.jsonl

# Or do both
cd build && cmake --build .; cd ..; \
./build/Debug/tagbench.exe data/arcore-tags-29-21/poses_frames.jsonl data/arcore-tags-29-21/frames/frame0391.png data/arcore-tags-29-21/frames/frame0399.png data/arcore-tags-29-21/frames/frame0477.png

### Useful things, here for reference
#
#   Create jsonl with just the camera matrices and camera parameters
#
#       jq -c 'select(.arcore != null or .frames != null)' ../data-orig/arcore-29-11/data.jsonl > data/arcore_poses_frames.jsonl
#
#   Generate some frames from .avi
#
#       ffmpeg -i data.avi frames/frame%04d.png -hide_banner
#
#   Run apriltags tagtest on frames
#
#       ../../refs/apriltags-cpp-win/build/Release/tagtest.exe -f Tag16h5 my/data/out*
#
#   Build and run input data preprocessor to prepare data for tagbench
#
#       cd build && cmake --build . --target input_data_preprocessor; cd ..; build/Debug/input_data_preprocessor.exe data/arcore-31-single-tag/ > tagbench_input.jsonl
#
#   Count frames in .avi
#
#       ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 data/arcore-29-11/data.avi
