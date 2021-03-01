# Compare benchmark results with increasingly noisy input VIO data
TAGBENCH_PATH="$1"
INPUT_DATA_PATH="$2"

echo tagbench results with original data
$TAGBENCH_PATH -s 0.198 < $INPUT_DATA_PATH
echo

for NOISE_SCALE in 0 0.001 0.005 0.01 0.05
do
    echo tagbench results with position noise with scale $NOISE_SCALE
    python3 scripts/augment_vio_data.py -i $INPUT_DATA_PATH --noise_scale $NOISE_SCALE | $TAGBENCH_PATH -s 0.198
    echo
done