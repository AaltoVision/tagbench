import argparse
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', help='Input file with position data (default: stdin)')
    parser.add_argument('-s', '--noise_scale', dest='noise_scale', help='Scale of noise to add (in meters)', type=float, required=True)
    parser.add_argument('-p', '--plot', dest='plot', help='Show plot of original position data and noisy data', action='store_true')
    args = parser.parse_args()
    xs = []
    ys = []
    zs = []
    original_data = []
    f = sys.stdin if args.input is None else open(args.input, 'r')
    for line in f:
        j = json.loads(line)
        original_data.append(j)
        if "cameraExtrinsics" in j:
            e = j["cameraExtrinsics"]
            xs.append(e["position"]["x"])
            ys.append(e["position"]["y"])
            zs.append(e["position"]["z"])
    if f is not sys.stdin:
        f.close()

    # add noise, centered on 0, scale 's' (meters)
    noisy_xs = np.array(xs) + np.random.normal(0, args.noise_scale, len(xs))
    noisy_ys = np.array(ys) + np.random.normal(0, args.noise_scale, len(ys))
    noisy_zs = np.array(zs) + np.random.normal(0, args.noise_scale, len(zs))

    # 3d plot of original and new
    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xs=xs, ys=ys, zs=zs)
        ax.plot(xs=noisy_xs, ys=noisy_ys, zs=noisy_zs)
        plt.show()

    # write augmented positions
    curr = 0
    augmented_data = []
    for datum in original_data:
        if "cameraExtrinsics" in datum:
            e = datum["cameraExtrinsics"]
            e["position"]["x"] = noisy_xs[curr]
            e["position"]["y"] = noisy_ys[curr]
            e["position"]["z"] = noisy_zs[curr]
            curr += 1
        augmented_data.append(datum)

    # output augmented data
    for datum in augmented_data:
        print(json.dumps(datum, separators=(',', ':')))
