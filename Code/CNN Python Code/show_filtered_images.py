#!/usr/bin/env python
import argparse
import gzip
import os
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal

import fileio


def process_image(pickle_file, image, output_file):
    with gzip.open(pickle_file, mode='rb') as pickled_fd:
        un_pickle = pickle.Unpickler(pickled_fd)
        weights = un_pickle.load()

    layer_1 = weights[0]
    filter_energy = [np.sum(np.square(f[0])) for f in layer_1]

    largest_to_smallest_idx = (
        sorted(range(len(filter_energy)),
               key=lambda k: filter_energy[k],
               reverse=True))

    fig, axes = plt.subplots(2, 3)
    axes = np.ndarray.flatten(axes)
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].set_axis_off()

    for i in range(5):
        filt = layer_1[largest_to_smallest_idx[i]][0]

        filtered = signal.convolve2d(image, filt, mode='valid')
        axes[i+1].imshow(np.absolute(filtered), cmap='gray')
        axes[i+1].set_title('Filter %d' % (i+1))
        axes[i+1].set_axis_off()

    doc = PdfPages(output_file)
    fig.savefig(doc, format="pdf")
    doc.close()


def real_main(options):
    candidates = [f for f in os.listdir(options.directory)
                  if os.path.isdir(os.path.join(options.directory, f))]
    files = {"_".join(["filter", c]) + ".pdf": os.path.join(
        options.directory, c, options.pickled)
             for c in candidates if os.path.exists(
                os.path.join(options.directory, c, options.pickled))}

    faces = fileio.FaceReader(
        "../data/training.csv", "../data/training.pkl.gz", 10)
    data = faces.load_file()
    image = data['X'][options.image_num][0]

    for out_file, in_file in sorted(files.items()):
        process_image(in_file, image, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", dest='directory', required=True, help="Directory to search.")
    parser.add_argument(
        "-p", dest='pickled', default="state_01000.pkl.gz",
        help="Pickle File to search for.")
    parser.add_argument(
        "--image", dest='image_num', default=1,
        help="Image number to display.")

    options = parser.parse_args()
    real_main(options)


if __name__ == "__main__":
    main()
