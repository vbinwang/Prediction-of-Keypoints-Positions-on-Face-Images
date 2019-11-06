#!/usr/bin/env python
import argparse
import os
import re
import time

import pandas as pd
import numpy as np


COORD_COLUMNS = [
    "left_eye_center_x",            "left_eye_center_y",
    "right_eye_center_x",           "right_eye_center_y",
    "left_eye_inner_corner_x",      "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",      "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",     "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",     "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",     "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",     "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",    "right_eyebrow_outer_end_y",
    "nose_tip_x",                   "nose_tip_y",
    "mouth_left_corner_x",          "mouth_left_corner_y",
    "mouth_right_corner_x",         "mouth_right_corner_y",
    "mouth_center_top_lip_x",       "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",    "mouth_center_bottom_lip_y"]


def missing_cols_names():
    ordered_cols = [re.sub(r'_[xy]$', '', f) for f in COORD_COLUMNS]
    selected_cols = ([c for (i, c) in enumerate(ordered_cols) if i
                     in range(0, len(ordered_cols), 2)])
    assert set(selected_cols) == set(ordered_cols)
    return ['missing_' + c for c in selected_cols]


def load_candidates(in_dir, in_filename):
    candidate_sources = (
        [d for d in os.listdir(in_dir)
            if os.path.isdir(os.path.join(in_dir, d))])

    sources = (
        [d for d in candidate_sources if
            os.path.exists(os.path.join(in_dir, d, in_filename))])

    def process_file(source):
        y_hat_path = os.path.join(in_dir, source, in_filename)
        return pd.read_csv(y_hat_path, engine='c', index_col=0)

    start_time = time.time()
    print "Reading files"
    frames = [process_file(s) for s in sources]
    print [df.shape for df in frames]
    print "  took {:.3f}s".format(time.time() - start_time)

    return sources, frames


def process_loss_main(options):
    in_dir = options.in_dir
    in_filename = 'loss.csv'
    out_filepath = os.path.join(options.in_dir, "combined_loss.csv")

    sources, frames = load_candidates(in_dir, in_filename)

    original_loss_cols = ['train_loss', 'train_rmse']
    for i, source in enumerate(sorted(sources)):
        frames[i].rename(
            columns={c: '_'.join([source, c]) for c in original_loss_cols},
            inplace=True)

    loss_cols = (['_'.join([s, c]) for s in sorted(sources)
                  for c in original_loss_cols])
    all_column_names = np.concatenate(
        (COORD_COLUMNS, missing_cols_names(), loss_cols))

    start_time = time.time()
    print "Concatenating Dataframes/Writing Output"
    result = pd.concat(frames, axis=1)
    result = result[all_column_names]
    result.to_csv(out_filepath)
    print "  took {:.3f}s".format(time.time() - start_time)


def process_pred(in_dir, in_filename, out_filepath):
    _, frames = load_candidates(in_dir, in_filename)

    start_time = time.time()
    print "Concatenating Dataframes"
    result = pd.concat(frames, axis=1)
    all_column_names = np.concatenate((COORD_COLUMNS, missing_cols_names()))
    result.sort_index(inplace=True)
    result = result[all_column_names]
    print "  took {:.3f}s".format(time.time() - start_time)

    start_time = time.time()
    print "Writing output to %s" % out_filepath
    result.to_csv(out_filepath)
    print "  took {:.3f}s".format(time.time() - start_time)


def process_pred_main(options):
    datasources = {
        "valid": {
            "pred": "last_layer_val.csv",
            "actual": "y_validate.csv"
        },
        "train": {
            "pred": "last_layer_train.csv",
            "actual": "y_train.csv"
        }
    }

    for source_name, source_dict in datasources.items():
        for type_name, filename in source_dict.items():
            out_file = (
                "combined_" + "_".join([source_name, type_name]) + '.csv')
            process_pred(options.in_dir,
                         filename, os.path.join(options.in_dir, out_file))


def process_kaggle_main(options):
    process_pred(options.in_dir, "kaggle.csv",
                 os.path.join(options.in_dir, "kaggle.csv"))


def real_main(options):
    if options.which == "loss":
        process_loss_main(options)
    elif options.which == "pred":
        process_pred_main(options)
    elif options.which == "kaggle":
        process_kaggle_main(options)
    elif options.which == "all":
        process_pred_main(options)
        process_loss_main(options)
        pass
    else:
        raise IndexError("cannot find task for %s" % options.which)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--dir', dest='in_dir', help="Input Directory", required=True)
    parser.add_argument(
        'which', nargs="?", choices=["loss", "pred", "all", "kaggle"],
        default="all", help="What files to process")
    options = parser.parse_args()

    real_main(options)

if __name__ == "__main__":
    # missing_cols_names()
    main()
