#!/usr/bin/env python
import argparse
import code
import os
# import re
import time

import pandas as pd


def real_main(options):
    start_time = time.time()
    print "Loading Data"
    ids = pd.read_csv(options.id_file, index_col=0)
    predict = pd.read_csv(os.path.join(
        options.in_dir, "kaggle.csv"), index_col=0)

    submission = pd.DataFrame(index=ids.index, columns=['Location'])
    print "  took {:.3f}s".format(time.time() - start_time)

    start_time = time.time()
    print "Creating Submission DataFrame"
    # num_missing = 0
    for idx in ids.index:
        image_id = ids.loc[idx, 'ImageId']
        feature_name = ids.loc[idx, 'FeatureName']
        # mis_feature_name = "missing_" + re.sub(r'_[xy]$', '', feature_name)

        # prob_missing = predict.loc[image_id, mis_feature_name]
        # if prob_missing > 0.5:
        #     num_missing += 1
        #     continue
        location = predict.loc[image_id, feature_name]
        submission.set_value(idx, 'Location', location)

    # print "Predicted %3.2f%% missing" % (
    #     float(num_missing)/float(len(submission)) * 100.)

    submission.sort_index(inplace=True)
    print "  took {:.3f}s".format(time.time() - start_time)

    start_time = time.time()
    print "Writing Submission to Disk"
    submission.to_csv("submission.csv")
    print "  took {:.3f}s".format(time.time() - start_time)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', dest='in_dir', default="run_e1000_swapaxes",
        help='input directory')
    parser.add_argument(
        '--id_file', dest="id_file",
        default=os.path.abspath("../data/IdLookupTable.csv"))
    options = parser.parse_args()

    real_main(options)

if __name__ == "__main__":
    main()
