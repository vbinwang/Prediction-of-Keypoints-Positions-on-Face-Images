import argparse
import code
import os

import argcomplete
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model

def get_immediate_subdirectories(a_dir):
	return [name for name in os.listdir(a_dir)
			if os.path.isdir(os.path.join(a_dir, name))]

def main():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-o', '--output_dir', dest='run_data_path',
		metavar="PATH",
		help="subdirectory where AmputatedMLP results are stored")

	argcomplete.autocomplete(parser)
	options = parser.parse_args()

	output_dir = options.run_data_path
	feature_dirs = get_immediate_subdirectories(output_dir)

	rmse_rf = []
	# rmse_lin = []
	col_names = []

	os.chdir(output_dir)
	idx = 0
	for fd in feature_dirs:
		print "feature = " + fd
		os.chdir(fd)
		X_train = pd.read_csv("last_layer_train.csv",
			sep=",", engine="c", index_col=False).values
		X_validate = pd.read_csv("last_layer_val.csv",
			sep=",", engine="c", index_col=False).values
		Y_train = pd.read_csv("y_train.csv",
			sep=",", engine="c", index_col=False, header=True).values
		Y_validate = pd.read_csv("y_validate.csv",
			sep=",", engine="c", index_col=False, header=True).values

		col_names.extend(list(pd.read_csv("y_train.csv",
			sep=",", engine="c", index_col=False, nrows=1).columns.values))

		to_keep = ~(np.isnan(Y_train).any(1))
		X_train = X_train[to_keep]
		Y_train = Y_train[to_keep]
		print "Dropping samples with NaNs: {:3.1f}% dropped".format(float(sum(~to_keep))/float(len(to_keep))*100.)

		num_keypoints = Y_train.shape[1]
		for kp in range(num_keypoints):
			rf = sklearn.ensemble.RandomForestRegressor(n_jobs=-1)
			rf = rf.fit(X_train, Y_train[:, kp])
			pred = rf.predict(X_validate)
			rmse_rf.append(np.sqrt(np.mean(np.power(pred - Y_validate[:, kp], 2))))
			print "RMSE (RF) = " + str(rmse_rf[idx]) + "\n"
			idx += 1

		os.chdir("..")

	data_frame = pd.DataFrame(rmse_rf).transpose()
	data_frame.columns = col_names
	with open("rf_rmse.csv", 'w') as file_desc:
		data_frame.to_csv(file_desc, index=False)

if __name__ == "__main__":
	main()
