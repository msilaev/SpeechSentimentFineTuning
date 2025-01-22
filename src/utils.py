import h5py
import numpy as np


def load_h5(h5_path):
    with h5py.File(h5_path, "r") as hf:
        X = np.array(hf.get("features"))
        Y = np.array(hf.get("labels"))

        return X, Y
