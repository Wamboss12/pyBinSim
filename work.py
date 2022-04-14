import soundfile as sf
import numpy as np
import logging
import ast
import scipy.io

path_to_mat = "example/brirs/BRIRs_ListeningLab_2m_LS-Front/brirs125.mat"
path_to_mat_2 = "example/brirs/example_mat.mat"
tracking_data = scipy.io.loadmat(path_to_mat)
tracking_data_2 = scipy.io.loadmat(path_to_mat_2)

print()