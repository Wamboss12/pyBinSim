import copy
import math
import random
from scipy.signal import windows
import matplotlib.pyplot as plt
import numpy as np
import ast
import soundfile as sf
import librosa
import scipy.io

path_to_HRIR = "data/HRIR_KEMAR_AC0010_1.mat"

def openMHAarray_2_numpyarray(arr_as_string):
    """
    This functions takes a String containing a array in openMHA format and converts it into an numpy array
    :param arr_as_string: String containing a array saved in openMHA format
    :type arr_as_string: str
    :return array_from_list: input Array in numpy
    :rtype array_from_list: numpy Array  (numpy.ndarray)
    """

    arr_as_string = arr_as_string.replace(";", ",")
    arr_as_string = arr_as_string.replace("i", "j")
    arr_as_string_splitted = arr_as_string.split(sep=" ")
    new_text = ",".join(arr_as_string_splitted)

    list_from_text = ast.literal_eval(new_text)
    array_from_list = np.array(list_from_text)

    return array_from_list

with open("data/crap/mean_filter_279.txt", "r") as file:
    text = file.read()
    equalization_array = openMHAarray_2_numpyarray(text)

left_concha = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 8], orig_sr=44100, target_sr=48000)
right_concha = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 9], orig_sr=44100, target_sr=48000)

left_closed_ear = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 0], orig_sr=44100, target_sr=48000)
right_closed_ear = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 1], orig_sr=44100, target_sr=48000)

filter_left_hearthrough = np.convolve(left_concha, equalization_array[0])
filter_right_hearthroug = np.convolve(right_concha, equalization_array[1])


signal, _ = sf.read("example/signals/Dialog_48000.wav")
signal_energy = np.sum(np.square(signal))

convolved_signal_left = np.convolve(np.convolve( signal[:, 0], equalization_array[0] ), left_concha)
convolved_signal_right = np.convolve(np.convolve( signal[:, 1], equalization_array[1] ), right_concha)
convolved_signal = np.vstack([convolved_signal_left, convolved_signal_right]).T
convolved_signal /= np.max(np.abs(convolved_signal))

sf.write("Dialog_convolved_withequalization_and_concha.wav", convolved_signal, 48000, subtype="DOUBLE")
