import numpy as np
import ast
import librosa
from scipy.io import loadmat
from smoothing_vector import smooth_complex
import pickle
from datetime import datetime

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

with open("mean_filter_279.txt", "r") as file:
    text = file.read()
    equalization_array = openMHAarray_2_numpyarray(text)

N_F = 16384

left_equalization = np.fft.fft(equalization_array[0], N_F)
right_equalization = np.fft.fft(equalization_array[1], N_F)

dict_weightings = {}

for device in ["AC0010", "AC0011"]:
    print(device)
    dict_weightings[device] = {}
    for i in range(1, 4):
        print("-->", i, datetime.now().strftime('%H:%M:%S'))
        dict_weightings[device][i] = {}
        for hrir_idx in range(48):
            path_to_HRIR = f"HRIR_KEMAR_{device}_{i}.mat"

            left_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, hrir_idx, 8], orig_sr=44100, target_sr=48000), N_F)
            left_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, hrir_idx, 0], orig_sr=44100, target_sr=48000), N_F)

            right_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, hrir_idx, 9], orig_sr=44100, target_sr=48000), N_F)
            right_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, hrir_idx, 1], orig_sr=44100, target_sr=48000), N_F)


            spectrum_transparency_left = np.abs((left_concha * left_equalization) + left_closed_ear)
            smoothed_spectrum_transparency_left = smooth_complex(N_F, 48000, 1 / 6, spectrum_transparency_left)

            spectrum_transparency_right = np.abs((right_concha * right_equalization) + right_closed_ear)
            smoothed_spectrum_transparency_right = smooth_complex(N_F, 48000, 1 / 6, spectrum_transparency_right)

            dict_weightings[device][i][hrir_idx] = np.vstack([smoothed_spectrum_transparency_right, smoothed_spectrum_transparency_right])

new_dict = {}
for hrir_idx in range(48):
    hrir_mean = np.zeros((2, N_F))
    for device in ["AC0010", "AC0011"]:
        for i in range(1, 4):
            hrir_mean += dict_weightings[device][i][hrir_idx]
    hrir_mean /= 6
    new_dict[hrir_idx] = hrir_mean

with open("mean_smoothed_spectra_hearpiece.pkl", "wb") as file:
    pickle.dump(new_dict, file, protocol=0)