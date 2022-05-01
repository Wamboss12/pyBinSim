import numpy as np
import ast
import librosa
from scipy.io import loadmat
import scipy.signal.windows as windows
from smoothing_vector import smooth_complex
import pickle


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


borders = {"Front": {125: 240, 150: 280, 175: 310, 200: 345,
                     225: 380, 250: 410, 275: 450, 300: 480,
                     325: 520},
           "Side": {125: 250, 150: 240, 175: 230, 200: 240,
                     225: 250, 250: 260, 275: 280, 300: 310,
                     325: 340}}

with open("mean_filter_279.txt", "r") as file:
    text = file.read()
    equalization_array = openMHAarray_2_numpyarray(text)

path_to_HRIR = "../HRIR_KEMAR_AC0010_2.mat"
N_F = 16384
freq_bin = 48000/N_F
freq_bins = np.arange(N_F) * freq_bin

paare = [[0, 0, "Front", 125], [15, 8, "Front", 125], [30, 16, "Front", 125], [45, 24, "Front", 125], [60, 32, "Front", 125], [75, 40, "Front", 125],
         [75, 4, "Side", 175], [0, 12, "Side", 175], [15, 20, "Side", 175], [30, 28, "Side", 175], [45, 36, "Side", 175], [60, 44, "Side", 175]]


left_equalization = np.fft.fft(equalization_array[0], N_F)
right_equalization = np.fft.fft(equalization_array[1], N_F)

mat_data = loadmat(f'../../example/brirs/BRIR_Front_Audiolab/brirs125.mat')
brirs = mat_data.get('brirMat')

dict_weightings = {}

for device in ["AC0010", "AC0011"]:
    print(device)
    dict_weightings[device] = {}
    for i in range(1, 4):
        print("-->", i)
        dict_weightings[device][i] = {}
        for [brir_idx, hrir_idx, speaker, position] in paare:
            path_to_HRIR = f"HRIR_KEMAR_{device}_{i}.mat"
            mat_data = loadmat(f'../example/brirs/BRIR_{speaker}_Audiolab/brirs{position}.mat')
            brirs = mat_data.get('brirMat')

            left_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, hrir_idx, 8], orig_sr=44100, target_sr=48000), N_F)
            left_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, hrir_idx, 0], orig_sr=44100, target_sr=48000), N_F)

            right_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, hrir_idx, 9], orig_sr=44100, target_sr=48000), N_F)
            right_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, hrir_idx, 1], orig_sr=44100, target_sr=48000), N_F)

            brir = brirs[brir_idx]
            border = borders.get("Front").get(125)
            max_left = np.argmax(np.abs(brir[0, :border + 30]))
            max_right = np.argmax(np.abs(brir[1, :border + 30]))
            window_direct_left = np.hstack([np.ones(max_left + 60), windows.cosine(61)[31:], np.zeros(N_F - max_left - 90)])
            window_direct_right = np.hstack([np.ones(max_right + 60), windows.cosine(61)[31:], np.zeros(N_F - max_right - 90)])

            spectrum_transparency_left = np.abs((left_concha * left_equalization) + left_closed_ear)
            spectrum_synthesis_left = np.abs(np.fft.fft(brir[0] * window_direct_left, N_F))
            smoothed_spectrum_transparency_left = smooth_complex(N_F, 48000, 1 / 6, spectrum_transparency_left)
            smoothed_spectrum_synthesis_left = smooth_complex(N_F, 48000, 1 / 6, spectrum_synthesis_left)

            spectrum_transparency_right = np.abs((right_concha * right_equalization) + right_closed_ear)
            spectrum_synthesis_right = np.abs(np.fft.fft(brir[1] * window_direct_right, N_F))
            smoothed_spectrum_transparency_right = smooth_complex(N_F, 48000, 1 / 6, spectrum_transparency_right)
            smoothed_spectrum_synthesis_right = smooth_complex(N_F, 48000, 1 / 6, spectrum_synthesis_right)

            weightings_left = smoothed_spectrum_transparency_left / smoothed_spectrum_synthesis_left
            weightings_right = smoothed_spectrum_transparency_right / smoothed_spectrum_synthesis_right
            dict_weightings[device][i][int(hrir_idx*7.5)] = np.vstack([weightings_left, weightings_right])

with open("weightings_brir_with_leaking_direct.pkl", "wb") as file:
    pickle.dump(dict_weightings, file, protocol=0)
