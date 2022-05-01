import numpy as np
import ast
from scipy.io import loadmat
import soundfile as sf
import copy
import librosa
import matplotlib.pyplot as plt
from smoothing_vector import  smooth_complex
import os
import scipy.signal.windows as windows
import time

path_to_HRIR = "example/HRIR_KEMAR_AC0010_1.mat"
path_to_Driver = "example/DriverResponse_KEMAR_AC0010_1.mat"
path_to_brir = "example/brirs/BRIR_Front_Audiolab_HP2/125/brir0.wav"
N_F = 16384
freq_bin = 48000/N_F
freq_bins = np.arange(N_F) * freq_bin

borders = {"Front": {125: 240, 150: 280, 175: 310, 200: 345,
                     225: 380, 250: 410, 275: 450, 300: 480,
                     325: 520},
           "Side": {125: 250, 150: 240, 175: 230, 200: 240,
                     225: 250, 250: 260, 275: 280, 300: 310,
                     325: 340}}

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


# load equalizaztion filter
with open("mean_filter_279.txt", "r") as file:
    text = file.read()
    equalization_array = openMHAarray_2_numpyarray(text)

left_equalization = np.fft.fft(equalization_array[0], N_F)
left_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 8], orig_sr=44100, target_sr=48000), N_F)
left_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 0], orig_sr=44100, target_sr=48000), N_F)

right_equalization = np.fft.fft(equalization_array[1], N_F)
right_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 9], orig_sr=44100, target_sr=48000), N_F)
right_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 1], orig_sr=44100, target_sr=48000), N_F)


positions = range(125, 326, 25)
speakers = ["Front", "Side"]
rooms = ["Audiolab"]
for room in rooms:
    print(room)
    for speaker in speakers:
        print(" ->",speaker)
        for position in positions:
            print("    ->",position)
            if not os.path.isdir(f"../example/brirs/BRIR_{speaker}_{room}/{position}"):
                os.mkdir(f"../example/brirs/BRIR_{speaker}_{room}/{position}")
            if not os.path.isdir(f"../example/brirs/BRIR_{speaker}_{room}_altered/{position}"):
                os.mkdir(f"../example/brirs/BRIR_{speaker}_{room}_altered/{position}")

            mat_data = loadmat(f'../example/brirs/BRIR_{speaker}_{room}/brirs{position}.mat')
            brirs = mat_data.get('brirMat')

            # format = channels x samples
            for angle in range(90):
                brir = brirs[angle]

                border = borders.get(speaker).get(position)
                max_left = np.argmax(np.abs(brir[0, :border + 30]))
                max_right = np.argmax(np.abs(brir[1, :border + 30]))

                window_direct_left = np.hstack([np.ones(max_left + 60), windows.cosine(61)[31:], np.zeros(N_F - max_left - 90)])
                window_direct_right = np.hstack([np.ones(max_right + 60), windows.cosine(61)[31:], np.zeros(N_F - max_right - 90)])
                #window_reverb_left = np.hstack([np.zeros(max_left + 60), windows.cosine(61)[:30], np.ones(N_F - max_left - 90)])
                #window_reverb_right = np.hstack([np.zeros(max_right + 60), windows.cosine(61)[:30], np.ones(N_F - max_right - 90)])

                spectrum_transparency_left = np.abs(left_concha * left_equalization)
                spectrum_synthesis_left = np.abs(np.fft.fft(brir[0] * window_direct_left, N_F))
                smoothed_spectrum_transparency_left = smooth_complex(N_F, 48000, 1 / 6, spectrum_transparency_left)
                smoothed_spectrum_synthesis_left = smooth_complex(N_F, 48000, 1 / 6, spectrum_synthesis_left)

                spectrum_transparency_right = np.abs(right_concha * right_equalization)
                spectrum_synthesis_right = np.abs(np.fft.fft(brir[1] * window_direct_right, N_F))
                smoothed_spectrum_transparency_right = smooth_complex(N_F, 48000, 1 / 6, spectrum_transparency_right)
                smoothed_spectrum_synthesis_right = smooth_complex(N_F, 48000, 1 / 6, spectrum_synthesis_right)

                weightings_left = smoothed_spectrum_transparency_left / smoothed_spectrum_synthesis_left
                #weightings_altered_left = np.min(np.vstack([weightings_left, np.ones(N_F) * 16]), axis=0)
                altered_spectrum_synthesis_left = (np.fft.fft(brir[0], N_F) * weightings_left) + left_closed_ear

                weightings_right = smoothed_spectrum_transparency_right / smoothed_spectrum_synthesis_right
                #weightings_altered_right = np.min(np.vstack([weightings_right, np.ones(N_F) * 16]), axis=0)
                altered_spectrum_synthesis_right = (np.fft.fft(brir[1], N_F) * weightings_right) + right_closed_ear

                brir_new_left = np.fft.ifft(altered_spectrum_synthesis_left, n=N_F)
                brir_new_right = np.fft.ifft(altered_spectrum_synthesis_right, n=N_F)
                brir_new = np.vstack([brir_new_left, brir_new_right])

                # samples x channels
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}/{position}/brir' + str(angle * 4) + '.wav', brir.T, 48000, subtype="DOUBLE")
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}_altered/{position}/brir'+str(angle*4)+'.wav', brir_new.T, 48000, subtype="DOUBLE")
