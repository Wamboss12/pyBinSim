import pickle
import soundfile as sf
from scipy.io import loadmat
import math
import copy
import numpy as np
import librosa
import scipy.io
import ast

N_f = 16384
s_rate = 48000

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

with open("2022_04_29--18_21_03--mean_filter_approximation_48000.txt", "r") as file:
    text = file.read()
    equalization_array = openMHAarray_2_numpyarray(text)

path_to_HP_minimum_IDMT = "inverse_Dl_2_minimum_phase.wav"
path_to_HRIR = "HRIR_KEMAR_AC0010_1.mat"
HP_minimum_IDMT, _ = sf.read(path_to_HP_minimum_IDMT)
filter_frequency_IDMT_minimum = np.fft.fft(HP_minimum_IDMT.T, N_f)

room = "Audiolab"
print("room")
for speaker in ["Front", "Side"]:
    print(" ->",speaker)
    for position in range(125, 326, 25):
        print("    ->",position)
        brirs = loadmat(f'../example/brirs/BRIR_{speaker}_{room}/brirs{position}.mat')

        matrixes = brirs.get('brirMat')
        # format = channels x samples
        for angle in range(90):
            # brir_0_links = matrixes[angle,0].T
            # brir_0_rechts = matrixes[angle,1].T
            brir = matrixes[angle]
            brir_fft = np.fft.fft(brir, N_f)
            energy_brir = np.sum(np.square(brir))

            if speaker == "Front":
                new_angle = angle * 4
            else:
                adaption = 0
                if position < 175:
                    adaption = 180 - math.degrees(math.atan(125/abs(175 - position)))
                elif position == 175:
                    adaption = 90
                else:
                    adaption = math.degrees(math.atan(125/abs(175 - position)))
                new_angle = angle * 4 + round(adaption, 2)
                if new_angle > 360:
                    new_angle = new_angle - 360
            new_angle = 360 - new_angle

            index_for_hrirs = int(round(new_angle / 7.5))
            if index_for_hrirs == 48:
                index_for_hrirs = 0

            left_concha = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, index_for_hrirs, 8], orig_sr=44100, target_sr=48000)
            right_concha = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, index_for_hrirs, 9], orig_sr=44100, target_sr=48000)

            left_closed_ear = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, index_for_hrirs, 0], orig_sr=44100, target_sr=48000)
            right_closed_ear = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, index_for_hrirs, 1], orig_sr=44100, target_sr=48000)

            filter_left_hearthrough = (np.fft.fft(left_concha, N_f) * np.fft.fft(equalization_array[0], N_f) + np.fft.fft(left_closed_ear, N_f)) * filter_frequency_IDMT_minimum[0]
            filter_right_hearthroug = (np.fft.fft(right_concha, N_f) * np.fft.fft(equalization_array[1], N_f) + np.fft.fft(right_closed_ear, N_f)) * filter_frequency_IDMT_minimum[1]
            filter_frequency_hearthrough = np.vstack((filter_left_hearthrough, filter_right_hearthroug))

            brir_hearthrough = np.real(np.fft.ifft(brir_fft * filter_frequency_hearthrough))
            energy_hearthrough = np.sum(np.square(brir_hearthrough))
            brir_hearthrough *= np.sqrt(energy_brir / energy_hearthrough)

            # samples x channels
            sf.write(f'../example/brirs/BRIR_{speaker}_{room}_hearthrough_new/{position}/brir'+str(angle*4)+'.wav', brir_hearthrough.T, 48000, subtype="DOUBLE")