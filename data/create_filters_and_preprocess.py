import numpy as np
import scipy.io
import scipy.signal.windows as windows
import matplotlib.pyplot as plt
from scipy.io import loadmat
import soundfile as sf
import os
import copy
import librosa
import ast

N_f = 16384
path_to_HRIR = "HRIR_KEMAR_AC0010_1.mat"
path_to_HP_minimum_IDMT = "inverse_Dl_2_minimum_phase.wav"
path_to_HP_minimum_Oldenburg = "inverse_Dl_2_minimum_phase.wav"
path_to_HP_Oldenburg="inverse_Dl_2_minimum_phase.wav"
path_to_Driver = "DriverResponse_KEMAR_AC0010_1.mat"

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

HP_minimum_IDMT, _ = sf.read(path_to_HP_minimum_IDMT)
HP_minimum_Oldenburg, _ = sf.read(path_to_HP_minimum_Oldenburg)
HP_Oldenburg, _ = sf.read(path_to_HP_Oldenburg)

HP_minimum_Oldenburg_new = librosa.resample(y=HP_minimum_Oldenburg.T, orig_sr=44100, target_sr=48000).T
HP_Oldenburg_new = librosa.resample(y=HP_Oldenburg.T, orig_sr=44100, target_sr=48000).T

left_concha = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 8], orig_sr=44100, target_sr=48000)
right_concha = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 9], orig_sr=44100, target_sr=48000)
left_tweeter = librosa.resample(y=loadmat(path_to_Driver).get("M_data")[:, 0, 0], orig_sr=44100, target_sr=48000)

left_closed_ear = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 0], orig_sr=44100, target_sr=48000)
right_closed_ear = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 1], orig_sr=44100, target_sr=48000)
right_tweeter = librosa.resample(y=loadmat(path_to_Driver).get("M_data")[:, 1, 1], orig_sr=44100, target_sr=48000)

filter_frequency_IDMT_minimum = np.fft.fft(HP_minimum_IDMT.T, N_f)
filter_frequency_Oldenburg_minimum = np.fft.fft(HP_minimum_Oldenburg_new.T, N_f)
filter_frequency_Oldenburg = np.fft.fft(HP_Oldenburg_new.T, N_f)

filter_left_hearthrough = (np.fft.fft(left_concha, N_f) * np.fft.fft(equalization_array[0], N_f) + np.fft.fft(left_closed_ear, N_f)) * filter_frequency_IDMT_minimum[0]
filter_right_hearthroug = (np.fft.fft(right_concha, N_f) * np.fft.fft(equalization_array[1], N_f) + np.fft.fft(right_closed_ear, N_f)) * filter_frequency_IDMT_minimum[1]
filter_frequency_hearthrough = np.vstack((filter_left_hearthrough, filter_right_hearthroug))


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
                os.mkdir(f"../example/brirs/BRIR_{speaker}_{room}_HP1/{position}")
                os.mkdir(f"../example/brirs/BRIR_{speaker}_{room}_HP2/{position}")
                os.mkdir(f"../example/brirs/BRIR_{speaker}_{room}_HP3/{position}")

            brirs = loadmat(f'../example/brirs/BRIR_{speaker}_{room}/brirs{position}.mat')

            matrixes = brirs.get('brirMat')
            # format = channels x samples
            for angle in range(90):
                # brir_0_links = matrixes[angle,0].T
                # brir_0_rechts = matrixes[angle,1].T
                brir = matrixes[angle]
                brir_fft = np.fft.fft(brir, N_f)
                energy_brir = np.sum(np.square(brir))

                brir_hearthrough = np.real(np.fft.ifft(brir_fft * filter_frequency_hearthrough))
                energy_hearthrough = np.sum(np.square(brir_hearthrough))
                brir_hearthrough *= np.sqrt(energy_brir / energy_hearthrough)

                brir_oldenburg = np.real(np.fft.ifft(brir_fft * filter_frequency_Oldenburg))
                energy_oldenburg = np.sum(np.square(brir_oldenburg))
                brir_oldenburg *= np.sqrt(energy_brir / energy_oldenburg)

                brir_oldenburg_minimum = np.real(np.fft.ifft(brir_fft * filter_frequency_Oldenburg_minimum))
                energy_oldenburg_minimum = np.sum(np.square(brir_oldenburg_minimum))
                brir_oldenburg_minimum *= np.sqrt(energy_brir / energy_oldenburg_minimum)

                brir_IDMT_minimum = np.real(np.fft.ifft(brir_fft * filter_frequency_IDMT_minimum))
                energy_IDMT_minimum = np.sum(np.square(brir_IDMT_minimum))
                brir_IDMT_minimum *= np.sqrt(energy_brir / energy_IDMT_minimum)

                #plt.plot(brir_hearthrough[0], label="hearthorugh")
                #plt.plot(brir_oldenburg[0], label="oldenburg")
                #plt.plot(brir_oldenburg_minimum[0], label="oldenburg minimum")
                #plt.plot(brir_IDMT_minimum[0], label="IDMT minimum")
                #plt.legend()
                #plt.show()

                #plt.plot(f_bins[:N_f // 2], 20 * np.log10(brir_new[:N_f // 2]))
                #plt.plot(f_bins[:N_f//2], brir_db)
                #plt.xscale("log")
                #plt.show()

                # samples x channels
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}/{position}/brir' + str(angle * 4) + '.wav', brir.T, 48000, subtype="DOUBLE")
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}_hearthrough/{position}/brir'+str(angle*4)+'.wav', brir_hearthrough.T, 48000, subtype="DOUBLE")
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}_oldenburg/{position}/brir' + str(angle * 4) + '.wav', brir_oldenburg.T, 48000, subtype="DOUBLE")
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}_oldenburg_minimum/{position}/brir' + str(angle * 4) + '.wav', brir_oldenburg_minimum.T, 48000, subtype="DOUBLE")
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}_idmt_minimum/{position}/brir' + str(angle * 4) + '.wav', brir_IDMT_minimum.T, 48000, subtype="DOUBLE")