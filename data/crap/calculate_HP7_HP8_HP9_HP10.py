import ast
from scipy.io import loadmat
import soundfile as sf
import copy
import librosa
import matplotlib.pyplot as plt
from smoothing_vector import  smooth_complex
import os
import time
from scipy.signal import windows as windows
import numpy as np

path_to_HRIR = "../HRIR_KEMAR_AC0010_1.mat"
path_to_Driver = "../DriverResponse_KEMAR_AC0010_1.mat"
path_to_brir = "../../example/brirs/BRIR_Front_Audiolab/125/brir0.wav"
path_to_HP = "../../example/hpirs/inverse_Dl_2.wav"
path_to_HP_minimum_phase = "../../example/hpirs/inverse_Dl_2_minimum_phase.wav"
N_F = 16384
freq_bin = 48000/N_F
freq_bins = np.arange(N_F) * freq_bin
window_direct = np.hstack([np.ones(300), windows.triang(61)[31:], np.zeros(16384 - 300 - 30)])
window_reverb = np.hstack([np.zeros(300), windows.triang(61)[:30], np.ones(16384 - 300 - 30)])


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


# load HP
HP, _ = sf.read(path_to_HP)
HP_minimum, _ = sf.read(path_to_HP_minimum_phase)
HP_left = np.fft.fft(HP[:, 0], N_F)
HP_right = np.fft.fft(HP[:, 1], N_F)
HP_left_abs = np.abs(np.fft.fft(HP[:, 0], N_F))
HP_right_abs = np.abs(np.fft.fft(HP[:, 1], N_F))

filter = np.vstack(( np.real(np.fft.ifft(np.abs(np.fft.fft(HP[:, 0], N_F)))), np.real(np.fft.ifft(np.abs(np.fft.fft(HP[:, 1], N_F)))))).T
sf.write("HP_mean.wav", filter, 48000, subtype="DOUBLE")

plt.plot(freq_bins[:N_F//2], np.angle(np.fft.fft(HP[:, 0], N_F)[:N_F//2]), label="HP")
plt.plot(freq_bins[:N_F//2], np.angle(np.fft.fft(HP_minimum[:, 0], N_F)[:N_F//2]), label="HP minimum")
#plt.plot(np.fft.ifft(HP_left_abs), label="HP abs")
plt.xscale("log")
plt.grid()
plt.legend()
plt.show()
exit()

# load equalizaztion filter
with open("mean_filter_279.txt", "r") as file:
    text = file.read()
    equalization_array = openMHAarray_2_numpyarray(text)

left_equalization_array = np.fft.fft(equalization_array[0], N_F)
left_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 8], orig_sr=44100, target_sr=48000), N_F)
left_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 0], orig_sr=44100, target_sr=48000), N_F)

right_equalization_array = np.fft.fft(equalization_array[1], N_F)
right_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 9], orig_sr=44100, target_sr=48000), N_F)
right_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 1], orig_sr=44100, target_sr=48000), N_F)

brir, srate = sf.read(path_to_brir)
left_direct = np.fft.fft(brir[:, 0] * window_direct, N_F)
right_direct = np.fft.fft(brir[:, 1] * window_direct, N_F)

left_smoothed_direct_HP9 = smooth_complex(N_F, 48000, 1 / 6, np.abs(left_direct * HP_left))
left_smoothed_direct_HP10 = smooth_complex(N_F, 48000, 1 / 6, np.abs(left_direct) * HP_left_abs)
left_smoothed_transparency = smooth_complex(N_F, 48000, 1 / 6, np.abs(left_equalization_array * left_concha + left_closed_ear))
right_smoothed_direct_HP9 = smooth_complex(N_F, 48000, 1 / 6, np.abs(right_direct * HP_right))
right_smoothed_direct_HP10 = smooth_complex(N_F, 48000, 1 / 6, np.abs(right_direct) * HP_right_abs)
right_smoothed_transparency = smooth_complex(N_F, 48000, 1 / 6, np.abs(right_equalization_array * right_concha + right_closed_ear))

weightings_left_HP9 = left_smoothed_transparency / left_smoothed_direct_HP9
weightings_altered_left_HP9 = np.min(np.vstack([weightings_left_HP9, np.ones(N_F) * 16]), axis=0)
weightings_right_HP9 = right_smoothed_transparency / right_smoothed_direct_HP9
weightings_altered_right_HP9 = np.min(np.vstack([weightings_right_HP9, np.ones(N_F) * 16]), axis=0)

weightings_left_HP10 = left_smoothed_transparency / left_smoothed_direct_HP10
weightings_altered_left_HP10 = np.min(np.vstack([weightings_left_HP10, np.ones(N_F) * 16]), axis=0)
weightings_right_HP10 = right_smoothed_transparency / right_smoothed_direct_HP10
weightings_altered_right_HP10 = np.min(np.vstack([weightings_right_HP10, np.ones(N_F) * 16]), axis=0)
print("weightings calculated!\n")

dict_grenzen = {125: 250, 150: 290, 175: 320, 200: 350, 225:380, 250:420, 275:460, 300:490, 325: 520}

room = "Audiolab"
print("room")
for speaker in ["Front", "Side"]:
    print(" ->",speaker)
    for position in range(125, 326, 25):
        print("    ->",position)
        brirs = loadmat(f'C:/Users/student/master_doll/pyBinSim-lightweight/example/brirs/BRIR_{speaker}_{room}/brirs{position}.mat')

        matrixes = brirs.get('brirMat')
        # format = channels x samples
        for angle in range(90):
            # brir_0_links = matrixes[angle,0].T
            # brir_0_rechts = matrixes[angle,1].T
            brir = matrixes[angle]
            left_brir = np.fft.fft(brir[0], N_F)
            right_brir = np.fft.fft(brir[1], N_F)

            brir_left_HP7 = np.real(np.fft.ifft(left_brir * HP_left, n=N_F))
            brir_right_HP7 = np.real(np.fft.ifft(right_brir * HP_right, n=N_F))
            brir_HP7 = np.vstack([brir_left_HP7, brir_right_HP7])

            brir_left_HP8 = np.real(np.fft.ifft(left_brir * HP_left_abs, n=N_F))
            brir_right_HP8 = np.real(np.fft.ifft(right_brir * HP_right_abs, n=N_F))
            brir_HP8 = np.vstack([brir_left_HP8, brir_right_HP8])

            brir_left_HP9 = np.real(np.fft.ifft(left_brir * weightings_altered_left_HP9, n=N_F))
            brir_right_HP9 = np.real(np.fft.ifft(right_brir * weightings_altered_right_HP9, n=N_F))
            brir_HP9 = np.vstack([brir_left_HP9, brir_right_HP9])

            brir_left_HP10 = np.real(np.fft.ifft(left_brir * weightings_altered_left_HP10, n=N_F))
            brir_right_HP10 = np.real(np.fft.ifft(right_brir * weightings_altered_right_HP10, n=N_F))
            brir_HP10 = np.vstack([brir_left_HP9, brir_right_HP9])


            # samples x channels
            sf.write(f'C:/Users/student/master_doll/pyBinSim-lightweight/example/brirs/BRIR_{speaker}_{room}_HP7/{position}/brir'+str(angle*4)+'.wav', brir_HP7.T, 48000, subtype="DOUBLE")
            sf.write(f'C:/Users/student/master_doll/pyBinSim-lightweight/example/brirs/BRIR_{speaker}_{room}_HP8/{position}/brir' + str(angle * 4) + '.wav', brir_HP8.T, 48000, subtype="DOUBLE")
            sf.write(f'C:/Users/student/master_doll/pyBinSim-lightweight/example/brirs/BRIR_{speaker}_{room}_HP9/{position}/brir' + str(angle * 4) + '.wav', brir_HP9.T, 48000, subtype="DOUBLE")
            sf.write(f'C:/Users/student/master_doll/pyBinSim-lightweight/example/brirs/BRIR_{speaker}_{room}_HP9/{position}/brir' + str(angle * 4) + '.wav', brir_HP10.T, 48000, subtype="DOUBLE")