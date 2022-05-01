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


path_to_HRIR = "../HRIR_KEMAR_AC0010_1.mat"

fs = 44100
N_f = 16384
length = 16384
roll_length = 50
window = np.concatenate( [windows.cosine(21)[:10], np.ones(length - 110), windows.cosine(101)[-50:], np.zeros([roll_length])] )
f_bin = fs / N_f
f_bins = np.arange(N_f) * f_bin

left_concha = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 8], orig_sr=44100, target_sr=48000)
right_concha = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 9], orig_sr=44100, target_sr=48000)

left_closed_ear = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 0], orig_sr=44100, target_sr=48000)
right_closed_ear = librosa.resample(y=scipy.io.loadmat(path_to_HRIR).get("M_data")[:, 0, 1], orig_sr=44100, target_sr=48000)

with open("mean_filter_279.txt", "r") as file:
    text = file.read()
    equalization_array = openMHAarray_2_numpyarray(text)

with open("../denk_279_0.0_48000.txt", "r") as file:
    text = file.read()
    equalization_array_front = openMHAarray_2_numpyarray(text)

with open("weightings_front.txt", "r") as file:
    text = file.read()
    weightings_array = openMHAarray_2_numpyarray(text)



filter_left_frequency_3 = np.fft.fft(left_concha, N_f) * np.fft.fft(equalization_array[0], N_f) + np.fft.fft(left_closed_ear, N_f) * weightings_array[0]
filter_right_frequency_3 = np.fft.fft(right_concha, N_f) * np.fft.fft(equalization_array[1], N_f) + np.fft.fft(right_closed_ear, N_f) * weightings_array[1]
filter_left_frequency_2 = np.fft.fft(left_concha, N_f) * np.fft.fft(equalization_array[0], N_f) + np.fft.fft(left_closed_ear, N_f)
filter_right_frequency_2 = np.fft.fft(right_concha, N_f) * np.fft.fft(equalization_array[1], N_f) + np.fft.fft(right_closed_ear, N_f)
filter_left_frequency_1 = np.fft.fft(left_concha, N_f) * np.fft.fft(equalization_array[0], N_f)
filter_right_frequency_1 = np.fft.fft(right_concha, N_f) * np.fft.fft(equalization_array[1], N_f)

#plt.plot(left_concha, label="concha")
#plt.plot(equalization_array[0], label="mean")
#plt.plot(equalization_array_front[0], label="front")
#plt.plot(left_closed_ear, label="left_closed_ear")
#plt.plot(np.fft.ifft(weightings_array[0]), label="Weightings")
plt.plot(np.fft.ifft(filter_left_frequency_3), label="filter_left_frequency_3")
plt.plot(np.fft.ifft(filter_left_frequency_2), label="filter_left_frequency_2")
plt.plot(np.fft.ifft(filter_left_frequency_1), label="filter_left_frequency_1")
plt.legend()
plt.grid()
plt.show()

filter_1 = np.vstack((np.real(np.fft.ifft(filter_left_frequency_1)), np.real(np.fft.ifft(filter_right_frequency_1))))
filter_2 = np.vstack((np.real(np.fft.ifft(filter_left_frequency_2)), np.real(np.fft.ifft(filter_right_frequency_2))))
filter_3 = np.vstack((np.real(np.fft.ifft(filter_left_frequency_3)), np.real(np.fft.ifft(filter_right_frequency_3))))

sf.write('Filter_1.wav', filter_1.T, 48000, subtype="DOUBLE")
sf.write("filter_2.wav", filter_2.T, 48000, subtype="DOUBLE")
sf.write("filter_3.wav", filter_3.T, 48000, subtype="DOUBLE")

exit()

filter_frequency_1 = np.vstack((filter_left_frequency_1, filter_right_frequency_1))
filter_frequency_2 = np.vstack((filter_left_frequency_2, filter_right_frequency_2))
filter_frequency_3 = np.vstack((filter_left_frequency_3, filter_right_frequency_3))

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

                #plt.plot(f_bins[:N_f // 2], 20 * np.log10(brir_new[:N_f // 2]))
                #plt.plot(f_bins[:N_f//2], brir_db)
                #plt.xscale("log")
                #plt.show()

                brir_neu_1 = np.real( np.roll(np.fft.ifft(brir_fft * filter_frequency_1)[:, :length], -roll_length) * window )
                brir_neu_2 = np.real( np.roll(np.fft.ifft(brir_fft * filter_frequency_2)[:, :length], -roll_length) * window )
                brir_neu_3 = np.real(np.roll(np.fft.ifft(brir_fft * filter_frequency_3)[:, :length], -roll_length) * window)


                # samples x channels
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}/{position}/brir' + str(angle * 4) + '.wav', brir.T, 48000, subtype="DOUBLE")
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}_HP1/{position}/brir'+str(angle*4)+'.wav', brir_neu_1.T, 48000, subtype="DOUBLE")
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}_HP2/{position}/brir' + str(angle * 4) + '.wav', brir_neu_2.T, 48000, subtype="DOUBLE")
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}_HP3/{position}/brir' + str(angle * 4) + '.wav', brir_neu_3.T, 48000, subtype="DOUBLE")

