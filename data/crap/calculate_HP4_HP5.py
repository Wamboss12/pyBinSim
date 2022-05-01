import numpy as np
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

path_to_HRIR = "../HRIR_KEMAR_AC0010_1.mat"
path_to_Driver = "../DriverResponse_KEMAR_AC0010_1.mat"
path_to_brir = "../../example/brirs/BRIR_Front_Audiolab/125/brir0.wav"
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


def array_numpy2openMHA(arr, out_path=None, decimals=6, outname="filter", timecode=True):
    """
    Diese Methode wandelt einen numpy-Array so um, dass er dem openMHA-Format entspricht. Anschliessend wird dieser als
    .txt-Datei gespeichert.

    ---------------------% openMHA-Format %---------------------
    vector = [1.0 2.7 4]
    matrix = [[1 2 3];[4 5 6]]

    complex = (1.3 + 2.7i)
    vcomplex = [(1.3 + 2.7i) (2.0 - 1.1i) 6.3] <--- rein reelle Werte koennen trotzdem ohne runde Klammern
    ------------------------------------------------------------

    :param arr: numpy-Array, der umgewandelt werden soll
    :param out_path: (optional) Pfad, an dem die neue .txt-Datei gespeichert werden soll;
                     default="/openMHA/filter_data/"
    :param decimals: (optional) Anzahl der Nachkommastellen, default=6
    :param outname: (optional) Name, der in .txt-Dateinamen mitverwendet werden soll, default="filter"
    :param timecode: (optional) Timecode an Speichernamen anhängen, default=True

    :return: _
    """
    assert 0 < len(arr.shape) < 3, f"Es sind nur 1D und 2D Matritzen in openMHA erlaubt! (Diese Funktion ist nicht fuer " \
                                   f"Beamformingmatritzen)\n" \
                                   f"Inputdimensionen: {len(arr.shape)}"

    arr = arr.round(decimals=decimals)

    resulting_text = "["
    if len(arr.shape) == 1:
        if "complex" in str(arr.dtype):
            real_values = np.real(arr).astype("str")
            imag_values = np.imag(arr)
            imag_signs = np.where(np.sign(imag_values) > 0, "+", "-")
            imag_values = np.abs(imag_values).astype("str")

            resulting_text += "("+real_values[0]+" "+imag_signs[0]+" "+imag_values[0]+"i)"
            for value_index in range(1, arr.shape[0]):
                resulting_text += " ("+real_values[value_index]+" "+imag_signs[value_index]+" "+imag_values[value_index]+"i)"
        else:
            arr = arr.astype("str")
            resulting_text += arr[0]
            for value in arr[1:]:
                resulting_text += " " + value
    else:
        resulting_rows_list = list()
        if "complex" in str(arr.dtype):
            real_values = np.real(arr).astype("str")
            imag_values = np.imag(arr)
            imag_signs = np.where(np.sign(imag_values) > 0, "+", "-")
            imag_values = np.abs(imag_values).astype("str")

            for row in range(arr.shape[0]):
                resulting_row = "["
                resulting_row += "(" + real_values[row, 0] + " " + imag_signs[row, 0] + " " + imag_values[row, 0] + "i)"
                for value_index in range(1, arr.shape[1]):
                    resulting_row += " (" + real_values[row, value_index] + " " + imag_signs[row, value_index] + " " + \
                                      imag_values[row, value_index] + "i)"
                resulting_row += "]"
                resulting_rows_list.append(resulting_row)
        else:
            arr = arr.astype("str")
            for row in range(arr.shape[0]):
                resulting_row = "["
                resulting_row += arr[row, 0]
                for value in arr[row, 1:]:
                    resulting_row += " " + value
                resulting_row += "]"
                resulting_rows_list.append(resulting_row)

        resulting_text += resulting_rows_list[0]
        for resulting_row in resulting_rows_list[1:]:
            resulting_text += ";" + resulting_row
    resulting_text += "]"

    if timecode:
        out_file_path = time.strftime(f"%Y_%m_%d--%H_%M_%S--{outname}.txt")
    else:
        out_file_path = f"{outname}.txt"
    if out_path is not None:
        if os.path.exists(out_path):
            if out_path[-1] not in ["/", "\\"]:
                out_path += "/"
            out_file_path = out_path + out_file_path
        else:
            print(f"Der Pfad   {out_path}   existiert nicht! Standardpfad wird stattdessen verwendet...")
            out_path = None

    if out_path is None:
        working_directory = os.getcwd()
        directory_list = working_directory.split("\\")
        if directory_list[-1] != "Masterarbeit":
            directory_list = directory_list[:directory_list.index("Masterarbeit")+1]
            out_file_path = "/".join(directory_list) + \
                            "/openMHA/filter_data/" + out_file_path
        else:
            out_file_path = working_directory + "/openMHA/filter_data/" + out_file_path
    print(f"Array wird in   {out_file_path}   gespeichert.")

    with open(out_file_path, "w") as file:
        file.write(resulting_text)



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
left_reverb = brir[:, 0] * window_reverb
right_reverb = brir[:, 1] * window_reverb

left_smoothed_direct = smooth_complex(N_F, 48000, 1 / 6, np.abs(left_direct))
left_smoothed_transparency = smooth_complex(N_F, 48000, 1 / 6, np.abs(left_equalization_array * left_concha))
right_smoothed_direct = smooth_complex(N_F, 48000, 1 / 6, np.abs(right_direct))
right_smoothed_transparency = smooth_complex(N_F, 48000, 1 / 6, np.abs(right_equalization_array * right_concha))

weightings_left = left_smoothed_transparency / left_smoothed_direct
weightings_altered_left = np.min(np.vstack([weightings_left, np.ones(N_F) * 16]), axis=0)
weightings_right = right_smoothed_transparency / right_smoothed_direct
weightings_altered_right = np.min(np.vstack([weightings_right, np.ones(N_F) * 16]), axis=0)
print("weightings calculated!\n")

"""
plt.plot(freq_bins[:N_F//2], weightings_left[:N_F//2], label="normal weightings")
plt.plot(freq_bins[:N_F//2], weightings_altered_left[:N_F//2], label="altered weightings")
plt.grid()
plt.xscale("log")
plt.legend()
plt.show()
"""

plt.plot(freq_bins[:N_F//2], np.angle(left_direct)[:N_F//2], label="left direct")
plt.plot(freq_bins[:N_F//2], np.angle(np.fft.fft(np.real(np.fft.ifft(left_direct * weightings_left)) * window_direct, N_F))[:N_F//2], label="new")
plt.legend()
plt.grid()
plt.xscale("log")
plt.show()

weighted_direct_left = np.real(np.fft.ifft(left_direct * weightings_left)) * window_direct
weighted_direct_right = np.real(np.fft.ifft(right_direct * weightings_right)) * window_direct
new_brir_left = np.fft.fft(weighted_direct_left + left_reverb, N_F)
new_brir_right = np.fft.fft(weighted_direct_right + right_reverb, N_F)

new_brir_left = np.real(np.fft.ifft(new_brir_left + left_closed_ear, n=N_F))
new_brir_right = np.real(np.fft.ifft(new_brir_right + right_closed_ear, n=N_F))

plt.plot(brir[:, 0], label="left brir")
plt.plot(new_brir_left, label="new left_brir")
plt.grid()
plt.legend()
plt.show()


"""
plt.plot(freq_bins[:N_F//2], 20 * np.log10(np.abs(left_direct))[:N_F//2], label = "left_direct")
plt.plot(freq_bins[:N_F//2], 20 * np.log10(np.abs(left_equalization_array * left_concha))[:N_F//2], label = "left equalization")
plt.plot(freq_bins[:N_F//2], 20 * np.log10(np.abs(left_direct * weightings_altered_left))[:N_F//2], label = "new left_direct")
plt.xscale("log")
plt.grid()
plt.legend()
plt.show()
"""

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

            max_left = np.argmax(brir[0, :dict_grenzen[position] + 30])
            max_right = np.argmax(brir[1, :dict_grenzen[position] + 30])
            window_direct_left = np.hstack([np.ones(max_left + 60), windows.triang(61)[31:], np.zeros(16384 - max_left - 90)])
            window_direct_right = np.hstack([np.ones(max_right + 60), windows.triang(61)[31:], np.zeros(16384 - max_right - 90)])
            window_reverb_left = np.hstack([np.zeros(max_left + 60), windows.triang(61)[:30], np.ones(16384 - max_left - 90)])
            window_reverb_right = np.hstack([np.zeros(max_right + 60), windows.triang(61)[:30], np.ones(16384 - max_right - 90)])

            left_direct = np.fft.fft(brir[0, :] * window_direct_left, N_F)
            right_direct = np.fft.fft(brir[1, :] * window_direct_right, N_F)
            left_reverb = brir[0, :] * window_reverb_left
            right_reverb = brir[1, :] * window_reverb_right

            weighted_direct_left = np.real(np.fft.ifft(left_direct * weightings_left)) * window_direct_left
            weighted_direct_right = np.real(np.fft.ifft(right_direct * weightings_right)) * window_direct_right

            new_brir_left = np.fft.fft(weighted_direct_left + left_reverb, N_F)
            new_brir_right = np.fft.fft(weighted_direct_right + right_reverb, N_F)

            new_brir_left = np.real(np.fft.ifft(new_brir_left + left_closed_ear, n=N_F))
            new_brir_right = np.real(np.fft.ifft(new_brir_right + right_closed_ear, n=N_F))
            new_brir = np.vstack([new_brir_left, new_brir_right])

            weighted_left = np.fft.fft(brir[0], N_F) * weightings_left
            weighted_right = np.fft.fft(brir[1], N_F) * weightings_right

            new_brir_left_alt = np.real(np.fft.ifft(weighted_left + left_closed_ear, n=N_F))
            new_brir_right_alt = np.real(np.fft.ifft(weighted_right + right_closed_ear, n=N_F))
            new_brir_alt = np.vstack([new_brir_left_alt, new_brir_right_alt])

            # samples x channels
            sf.write(f'C:/Users/student/master_doll/pyBinSim-lightweight/example/brirs/BRIR_{speaker}_{room}_HP4/{position}/brir'+str(angle*4)+'.wav', new_brir.T, 48000, subtype="DOUBLE")
            sf.write(f'C:/Users/student/master_doll/pyBinSim-lightweight/example/brirs/BRIR_{speaker}_{room}_HP5/{position}/brir' + str(angle * 4) + '.wav', new_brir_alt.T, 48000, subtype="DOUBLE")