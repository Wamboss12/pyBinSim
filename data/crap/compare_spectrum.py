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

path_to_HRIR = "../HRIR_KEMAR_AC0010_1.mat"
path_to_Driver = "../DriverResponse_KEMAR_AC0010_1.mat"
path_to_brir = "../../example/brirs/BRIR_Front_Audiolab/125/brir0.wav"
N_F = 16384
freq_bin = 48000/N_F
freq_bins = np.arange(N_F) * freq_bin

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
left_tweeter = np.fft.fft(librosa.resample(y=loadmat(path_to_Driver).get("M_data")[:, 0, 0], orig_sr=44100, target_sr=48000), N_F)

right_equalization_array = np.fft.fft(equalization_array[1], N_F)
right_concha = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 9], orig_sr=44100, target_sr=48000), N_F)
right_closed_ear = np.fft.fft(librosa.resample(y=loadmat(path_to_HRIR).get("M_data")[:, 0, 1], orig_sr=44100, target_sr=48000), N_F)
right_tweeter = np.fft.fft(librosa.resample(y=loadmat(path_to_Driver).get("M_data")[:, 1, 1], orig_sr=44100, target_sr=48000), N_F)

brir, srate = sf.read(path_to_brir)
left_brir = np.fft.fft(brir[:, 0], N_F)
right_brir = np.fft.fft(brir[:, 1], N_F)

spectrum_transparency_left = np.abs(left_concha * left_equalization_array * left_tweeter + left_closed_ear)
spectrum_synthesis_left = np.abs(left_brir * left_tweeter)
smoothes_spectrum_transparency_left = smooth_complex(N_F, 48000, 1 / 6, spectrum_transparency_left)
smoothes_spectrum_synthesis_left = smooth_complex(N_F, 48000, 1 / 6, spectrum_synthesis_left)

spectrum_transparency_right = np.abs(right_concha * right_equalization_array * right_tweeter + right_closed_ear)
spectrum_synthesis_right = np.abs(right_brir * right_tweeter)
smoothes_spectrum_transparency_right = smooth_complex(N_F, 48000, 1 / 6, spectrum_transparency_right)
smoothes_spectrum_synthesis_right = smooth_complex(N_F, 48000, 1 / 6, spectrum_synthesis_right)


weightings_left = smoothes_spectrum_transparency_left / smoothes_spectrum_synthesis_left
weightings_altered_left = np.min(np.vstack([weightings_left, np.ones(N_F) * 16]), axis=0)
altered_spectrum_synthesis_left = weightings_altered_left * spectrum_synthesis_left

weightings_right = smoothes_spectrum_transparency_right / smoothes_spectrum_synthesis_right
weightings_altered_right = np.min(np.vstack([weightings_right, np.ones(N_F) * 16]), axis=0)
altered_spectrum_synthesis_right = weightings_altered_right * spectrum_synthesis_right

weightings = np.vstack([weightings_left, weightings_right])
weightings_normal_left = (left_concha * left_equalization_array * left_tweeter + left_closed_ear) / (left_brir * left_tweeter)
weightings_normal_right = (right_concha * right_equalization_array * right_tweeter + right_closed_ear) / (right_brir * right_tweeter)

a = np.real(np.fft.ifft(weightings_normal_left))
sf.write("weightings_normal.wav", a, 48000, subtype="DOUBLE")
plt.plot(np.fft.ifft(weightings_normal_left))
plt.show()

array_numpy2openMHA(weightings, out_path=os.getcwd(), outname="weightings_front", timecode=False, decimals=9)
array_numpy2openMHA(np.vstack([weightings_normal_left, weightings_normal_right]), out_path=os.getcwd(), outname="weightings_front_normal", timecode=False, decimals=9)

#plt.plot(freq_bins[:N_F//2], weightings_altered_right[:N_F//2], label="right")
#plt.plot(freq_bins[:N_F//2], weightings_altered_left[:N_F // 2], label="left")
#plt.grid()
#plt.legend()
#plt.xscale("log")
#plt.show()


plt.plot(freq_bins[:N_F//2], 20 * np.log10(spectrum_transparency_left)[:N_F // 2], label ="transparency")
plt.plot(freq_bins[:N_F//2], 20 * np.log10(spectrum_synthesis_left)[:N_F // 2], label ="synthesis")
plt.plot(freq_bins[:N_F//2], 20 * np.log10(altered_spectrum_synthesis_left)[:N_F // 2], label ="altered synthesis", alpha = 0.5)
#plt.plot(freq_bins[:N_F//2], 20 * np.log10(smoothes_spectrum_transparency)[:N_F//2], label = "smoothed transparency")
#plt.plot(freq_bins[:N_F//2], 20 * np.log10(smoothes_spectrum_synthesis)[:N_F//2], label = "smoothed synthesis")
plt.legend()
plt.grid()
plt.xscale("log")
plt.show()
