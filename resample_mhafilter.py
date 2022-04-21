import os.path
import numpy as np
import ast
import soundfile as sf

filter = "mean_filter_279_48000.txt.txt"

with open(filter, "r") as file:
    text = file.read()

    text = text.replace(";", ",")
    text_splitted = text.split(sep=" ")
    new_text = ",".join(text_splitted)

list_from_text = ast.literal_eval(new_text)
array_from_list = np.array(list_from_text)

array_to_save = np.hstack([array_from_list, np.zeros((2, 512-279))])
print(array_to_save.shape)
sf.write("HP_mean_filter_512_48000.wav", array_to_save.T, 48000)

#plt.plot(20*np.log10(np.abs(np.fft.fft(array_from_list[0], 512))), label="old")
#plt.plot(20*np.log10(np.abs(np.fft.fft(resampled_array[0], 512))), alpha=0.5, label="new resample")
#plt.grid()
#plt.show()