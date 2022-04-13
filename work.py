import soundfile as sf
import numpy as np

samplerate = 44100
block_size = 128
desired_length_in_samples = 44100

soundPath_to_write = ""
soundPath_to_read = ""
data_length = (desired_length_in_samples // block_size)*block_size
print("data length = {} samples".format(data_length))

data = np.zeros((samplerate, 2), dtype='float32')
data[0, :] = 1.0

sf.write('example/dirac_44100.wav', data, samplerate)
audio_file_data, fs = sf.read('example/dirac_44100.wav', dtype='float32', )
print("dirac shape = {}".format(audio_file_data.shape))
audio_file_data, fs = sf.read('example/example-IR.wav', dtype='float32', )
print("dirac shape = {}".format(audio_file_data.shape))


