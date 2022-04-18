import numpy as np
import logging
import ast
import scipy.io
import soundfile as sf


scipy.io.loadmat()

audio = "example/brirs/BRIR_Front_Audiolab/125/brir0.wav"
#audio = "example/brirs/BRIR_Front_Seminarroom/125/brir0.wav"
data, samplerate = sf.read(audio)

print(data.shape, samplerate)