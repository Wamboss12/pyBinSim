import numpy as np
import logging
import ast
import scipy.io
import soundfile as sf
import msvcrt

a = [1, 2]
b = [3, 4]
c = a + b
print(c)

while True:
    if msvcrt.kbhit():
        # char = msvcrt.getch()
        key = ord(msvcrt.getch())
        print(key)