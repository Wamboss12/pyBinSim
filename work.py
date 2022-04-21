import numpy as np
import logging
import ast
import msvcrt

while True:
    if msvcrt.kbhit():
        # char = msvcrt.getch()
        key = ord(msvcrt.getch())
        print(key)