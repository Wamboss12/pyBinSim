import soundfile as sf
import numpy as np
import logging
import ast

x = [1, 'bin', 2, 'bin']
x_ = [i for i in x if i != 'bin']
print(len(x_))
print(len(set(x_)))
print(set(x_))