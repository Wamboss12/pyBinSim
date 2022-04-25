import copy
import math
import random


def random_loudnessad_justment(loudness, min_adjust=-3, max_adjust=1):

    if min_adjust > max_adjust:
        save = copy.deepcopy(min_adjust)
        min_adjust = copy.deepcopy(max_adjust)
        max_adjust = copy.deepcopy(save)

    loudness_adjustment_in_dB = random.randrange(min_adjust * 10, max_adjust * 10)/10
    print("adjustment in dB", loudness_adjustment_in_dB)
    loudness_in_dB = 20 * math.log10(loudness)

    return round(math.pow(10, (loudness_in_dB + loudness_adjustment_in_dB)/20) - loudness, 3)


l = 0.5
l_ = random_loudnessad_justment(l, -6, 6)
print(l, "+", l_)
print("original", round(20 * math.log10(l), 3))
print("new", round(20 * math.log10(l + l_), 3))
