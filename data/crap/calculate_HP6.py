import pickle
import soundfile as sf
from scipy.io import loadmat
import math
import copy
import numpy as np

with open("weightings_brir_with_leaking_direct.pkl", "rb") as file:
    weightings = pickle.load(file)

N_F = 16384
s_rate = 48000


mean_weightings = {}
for direction in range(0, 360, 30):
    direction_Mean = np.zeros((2, N_F))
    for device in ["AC0010", "AC0011"]:
        for i in range(1, 4):
            direction_Mean += weightings[device][i][direction]
    direction_Mean /= 6
    mean_weightings[direction] = direction_Mean
mean_weightings[360] = copy.deepcopy(mean_weightings[0])

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
            brir = np.fft.fft(matrixes[angle], N_F)


            if speaker == "Front":
                new_angle = angle * 4
            else:
                adaption = 0
                if position < 175:
                    adaption = 180 - math.degrees(math.atan(125/abs(175 - position)))
                elif position == 175:
                    adaption = 90
                else:
                    adaption = math.degrees(math.atan(125/abs(175 - position)))
                new_angle = angle * 4 + round(adaption, 1)
                if new_angle > 360:
                    new_angle = new_angle - 360

            rest_angle = new_angle % 30
            upper = int( math.ceil(new_angle / 30) * 30 )
            lower = int( math.floor(new_angle / 30) * 30 )
            upper_weight = rest_angle / 30

            weighting = mean_weightings[upper] * upper_weight + mean_weightings[lower] * (1 - upper_weight)

            new_brir_left = np.real(np.fft.ifft(brir[0] * weighting[0], n=N_F))
            new_brir_right = np.real(np.fft.ifft(brir[1] * weighting[1], n=N_F))
            new_brir = np.vstack([new_brir_left, new_brir_right])

            # samples x channels
            sf.write(f'C:/Users/student/master_doll/pyBinSim-lightweight/example/brirs/BRIR_{speaker}_{room}_HP6/{position}/brir'+str(angle*4)+'.wav', new_brir.T, 48000, subtype="DOUBLE")
