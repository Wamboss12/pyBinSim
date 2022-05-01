import soundfile as sf
import os
from scipy.io import loadmat


positions = range(125, 326, 25)
speakers = ["Front", "Side"]
rooms = ["Audiolab", "Seminarroom"]
for room in rooms:
    print(room)
    for speaker in speakers:
        print(" ->",speaker)
        for position in positions:
            print("    ->",position)
            if not os.path.isdir(f"../example/brirs/BRIR_{speaker}_{room}/{position}"):
                os.mkdir(f"../example/brirs/BRIR_{speaker}_{room}/{position}")

            brirs = loadmat(f'../example/brirs/BRIR_{speaker}_{room}/brirs{position}.mat')

            matrixes = brirs.get('brirMat')
            for angle in range(90):
                # brir_0_links = matrixes[angle,0].T
                # brir_0_rechts = matrixes[angle,1].T
                brir = matrixes[angle].T
                sf.write(f'../example/brirs/BRIR_{speaker}_{room}/{position}/brir'+str(angle*4)+'.wav', brir, 48000, subtype="DOUBLE")