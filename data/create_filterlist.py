import csv
import os

positions = range(0, 201, 25)
speakers = ["Front", "Side"]
rooms = ["Audiolab_HP1", "Audiolab_HP2", "Audiolab_HP3"]
angles = range(0, 360, 4)


working_directory = os.getcwd()
directory_list = working_directory.split("\\")
if directory_list[-1] != "pyBinSim":
    directory_list = directory_list[:directory_list.index("pyBinSim") + 1]
    working_directory = "/".join(directory_list)
outfile_path = working_directory + "/example/"

speaker_value = 0
filterlist = list()
for room in rooms:
    if room == "Audiolab_HP1":
        hp_value = 0
    elif room == "Audiolab_HP2":
        hp_value = 1
    else:
        hp_value = 2

    for speaker in speakers:
        if speaker == "Front":
            speaker_value = 0
        else:
            speaker_value = 1

        for position in positions:
            for angle in angles:
                # Value 1 - 3: listener orientation[yaw, pitch, roll]
                # Value 4 - 6: listener position[x, y, z]
                # Value 7 - 9: custom values[a, b, c]

                filterlist.append(["DS", angle, 0, 0, position, 0, 0, 0, 0, 0, 0, 0, 0, 0, hp_value, speaker_value, f"brirs/BRIR_{speaker}_{room}/{position + 125}/brir{angle}.wav"])


with open(outfile_path + f"/filters_Audiolab_new.txt", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file, delimiter=" ")
    for row in filterlist:
        writer.writerow(row)
    csv_file.close()
