import csv
import os

positions = range(125, 326, 25)
speakers = ["Front", "Side"]
rooms = ["Audiolab", "Seminarroom"]
angles = range(0, 360, 4)


working_directory = os.getcwd()
directory_list = working_directory.split("\\")
if directory_list[-1] != "pyBinSim":
    directory_list = directory_list[:directory_list.index("pyBinSim") + 1]
    working_directory = "/".join(directory_list)
outfile_path = working_directory + "/example/"

for room in rooms:
    for speaker in speakers:
        filterlist = list()

        for position in positions:
            for angle in angles:
                # Value 1 - 3: listener orientation[yaw, pitch, roll]
                # Value 4 - 6: listener position[x, y, z]
                # Value 7 - 9: custom values[a, b, c]

                filterlist.append(["DS", angle, 0, 0, position, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, outfile_path + f"brirs/BRIR_{speaker}_{room}/{position}/brir{angle}.wav"])


        with open(outfile_path + f"/filters_{room}_{speaker}.txt", mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=" ")
            for row in filterlist:
                writer.writerow(row)
            csv_file.close()
