# This file is part of the pyBinSim project.
#
# Copyright (c) 2017 A. Neidhardt, F. Klein, N. Knoop, T. Köllmer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import copy

from pythonosc import osc_message_builder
from pythonosc import udp_client
import string
import math
import sys
import time
import openvr
import msvcrt
import numpy as np
import random


def matrixFromPose(pose):
    v = pose.mDeviceToAbsoluteTracking

    pose_matrix = np.matrix(
        [[v[0][0], v[0][1], v[0][2], v[0][3]], [v[1][0], v[1][1], v[1][2], v[1][3]],
         [v[2][0], v[2][1], v[2][2], v[2][3]]])

    return pose_matrix


def rotatePose(pose):
    # Drehwinkel
    alpha = np.radians(0)

    # Drehung x Achse
    rotation_matrix = np.matrix(
        [[1, 0, 0], [0, np.cos(alpha), np.sin(alpha) * -1], [0, np.sin(alpha), np.cos(alpha)]])

    # Drehung y Achse
    # rotation_matrix = np.matrix(
    #    [[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [np.sin(alpha) * -1, 0, np.cos(alpha)]])

    # Drehung Z Achse
    # rotation_matrix = np.matrix(
    #    [[np.cos(alpha), np.sin(alpha) * -1, 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])

    pose[0:3, 0:3] = np.dot(pose[0:3, 0:3], rotation_matrix)
    pose_rotated = pose

    return pose_rotated


def getEulerAngles(pose):
    yawRad = np.arctan2(pose.item((0, 2)), pose.item((2, 2)))
    # yaw = int(round(np.degrees(yawRad)))
    yaw = np.round(np.degrees(yawRad), 1)

    pitchRad = np.arctan2(-pose.item((1, 2)), np.sqrt(np.square(pose.item((0, 2))) + np.square(pose.item((2, 2)))))
    # pitch = int(round(np.degrees(pitchRad)))
    pitch = np.round(np.degrees(pitchRad), 1)

    rollRad = np.arctan2(pose.item((1, 0)), pose.item((1, 1)))
    # roll = int(round(np.degrees(rollRad)))
    roll = np.round(np.degrees(rollRad), 1)

    return yaw, pitch, roll


def getXYZPosition(pose):
    posX = pose.item((0, 3))
    posY = pose.item((1, 3))
    posZ = pose.item((2, 3)) - .2
    return posX, posY, posZ


def random_loudnessad_justment(loudness, min_adjust=-3, max_adjust=1):
    if min_adjust > max_adjust:
        save = copy.deepcopy(min_adjust)
        min_adjust = copy.deepcopy(max_adjust)
        max_adjust = copy.deepcopy(save)

    loudness_adjustment_in_dB = random.randrange(min_adjust * 10, max_adjust * 10) / 10
    loudness_in_dB = 20 * math.log10(loudness)

    return round(math.pow(10, (loudness_in_dB + loudness_adjustment_in_dB) / 20) - loudness,
                 3), loudness_adjustment_in_dB


def start_tracker():
    oscIdentifier_ds = '/pyBinSim_ds_Filter'
    oscIdentifier_loudness = '/pyBinSimLoudness'
    oscIdentifier_convolution = '/pyBinSimPauseConvolution'
    oscIdentifier_playback = '/pyBinSimPauseAudioPlayback'
    oscIdentifier_soundfile = '/pyBinSimFile'
    oscIdentifier_mapping = '/pyBinSimMapping'
    oscIdentifier_hp = "/pyBinSimHP"

    ip = '127.0.0.1'
    port_ds = 10000
    port_misc = 10003

    nSources = 2
    minAngleDifference = 4
    minxDifference = 25

    # Internal settings:
    yawVectorSubSampled = range(0, 360, minAngleDifference)
    positionVectorSubSampled = range(0, 201, minxDifference)

    # Create OSC client
    client_ds = udp_client.SimpleUDPClient(ip, port_ds)
    client_misc = udp_client.SimpleUDPClient(ip, port_misc)

    # init openvr for HTC Vive
    help(openvr.VRSystem)
    vr = openvr.init(openvr.VRApplication_Scene)

    poses = openvr.VRSystem().getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0,
                                                              openvr.k_unMaxTrackedDeviceCount)
    # find tracker
    trackernr = 3
    print(openvr.k_unMaxTrackedDeviceCount)
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if poses[i].bPoseIsValid:
            device_class = vr.getTrackedDeviceClass(i)
            if (device_class == openvr.TrackedDeviceClass_GenericTracker):
                trackernr = i
                print(['Tracker number is: ', trackernr])
                break
    if trackernr == 99:
        print("Could not find suitable tracker!")
        exit()

    loudness_trans = 0.07
    loudness_bin = 0.34
    loudness_adjustment_trans = 0.0
    loudness_adjustment_bin = 0.0
    db_trans = 0
    db_bin = 0
    max_loudness_adjustment = 1
    min_loudness_adjustment = -3
    loudness_step = 0.01

    state = "normal"
    BRIR = 0
    BRIRs = ["normal", "hearthrough", "idmt_minimum", "oldenburg", "oldenburg_minimum"]
    use_HP = False
    pauseConvolution = False
    pausePlayback = True
    pausePlayback_binsim = True
    pausePlayback_tracking = False
    pausePlayback_tracking_before = False
    pausePlayback = True
    stop_for_deviation = False

    source_front = "virtual"
    source_side = "real"

    distance_tracker_to_ears = 10
    max_distance_to_line = 15
    max_pitch_difference = 20
    max_roll_difference = 15

    yaw_offset = 0
    position_offset = 0
    posX_offset = 0

    radius = 0
    pitch_offset = 0
    roll_offset = 0

    in_channels_trans = [1, 2]
    in_channels_bin = [0, 3]
    in_channels_trans_before = []
    in_channels_bin_before = []
    virtual_outchannel_bin = [0, 1]  # erste Quelle auf LS 1
    out_channels_trans = [1, 0]  # erste Quelle auf LS 2
    save_in_channels = []

    audio_files = ["example/new_signals/Dialog_highpassed.wav",
                   "example/new_signals/music_sherlock_highpassed_200.wav",
                   "example/new_signals/music_sherlock_alt_highpassed_200.wav"]
    audio_index = 0

    settings = {
        # S: real   F: virtual
        0: {"in_trans": [0, 2], "in_bin": [1, 3], "out_trans": [1, 0], "out_bin": [0, 1], "loud_adjust": loudness_bin_HP},"audio": 0,
        # S: real   F: virtual
        1: {"in_trans": [1, 2], "in_bin": [0, 3], "out_trans": [1, 0], "out_bin": [0, 1], "loud_adjust": loudness_bin_HP, "audio": 0},
        # S: real   F: real
        2: {"in_trans": [0, 1], "in_bin": [2, 3], "out_trans": [1, 0], "out_bin": [0, 1], "loud_adjust": loudness_bin_HP, "audio": 0},
        # S: virtual   F: virtual
        3: {"in_trans": [2, 3], "in_bin": [0, 1], "out_trans": [1, 0], "out_bin": [0, 1], "loud_adjust": loudness_bin_HP, "audio": 0},

        # S: virtual   F: real
        4: {"in_trans": [0, 2], "in_bin": [1, 3], "out_trans": [0, 1], "out_bin": [1, 0], "loud_adjust": loudness_bin_HP, "audio": 0},
        # S: virtual   F: real
        5: {"in_trans": [1, 2], "in_bin": [0, 3], "out_trans": [0, 1], "out_bin": [1, 0], "loud_adjust": loudness_bin_HP, "audio": 0},
        # S: real   F: real
        6: {"in_trans": [0, 1], "in_bin": [2, 3], "out_trans": [0, 1], "out_bin": [1, 0], "loud_adjust": loudness_bin_HP, "audio": 0},
        # S: virtual   F: virtual
        7: {"in_trans": [2, 3], "in_bin": [0, 1], "out_trans": [0, 1], "out_bin": [1, 0], "loud_adjust": loudness_bin_HP, "audio": 0},

        # S: real   F: virtual
        8: {"in_trans": [0, 2], "in_bin": [1, 3], "out_trans": [1, 0], "out_bin": [0, 1], "loud_adjust": loudness_bin_HP, "audio": 1},
        # S: real   F: virtual
        9: {"in_trans": [1, 2], "in_bin": [0, 3], "out_trans": [1, 0], "out_bin": [0, 1], "loud_adjust": loudness_bin_HP, "audio": 1},
        # S: real   F: real
        10: {"in_trans": [0, 1], "in_bin": [2, 3], "out_trans": [1, 0], "out_bin": [0, 1], "loud_adjust": loudness_bin_HP, "audio": 1},
        # S: virtual   F: virtual
        11: {"in_trans": [2, 3], "in_bin": [0, 1], "out_trans": [1, 0], "out_bin": [0, 1], "loud_adjust": loudness_bin_HP, "audio": 1},

        # S: virtual   F: real
        12: {"in_trans": [0, 2], "in_bin": [1, 3], "out_trans": [0, 1], "out_bin": [1, 0], "loud_adjust": loudness_bin_HP, "audio": 2},
        # S: virtual   F: real
        13: {"in_trans": [1, 2], "in_bin": [0, 3], "out_trans": [0, 1], "out_bin": [1, 0], "loud_adjust": loudness_bin_HP, "audio": 2},
        # S: real   F: real
        14: {"in_trans": [0, 1], "in_bin": [2, 3], "out_trans": [0, 1], "out_bin": [1, 0], "loud_adjust": [], "audio": 2},
        # S: virtual   F: virtual
        15: {"in_trans": [2, 3], "in_bin": [0, 1], "out_trans": [0, 1], "out_bin": [1, 0], "loud_adjust": [], "audio": 2}

        }

    try:
        while 1:
            poses = openvr.VRSystem().getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0,
                                                                      openvr.k_unMaxTrackedDeviceCount)

            # get data from tracker
            tracker_pose = matrixFromPose(poses[trackernr])
            tracker_pose = rotatePose(tracker_pose)

            # get the Trackingdata in the disired form
            yaw, pitch, roll = getEulerAngles(tracker_pose)
            posX, posY, posZ = getXYZPosition(tracker_pose)
            posZ = -1 * posZ

            yaw = yaw + 0
            if yaw < 0:
                yaw = 360 + yaw

            if msvcrt.kbhit():
                # char = msvcrt.getch()
                key = ord(msvcrt.getch())

                # key "p" for pausing the playback
                if key == 112:
                    pausePlayback_binsim = not pausePlayback_binsim
                    client_misc.send_message(oscIdentifier_playback, [str(pausePlayback_binsim)])
                # 'Space' for calibrate the position
                if key == 32:
                    position_offset = -1 * posZ
                    posX_offset = -1 * posX
                # key 'enter' to calibrate orientation
                if key == 13:
                    radius = distance_tracker_to_ears
                    yaw_offset = -1 * yaw
                    pitch_offset = -1 * pitch
                    roll_offset = -1 * roll
                    stop_for_deviation = True

                # key "h" to choose which HP filter to use
                if key == 113:
                    BRIR = BRIR + 1
                    if BRIR >= len(BRIRs):
                        BRIR = 0

                # key "w" to get a random loudness adjustment
                if key == 119:
                    loudness_adjustment_trans, db_trans = random_loudnessad_justment(loudness_trans,
                                                                                     min_loudness_adjustment,
                                                                                     max_loudness_adjustment)
                    loudness_adjustment_bin, db_bin = random_loudnessad_justment(loudness_bin,
                                                                                 min_loudness_adjustment,
                                                                                 max_loudness_adjustment)
                    client_misc.send_message(oscIdentifier_loudness, [loudness_trans + loudness_adjustment_trans,
                                                                      loudness_bin + loudness_adjustment_bin])

            current_position_before = (posZ + position_offset) * 100
            yaw += yaw_offset
            posX = (posX + posX_offset) * 100
            current_position_before += (math.cos(math.radians(yaw)) * radius)
            posX += (-1 * math.sin(math.radians(yaw)) * radius)
            pitch += pitch_offset
            roll += roll_offset
            if yaw < 0:
                yaw = 360 + yaw

            if stop_for_deviation:
                pausePlayback_tracking = (abs(pitch) > max_pitch_difference or abs(
                    roll) > max_roll_difference or current_position_before > 200 + max_distance_to_line or current_position_before < -max_distance_to_line or abs(
                    posX) > max_distance_to_line)

            if pausePlayback_tracking != pausePlayback_tracking_before:
                if pausePlayback_tracking:
                    client_misc.send_message(oscIdentifier_loudness, [0.0, 0.0])
                else:
                    client_misc.send_message(oscIdentifier_loudness, [loudness_trans + loudness_adjustment_trans,
                                                                      loudness_bin + loudness_adjustment_bin])
                pausePlayback_tracking_before = copy.deepcopy(pausePlayback_tracking)

            if (pausePlayback_binsim or pausePlayback_tracking) != pausePlayback:
                pausePlayback = (pausePlayback_binsim or pausePlayback_tracking)
                # client_misc.send_message(oscIdentifier_playback, [str(pausePlayback)])

            # Build and send OSC message
            # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
            yaw_before = copy.deepcopy(yaw)
            yaw = min(yawVectorSubSampled, key=lambda x: abs(x - yaw))
            current_position = int(np.argmin(
                np.array([round(abs(x - current_position_before) / 25) for x in positionVectorSubSampled])) * 25)

            if in_channels_trans[0] in [0, 1]:
                if out_channels_trans[0] == 0:
                    source_front = f"real ({in_channels_trans[0]})"
                else:
                    source_side = f"real ({in_channels_trans[0]})"
            if in_channels_trans[1] in [0, 1]:
                if out_channels_trans[1] == 0:
                    source_front = f"real ({in_channels_trans[1]})"
                else:
                    source_side = f"real ({in_channels_trans[1]})"
            if in_channels_bin[0] in [0, 1]:
                if virtual_outchannel_bin[0] == 0:
                    source_front = f"virtual ({in_channels_bin[0]})"
                else:
                    source_side = f"virtual ({in_channels_bin[0]})"
            if in_channels_bin[1] in [0, 1]:
                if virtual_outchannel_bin[1] == 0:
                    source_front = f"virtual ({in_channels_bin[1]})"
                else:
                    source_side = f"virtual ({in_channels_bin[1]})"

            print(
                f"z/x: {current_position} ({round(current_position_before)}) / {round(posX)}    "
                f"y/r/p: {round(yaw)} ({round(yaw_before)}) / {round(roll)} / {round(pitch)}    "
                f"vol: ({round(loudness_trans + loudness_adjustment_trans, 3)}|{round(loudness_bin + loudness_adjustment_bin, 3)})  "
                f"adjusted: ({db_trans} dB|{db_bin} dB)   "
                f"Sound: {str(not pausePlayback)[0]} ({str(not pausePlayback_binsim)[0]}/{str(not pausePlayback_tracking)[0]})    "
                f"audio: {audio_files[audio_index].split('/')[-1]}    "
                f"BRIR: {BRIRs[BRIR]}    "
                f"Front: {source_front}    "
                f"Side: {source_side}"
            )

            # build OSC Message and send it
            for n in range(0, nSources):
                # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
                binsim_ds = [n, int(round(yaw)), 0, 0, current_position, 0, 0, 0, 0, 0, 0, 0, 0, 0, BRIR,
                             virtual_outchannel_bin[n]]
                client_ds.send_message(oscIdentifier_ds, binsim_ds)

            sys.stdout.flush()

            time.sleep(0.01)

    except KeyboardInterrupt:  # Break if ctrl+c is pressed

        # Console output
        print("Done")


if __name__ == "__main__":
    start_tracker()
