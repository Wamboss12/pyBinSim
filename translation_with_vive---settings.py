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

from pythonosc import osc_message_builder
from pythonosc import udp_client
import string
import math
import sys
import time
import openvr
import msvcrt
import numpy as np
import copy

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


def start_tracker():
    oscIdentifier_ds = '/pyBinSim_ds_Filter'
    oscIdentifier_loudness = '/pyBinSimLoudness'
    oscIdentifier_playback = '/pyBinSimPauseAudioPlayback'
    oscIdentifierSetting = "/pyBinSimSetting"

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

    loudness_trans = 0.08
    loudness_bin = 0.05
    loudness_bin_HP = 0.60
    loudness_bin_woHP = 0.31
    loudness_step = 0.01
    pausePlayback_binsim = True
    pausePlayback_tracking = False
    pausePlayback = True
    stop_for_deviation = False
    audio_files = ["signals/noise_noise.wav",
                   "signals/speech_noise.wav",
                   "signals/bandpassNoise_silence.wav",
                   "signals/bandpassDialog.wav"]
    audio_index = 0
    virtual_outchannel_bin = [0, 1]  # erste Quelle auf LS 1
    use_HP = False
    settings = {
        # beides
        # real: weiblich
        1: {"audio": 3, "in_trans": [0, 2], "in_bin": [1, 3], "out_trans": [1, 0], "out_bin": [0, 1], "HP": False,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_woHP},
        # real: männlich
        2: {"audio": 3, "in_trans": [1, 2], "in_bin": [0, 3], "out_trans": [0, 1], "out_bin": [1, 0], "HP": False,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_woHP},
        #real: männlich
        3: {"audio": 3, "in_trans": [1, 2], "in_bin": [0, 3], "out_trans": [0, 1], "out_bin": [1, 0], "HP": True,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_HP},
        # real: weiblich
        4: {"audio": 3, "in_trans": [0, 2], "in_bin": [1, 3], "out_trans": [1, 0], "out_bin": [0, 1], "HP": True,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_HP},

        # virtuell
        5: {"audio": 3, "in_trans": [2, 3], "in_bin": [0, 1], "out_trans": [1, 0], "out_bin": [0, 1], "HP": True,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_HP},
        6: {"audio": 3, "in_trans": [2, 3], "in_bin": [0, 1], "out_trans": [1, 0], "out_bin": [0, 1], "HP": False,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_woHP},

        # real
        7: {"audio": 3, "in_trans": [0, 1], "in_bin": [2, 3], "out_trans": [0, 1], "out_bin": [1, 0], "HP": True,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_HP},

        8: {"audio": 2, "in_trans": [0, 1], "in_bin": [2, 3], "out_trans": [0, 1], "out_bin": [1, 0], "HP": False,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_woHP},
        9: {"audio": 3, "in_trans": [0, 1], "in_bin": [2, 3], "out_trans": [0, 1], "out_bin": [1, 0], "HP": False,
            "loudness_trans": loudness_trans, "loudness_bin": loudness_bin_woHP},

        }
    num_settings = len(settings.keys())
    num_previous_setting = 0
    num_current_setting = 1
    current_setting = settings[num_current_setting]

    # poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    # poses = poses_t()
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

    max_distance_to_line = 10
    max_pitch_difference = 15
    max_roll_difference = 15

    yaw_offset = 0
    position_offset = 0
    posX_offset = 0

    radius = 0
    pitch_offset = 0
    roll_offset = 0

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

            # pitch = pitch + 0
            # if pitch < 0:
            #	pitch = 360 + pitch

            # roll = roll
            #if roll < 0:
            #    roll = 360 + roll

            # key "+" on numpad for increasing volume
            if msvcrt.kbhit():
                # char = msvcrt.getch()
                key = ord(msvcrt.getch())

                # "up_arrow"
                if key == 72:
                    if pausePlayback_binsim and num_current_setting < num_settings:
                        num_current_setting += 1

                # "down_arrow"
                if key == 80:
                    if pausePlayback_binsim and num_current_setting > 1:
                        num_current_setting -= 1
                # key "p" for pausing the convolution
                if key == 112:
                    pausePlayback_binsim = not pausePlayback_binsim
                # 'Space' for setting the offset so that the current position and orientation are azimut = 0 and
                # position = 0
                if key == 32:
                    position_offset = -1 * posZ
                    posX_offset = -1 * posX

                # key 'enter' to calibrate orientation
                if key == 13:
                    radius = 10
                    yaw_offset = -1 * yaw
                    pitch_offset = -1 * pitch
                    roll_offset = -1 * roll
                    stop_for_deviation = True

                # key "+" on numpad for increasing volume for transparency
                if key == 43:
                    if loudness_trans < 1:
                        loudness_trans += loudness_step
                        client_misc.send_message(oscIdentifier_loudness, [loudness_trans, loudness_bin])

                # key "-" on numpad for increasing volume for transparency
                if key == 45:
                    if loudness_trans > 0.05:
                        loudness_trans -= loudness_step
                        client_misc.send_message(oscIdentifier_loudness, [loudness_trans, loudness_bin])

                # key "*" on numpad for increasing volume for binaural synthesis
                if key == 42:
                    if loudness_bin < 1:
                        loudness_bin += loudness_step
                        client_misc.send_message(oscIdentifier_loudness, [loudness_trans, loudness_bin])

                # key "/" on numpad for decreasing volume for binaural synthesis
                if key == 47:
                    if loudness_bin > 0.05:
                        loudness_bin -= loudness_step
                        client_misc.send_message(oscIdentifier_loudness, [loudness_trans, loudness_bin])


            # adjustment to desired global origin
            current_position_before = (posZ + position_offset) * 100
            yaw += yaw_offset
            posX = (posX + posX_offset) * 100
            current_position_before += (math.cos(math.radians(yaw)) * radius)
            posX += (-1 * math.sin(math.radians(yaw)) * radius)
            pitch += pitch_offset
            roll += roll_offset

            if stop_for_deviation:
                pausePlayback_tracking = (abs(pitch) > max_pitch_difference or abs(roll) > max_roll_difference or current_position_before > 200+max_distance_to_line or current_position_before < -max_distance_to_line or abs(posX) > max_distance_to_line)

            if (pausePlayback_binsim or pausePlayback_tracking) != pausePlayback:
                pausePlayback = (pausePlayback_binsim or pausePlayback_tracking)
                client_misc.send_message(oscIdentifier_playback, [str(pausePlayback)])

            if num_current_setting != num_previous_setting:
                num_previous_setting = copy.deepcopy(num_current_setting)
                current_setting = settings[num_previous_setting]

                audio_index = current_setting["audio"]
                in_channels_trans = current_setting["in_trans"]
                in_channels_bin = current_setting["in_bin"]
                out_channels_trans = current_setting["out_trans"]
                virtual_outchannel_bin = current_setting["out_bin"]
                use_HP = current_setting["HP"]
                loudness_trans = current_setting["loudness_trans"]
                loudness_bin = current_setting["loudness_bin"]

                client_misc.send_message(oscIdentifierSetting, [audio_files[audio_index],
                                                                use_HP,
                                                                loudness_trans,
                                                                loudness_bin,
                                                                in_channels_bin,
                                                                in_channels_trans,
                                                                out_channels_trans])

            # Build and send OSC message
            # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
            yaw = min(yawVectorSubSampled, key=lambda x: abs(x - yaw))
            current_position = int(np.argmin(
                np.array([round(abs(x - current_position_before) / 25) for x in positionVectorSubSampled])) * 25)

            print(
                f"Yaw: {round(yaw)}    "
                f"z: {current_position} ({round(current_position_before)})    "
                f"x: {round(posX)}    "
                f"roll: {round(roll)}    "
                f"pitch: {round(pitch)}    "
                f"vol: ({round(loudness_trans, 2)}|{round(loudness_bin, 2)})    "
                f"Sound: {str(not pausePlayback)[0]} ({str(not pausePlayback_binsim)[0]}/{str(not pausePlayback_tracking)[0]})    "
                f"audio: {audio_index}    "
                f"use HP: {use_HP}    "
                f"setting: {num_current_setting}/{num_settings}"
            )

            # build OSC Message and send it
            for n in range(0, nSources):
                # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
                binsim_ds = [n, int(round(yaw)), 0, 0, current_position, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             virtual_outchannel_bin[n]]
                client_ds.send_message(oscIdentifier_ds, binsim_ds)

            sys.stdout.flush()

            time.sleep(0.01)

    except KeyboardInterrupt:  # Break if ctrl+c is pressed

        # Console output
        print("Done")


if __name__ == "__main__":
    start_tracker()
