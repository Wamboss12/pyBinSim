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

import argparse
import sys
import time
import msvcrt

from pythonosc import udp_client
import copy


def start_tracker():
    # Default values
    oscIdentifier_ds = '/pyBinSim_ds_Filter'
    oscIdentifier_playback = '/pyBinSimPauseAudioPlayback'
    oscIdentifierSetting = "/pyBinSimSetting"
    oscIdentifier_loudness = '/pyBinSimLoudness'

    ip = '127.0.0.1'
    port_ds = 10000
    port_misc = 10003

    nSources = 2
    minAngleDifference = 4
    minxDifference = 25

    # Internal settings:
    yawVectorSubSampled = range(0, 360, minAngleDifference)
    xVectorSubSampled = range(0, 201, minxDifference)

    # Create OSC client
    client_ds = udp_client.SimpleUDPClient(ip, port_ds)
    client_misc = udp_client.SimpleUDPClient(ip, port_misc)

    yaw = 0
    current_x = 0
    loudness_trans = 0.08
    loudness_bin = 0.05
    loudness_step = 0.01
    pausePlayback_binsim = True
    pausePlayback_tracking = False
    pausePlayback = True
    audio_files = ["signals/noise_noise_silence.wav",
                   "signals/speech_noise_silence.wav",
                   "signals/speech_speech_silence.wav",
                   "signals/guitar_voice_silence.wav",
                   "signals/noise_silence.wav",
                   "signals/speech_silence.wav",]
    audio_index = 0
    virtual_outchannel_bin = [0, 1]  # erste Quelle auf LS 1
    use_HP = False
    settings = {
        1: {"audio": 2, "in_trans": [0, 2], "in_bin": [1, 3], "out_trans": [1, 0], "out_bin": [0, 1], "HP": False,
            "loudness_trans": 0.08, "loudness_bin": 0.05},
        2: {"audio": 2, "in_trans": [0, 2], "in_bin": [1, 3], "out_trans": [0, 1], "out_bin": [1, 0], "HP": False,
            "loudness_trans": 0.08, "loudness_bin": 0.05},
        3: {"audio": 5, "in_trans": [0, 1], "in_bin": [2, 3], "out_trans": [1, 0], "out_bin": [0, 1], "HP": True,
            "loudness_trans": 0.1, "loudness_bin": 0.2}
        }
    num_settings = len(settings.keys())
    num_previous_setting = 0
    num_current_setting = 1
    current_setting = settings[num_current_setting]

    while 1:

        # define current angles as "zero position"	when spacebar is hit
        if msvcrt.kbhit():
            # char = msvcrt.getch()
            key = ord(msvcrt.getch())

            # space = 32

            # up_arrow = 72
            # down_arrow = 80
            # left_arrow = 75
            # right_arrow = 77

            # + = 43
            # - = 45
            # * = 42
            # / = 47

            # c = 99
            # p = 112
            # r = 114
            # t = 116
            # a = 97
            # s = 115
            # d = 100
            # h = 104

            # 1,2,3,4 = 49,50,51,52

            # "up_arrow"
            if key == 72:
                if pausePlayback_binsim and num_current_setting < num_settings:
                    num_current_setting += 1

            # "down_arrow"
            if key == 80:
                if pausePlayback_binsim and num_current_setting > 1:
                    num_current_setting -= 1

            # "Space"
            if key == 32:
                yaw, current_x = 0, 0

            # key "p" for pausing the convolution
            if key == 112:
                pausePlayback_binsim = not pausePlayback_binsim

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

        yaw = min(yawVectorSubSampled, key=lambda x: abs(x - yaw))
        current_x = min(xVectorSubSampled, key=lambda x: abs(x - current_x))

        print(
            f"Yaw: {round(yaw)}    x: {current_x} ({round(current_x, 2)})    loudness: {round(loudness_trans, 2)}|{round(loudness_bin, 2)}    "
            f"Playback: {not pausePlayback} ({not pausePlayback_binsim}/{not pausePlayback_tracking})    audio: {audio_index + 1}    use HP: {use_HP}    "
            f"setting: {num_current_setting}/{num_settings}"
        )

        # build OSC Message and send it
        for n in range(0, nSources):
            # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
            binsim_ds = [n, int(round(yaw)), 0, 0, current_x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, virtual_outchannel_bin[n]]
            client_ds.send_message(oscIdentifier_ds, binsim_ds)

        time.sleep(0.1)


if __name__ == "__main__":
    start_tracker()
