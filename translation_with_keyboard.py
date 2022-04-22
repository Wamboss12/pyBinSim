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

from pybinsim.spark_fun import Spark9dof


def start_tracker():
    # Default values
    oscIdentifier_ds = '/pyBinSim_ds_Filter'
    #oscIdentifier_early = '/pyBinSim_early_Filter'
    #oscIdentifier_late = '/pyBinSim_late_Filter'
    oscIdentifier_loudness = '/pyBinSimLoudness'
    oscIdentifier_convolution = '/pyBinSimPauseConvolution'
    oscIdentifier_playback = '/pyBinSimPauseAudioPlayback'
    oscIdentifier_soundfile = '/pyBinSimFile'
    oscIdentifier_mapping = '/pyBinSimMapping'
    oscIdentifier_headphonefilter = '/pyBinSimHP'

    ip = '127.0.0.1'
    port_ds = 10000
    port_early = 10001
    port_late = 10002
    port_misc = 10003

    # please choose the correct COM-Port
    comPort = 'COM4'
    baudRate = 57600

    nSources = 2
    minAngleDifference = 4
    minxDifference = 25

    # Internal settings:
    yawVectorSubSampled = range(0, 360, minAngleDifference)
    xVectorSubSampled = range(0, 201, minxDifference)

    print(['IP: ', ip])
    print(['Using Ports ', port_ds, port_early, port_late, port_misc])

    # Create OSC client
    client_ds = udp_client.SimpleUDPClient(ip, port_ds)
    #client_early = udp_client.SimpleUDPClient(ip, port_early)
    #client_late = udp_client.SimpleUDPClient(ip, port_late)
    client_misc = udp_client.SimpleUDPClient(ip, port_misc)

    yaw = 0
    current_x = 0
    loudness = 0.05
    pauseConvolution = False
    pausePlayback_binsim = True
    pausePlayback_tracking = False
    pausePlayback = True
    audio_files = ["signals/noise_noise_silence.wav",
                   "signals/speech_noise_silence.wav",
                   "signals/speech_speech_silence.wav",
                   "signals/guitar_voice_silence.wav"]
    audio_index = 0
    virtual_outchannel_bin = [0, 1] # erste Quelle auf LS 1
    out_channels_trans = [1, 0] # erste Quelle auf LS 2

    in_channels_trans = [1, 2]
    in_channels_bin = [0, 3]
    in_channels_trans_before = []
    in_channels_bin_before = []
    save_in_channels = []
    state = "normal"
    use_HP = False

    while 1:

        # define current angles as "zero position"	when spacebar is hit
        if msvcrt.kbhit():
            # char = msvcrt.getch()
            key = ord(msvcrt.getch())

            # up_arrow = 72
            # down_arrow = 80
            # left_arrow = 75
            # right_arrow = 77

            # "Space"
            if key == 32:
                yaw, current_x = 0, 0

            # Key 'Up arrow' moves forward to the source, next point nearer to the front source
            if key == 72:
                if current_x < 200:
                    current_x += 25

            # Key 'Down arrow' moves
            if key == 80:
                if current_x > 0:
                    current_x -= 25

            # Key 'Left arrow' for increasing yaw --> turning to the left
            if key == 75:
                yaw += 30
                if yaw >= 360:
                    yaw -= 360

            # Key 'Right arrow' for decreasing yaw --> turning to the right
            if key == 77:
                yaw -= 30
                if yaw < 0:
                    yaw += 360


            # key "+" on numpad for increasing volume
            if key == 43:
                if loudness < 1:
                    loudness += 0.05
                    client_misc.send_message(oscIdentifier_loudness, [loudness])

            # key "-" on numpad for increasing volume
            if key == 45:
                if loudness > 0.05:
                    loudness -= 0.05
                    client_misc.send_message(oscIdentifier_loudness, [loudness])

            # key "*" on numpad for increasing volume for binaural synthesis
            if key == 42:
                if loudness < 1:
                    loudness += 0.05
                    client_misc.send_message(oscIdentifier_loudness, [loudness])

            # key "/" on numpad for decreasing volume for binaural synthesis
            if key == 47:
                if loudness > 0.05:
                    loudness -= 0.05
                    client_misc.send_message(oscIdentifier_loudness, [loudness])

            # key "c" for pausing the convolution
            if key == 99:
                pauseConvolution = not pauseConvolution
                client_misc.send_message(oscIdentifier_convolution, [str(pauseConvolution)])

            # key "p" for pausing the convolution
            if key == 112:
                pausePlayback_binsim = not pausePlayback_binsim


            # keys "1", "2", "3", "4" for choosing the audio
            if key in [49, 50, 51, 52]:
                audio_index = key - 49
                client_misc.send_message(oscIdentifier_soundfile, [audio_files[audio_index]])

            # key "r" to switch the source position of real and virtual sources
            if key == 114:
                out_channels_trans = out_channels_trans[::-1]
                virtual_outchannel_bin = virtual_outchannel_bin[::-1]

                client_misc.send_message(oscIdentifier_mapping, [in_channels_bin, in_channels_trans, out_channels_trans])

            # key "t" to switch real and virtual source
            if key == 116:
                out_channels_trans = out_channels_trans[::-1]
                virtual_outchannel_bin = virtual_outchannel_bin[::-1]

                save_in_channels = in_channels_bin
                in_channels_bin = in_channels_trans
                in_channels_trans = save_in_channels

                if state == "only real":
                    state = "only virtual"
                elif state == "only virtual":
                    state = "only real"

                client_misc.send_message(oscIdentifier_mapping, [in_channels_bin, in_channels_trans, out_channels_trans])
            # key "a" to have a real and a virtual source
            if key == 97:

                if state != "normal":
                    in_channels_bin = in_channels_bin_before.copy()
                    in_channels_trans = in_channels_trans_before.copy()
                    state = "normal"

                client_misc.send_message(oscIdentifier_mapping,
                                         [in_channels_bin, in_channels_trans, out_channels_trans])
            # key "s" to have only real sources
            if key == 115:

                if state == "normal":
                    in_channels_bin_before = in_channels_bin.copy()
                    in_channels_trans_before = in_channels_trans.copy()
                    state = "only real"
                elif state == "only virtual":
                    state = "only real"

                in_channels_bin = [2, 3]
                in_channels_trans = out_channels_trans.copy()

                client_misc.send_message(oscIdentifier_mapping,
                                         [in_channels_bin, in_channels_trans, out_channels_trans])
            # key "d" to have only virtual sources
            if key == 100:

                if state == "normal":
                    in_channels_bin_before = in_channels_bin.copy()
                    in_channels_trans_before = in_channels_trans.copy()
                    state = "only virtual"
                elif state == "only real":
                    state = "only virtual"

                in_channels_bin = virtual_outchannel_bin.copy()
                in_channels_trans = [2, 3]

                client_misc.send_message(oscIdentifier_mapping,
                                         [in_channels_bin, in_channels_trans, out_channels_trans])

            # key "s" to have only real sources
            if key == 104:
                use_HP = not use_HP

                client_misc.send_message(oscIdentifier_headphonefilter, [use_HP])

        if (pausePlayback_binsim or pausePlayback_tracking) != pausePlayback:
            pausePlayback = (pausePlayback_binsim or pausePlayback_tracking)
            client_misc.send_message(oscIdentifier_playback, [str(pausePlayback)])

        yaw = min(yawVectorSubSampled, key=lambda x: abs(x - yaw))
        current_x = min(xVectorSubSampled, key=lambda x: abs(x-current_x))
        #print(['Yaw:', round(yaw), 'x:', current_x, 'loudness:', round(loudness, 1), "Convol:", not(pauseConvolution),
        #       'Playback:', not(pausePlayback), 'audio:', audio_index, 'state:', state, 'use HP:', use_HP])

        print(
            f"Yaw: {round(yaw)}    x: {current_x} ({round(current_x, 2)})    loudness: {round(loudness, 2)}    "
            f"Playback: {pausePlayback} ({pausePlayback_binsim}/{pausePlayback_tracking})    audio: {audio_index + 1}    use HP: {use_HP}    "
            f"state: {state}"
        )

        # build OSC Message and send it
        for n in range(0, nSources):
            # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
            binsim_ds =     [n,  int(round(yaw)), 0, 0,  current_x, 0, 0,    0, 0, 0,   0, 0, 0,  0, 0, virtual_outchannel_bin[n]]
            #binsim_early =  [n,  int(round(yaw)), 0, 0,  current_x, 0, 0,    0, 0, 0,   0, 0, 0,  0, 0, 0]
            #binsim_late =   [n,  0, 0, 0,                0, 0, 0,            0, 0, 0,   0, 0, 0,  0, 0, 0]
            client_ds.send_message(oscIdentifier_ds, binsim_ds)
            #client_early.send_message(oscIdentifier_early, binsim_early)
            #client_late.send_message(oscIdentifier_late, binsim_late)

        time.sleep(0.1)


if __name__ == "__main__":
    start_tracker()
