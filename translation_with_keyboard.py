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
    oscIdentifier_early = '/pyBinSim_early_Filter'
    oscIdentifier_late = '/pyBinSim_late_Filter'
    oscIdentifier_loudness = '/pyBinSimLoudness'
    oscIdentifier_convolution = '/pyBinSimPauseConvolution'
    oscIdentifier_playback = '/pyBinSimPauseAudioPlayback'

    ip = '127.0.0.1'
    port_ds = 10000
    port_early = 10001
    port_late = 10002
    port_misc = 10003

    # please choose the correct COM-Port
    comPort = 'COM4'
    baudRate = 57600

    nSources = 1
    minAngleDifference = 4
    minxDifference = 25

    # Internal settings:
    yawVectorSubSampled = range(0, 360, minAngleDifference)
    xVectorSubSampled = range(0, 201, minxDifference)

    print(['IP: ', ip])
    print(['Using Ports ', port_ds, port_early, port_late, port_misc])

    # Create OSC client
    client_ds = udp_client.SimpleUDPClient(ip, port_ds)
    client_early = udp_client.SimpleUDPClient(ip, port_early)
    client_late = udp_client.SimpleUDPClient(ip, port_late)
    client_misc = udp_client.SimpleUDPClient(ip, port_misc)

    yaw = 0
    current_x = 0
    loudness = 0.1
    pauseConvolution = False
    pausePlayback = False

    while 1:

        # define current angles as "zero position"	when spacebar is hit
        if msvcrt.kbhit():
            # char = msvcrt.getch()
            key = ord(msvcrt.getch())

            # up_arrow = 72
            # down_arrow = 80
            # left_arrow = 75
            # right_arrow = 77

            "Space"
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
                    loudness += 0.1
                    client_misc.send_message(oscIdentifier_loudness, [loudness])
            # key "-" on numpad for increasing volume
            if key == 45:
                if loudness > 0.1:
                    loudness -= 0.1
                    client_misc.send_message(oscIdentifier_loudness, [loudness])
            # key "c" for pausing the convolution
            if key == 99:
                pauseConvolution = not pauseConvolution
                client_misc.send_message(oscIdentifier_convolution, [str(pauseConvolution)])
            # key "p" for pausing the convolution
            if key == 112:
                pausePlayback = not pausePlayback
                client_misc.send_message(oscIdentifier_playback, [str(pausePlayback)])

        yaw = min(yawVectorSubSampled, key=lambda x: abs(x - yaw))
        current_x = min(xVectorSubSampled, key=lambda x: abs(x-current_x))
        print(['Yaw: ', round(yaw), '  x: ', current_x, 'loudness: ', round(loudness, 1), "pause Convol: ", pauseConvolution,
               'pause Playback: ', pausePlayback])

        #Value 1 - 3: listener orientation[yaw, pitch, roll]
        #Value 4 - 6: listener position[x, y, z]
        #Value 7 - 9: custom values[a, b, c]

        # build OSC Message and send it
        for n in range(0, nSources):
            # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
            binsim_ds =     [n,  int(round(yaw)), 0, 0,  current_x, 0, 0,    0, 0, 0,   0, 0, 0,  0, 0, 0]
            binsim_early =  [n,  int(round(yaw)), 0, 0,  current_x, 0, 0,    0, 0, 0,   0, 0, 0,  0, 0, 0]
            binsim_late =   [n,  0, 0, 0,                0, 0, 0,            0, 0, 0,   0, 0, 0,  0, 0, 0]
            client_ds.send_message(oscIdentifier_ds, binsim_ds)
            client_early.send_message(oscIdentifier_early, binsim_early)
            client_late.send_message(oscIdentifier_late, binsim_late)

        time.sleep(0.1)


if __name__ == "__main__":
    start_tracker()
