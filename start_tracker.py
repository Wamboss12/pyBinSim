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
    oscIdentifier = '/pyBinSim'
    ip = '127.0.0.1'
    port = 10000

    # please choose the correct COM-Port
    comPort = 'COM4'
    baudRate = 57600

    nSources = 1
    minAngleDifference = 4
    filterSet = 0

    # Value override by user
    if (len(sys.argv)) > 1:
        for i in range(len(sys.argv)):

            if (sys.argv[i] == '-comPort'):
                comPort = sys.argv[i + 1]

            if (sys.argv[i] == '-port'):
                port = int(sys.argv[i + 1])

            if (sys.argv[i] == '-ip'):
                ip = sys.argv[i + 1]

            if (sys.argv[i] == '-baudRate'):
                baudRate = int(sys.argv[i + 1])

    # Internal settings:
    positionVectorSubSampled = range(0, 360, minAngleDifference)

    print(['IP: ', ip])
    print(['Using Port ', port])

    # Create OSC client
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)

    roll, pitch, yaw = 0, 0, 0

    while 1:

        # define current angles as "zero position"	when spacebar is hit
        if msvcrt.kbhit():
            # char = msvcrt.getch()
            key = ord(msvcrt.getch())

            # up_arrow = 72
            # down_arrow = 80
            # left_arrow = 75
            # right_arrow = 77

            if key == 32:
                roll, pitch, yaw = 0, 0, 0

            # Key 'Up arrow' moves forward to the source, next point nearer to the front source
            if key == 72:
                if filterSet < 8:
                    filterSet += 1
            # Key 'Down arrow' moves
            if key == 80:
                if filterSet > 0:
                    filterSet -= 1
            # Key 'Left arrow' for increasing jaw --> turning to the left
            if key == 75:
                yaw += 20
                if yaw >= 360:
                    yaw -= 360
            # Key 'Right arrow' for decreasing jaw --> turning to the right
            if key == 77:
                yaw -= 20
                if yaw < 0:
                    yaw += 360

        # build OSC Message and send it
        for n in range(0, nSources):
            # channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
            yaw = min(positionVectorSubSampled, key=lambda x: abs(x - yaw))
            binSimParameters = [n, int(round(yaw)), filterSet, 0, 0, 0, 0]
            print(['Source ', n, ' Yaw: ', round(yaw), '  Filterset: ', filterSet])
            client.send_message(oscIdentifier, binSimParameters)

        time.sleep(0.1)


if __name__ == "__main__":
    start_tracker()
