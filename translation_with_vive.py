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


def matrixFromPose(pose):
	v = pose.mDeviceToAbsoluteTracking

	pose_matrix = np.matrix(
		[[v[0][0], v[0][1], v[0][2], v[0][3]], [v[1][0], v[1][1], v[1][2], v[1][3]],
		 [v[2][0], v[2][1], v[2][2], v[2][3]]])

	return pose_matrix


def rotatePose(pose):
	# Drehwinkel
	alpha = np.radians(90)

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
	# Default orientation values
	rollOffset = 0
	pitchOffset = 0
	yawOffset = 0
	roll = 0
	pitch = 0
	yaw = 0
	Filterset = 0

	oscIdentifier_ds = '/pyBinSim_ds_Filter'
	oscIdentifier_early = '/pyBinSim_early_Filter'
	oscIdentifier_late = '/pyBinSim_late_Filter'
	oscIdentifier_loudness = '/pyBinSimLoudness'
	oscIdentifier_convolution = '/pyBinSimPauseConvolution'
	oscIdentifier_playback = '/pyBinSimPauseAudioPlayback'

	ip = '127.0.0.1'
	port_ds = 10000
	#port_early = 10001
	#port_late = 10002
	port_misc = 10003

	nSources = 1
	minAngleDifference = 4
	minxDifference = 25

	# Internal settings:
	yawVectorSubSampled = range(0, 360, minAngleDifference)
	positionVectorSubSampled = range(0, 201, minxDifference)

	# Create OSC client
	client_ds = udp_client.SimpleUDPClient(ip, port_ds)
	#client_early = udp_client.SimpleUDPClient(ip, port_early)
	#client_late = udp_client.SimpleUDPClient(ip, port_late)
	client_misc = udp_client.SimpleUDPClient(ip, port_misc)

	# init openvr for HTC Vive
	help(openvr.VRSystem)
	vr = openvr.init(openvr.VRApplication_Scene)

	# poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
	# poses = poses_t()
	poses = openvr.VRSystem().getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0,
															  openvr.k_unMaxTrackedDeviceCount)

	# find tracker
	trackernr = 4
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


	loudness = 0.1
	pauseConvolution = False
	pausePlayback = False

	yaw_offset = 0
	position_offset = 0

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

			yaw = yaw + 0
			if yaw < 0:
				yaw = 360 + yaw

			#pitch = pitch + 0
			#if pitch < 0:
			#	pitch = 360 + pitch

			#roll = roll + 0
			#if roll < 0:
			#	roll = 360 + roll

			# key "+" on numpad for increasing volume
			if msvcrt.kbhit():
				# char = msvcrt.getch()
				key = ord(msvcrt.getch())
				if key == 43:
					if loudness < 0.5:
						loudness += 0.05
						client_misc.send_message(oscIdentifier_loudness, [loudness])
				# key "-" on numpad for increasing volume
				if key == 45:
					if loudness > 0.05:
						loudness -= 0.05
						client_misc.send_message(oscIdentifier_loudness, [loudness])
				# key "c" for pausing the convolution
				if key == 99:
					pauseConvolution = not pauseConvolution
					client_misc.send_message(oscIdentifier_convolution, [str(pauseConvolution)])
				# key "p" for pausing the convolution
				if key == 112:
					pausePlayback = not pausePlayback
					client_misc.send_message(oscIdentifier_playback, [str(pausePlayback)])
				# 'Space' for setting the offset so that the current position and orientation are azimut = 0 and
				# position = 0
				if key == 32:
					position_offset = -1 * posZ
					yaw_offset = -1 * yaw
			# print(['YAW: ',yaw,' PITCH: ',pitch,'ROLL: ',roll])
			# print(['X: ',round(posX,2),' Y: ',round(posY,2),'Z: ',round(posZ,2)])

			# adjustment to desired global origin
			current_position_before = (posZ + position_offset) * 100
			yaw += yaw_offset

			if yaw < 0:
				yaw = 360 + yaw

			# yaw = 360 - yaw

			# Build and send OSC message
			# channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
			yaw = min(yawVectorSubSampled, key=lambda x: abs(x - yaw))
			current_position = int(np.argmin(np.array([round(abs(x - current_position_before)/25) for x in positionVectorSubSampled])) * 25)

			print(['Yaw: ', round(yaw), '  x: ', current_position, f'({round(current_position_before, 2)})', 'loudness: ', round(loudness, 2), "pause Convol: ",
				   pauseConvolution, 'pause Playback: ', pausePlayback])
			# build OSC Message and send it
			for n in range(0, nSources):
				# channel valueOne valueTwo ValueThree valueFour valueFive ValueSix
				binsim_ds = [n, int(round(yaw)), 0, 0, current_position, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				#binsim_early = [n, int(round(yaw)), 0, 0, current_position, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				#binsim_late = [n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				client_ds.send_message(oscIdentifier_ds, binsim_ds)
				#client_early.send_message(oscIdentifier_early, binsim_early)
				#client_late.send_message(oscIdentifier_late, binsim_late)

			sys.stdout.flush()

			time.sleep(0.02)

	except KeyboardInterrupt:  # Break if ctrl+c is pressed

		# Console output
		print("Done")


if __name__ == "__main__":
	start_tracker()
