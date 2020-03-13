import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


CROP_OFFSET = 20


def checkAction():
	#quit camera when q is pressed
	keyPressed = cv2.waitKey(1) & 0xFF

	if keyPressed == ord('q'):
		print("Webcam is going to stop gathering frames now")
		return "quit"
	return "No"

def drawBoundingBox(frame,faceParameters):
	x,y,w,h = faceParameters
	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

def displayName(frame,name,faceParameters):
	x,y,_,_ = faceParameters
	cv2.putText(frame,name,(x,y-15),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2,cv2.LINE_AA)

def getCroppedFace(frame,faceParameters):
	x,y,w,h = faceParameters
	croppedFace = frame[ y-CROP_OFFSET:y+h+CROP_OFFSET, x-CROP_OFFSET:x+w+CROP_OFFSET ]
	croppedFace = cv2.resize(croppedFace,(100,100))
	return croppedFace

def initCamera():
	webcam = cv2.VideoCapture(0)
	print("Initializing cam ..\n")
	return webcam

