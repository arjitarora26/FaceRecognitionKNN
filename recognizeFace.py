import numpy as np
import cv2
import os
from predict import processDataset,predictKNN
import cam
import time

DATASET_PATH = './data/'
CASCADE_CLASSIFIER = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
CROP_OFFSET = 20



def display(frame):
	cv2.imshow('Frame',frame)


def detectFaces():
	webcam = cam.initCamera()
	print('Press q to quit camera\n')

	while True:
		success,frame = webcam.read()
		#Succesful Capture	
		if not success:
			print("Picture could not be captured.\n")
			time.sleep(2)
			continue

		facesBoundingBoxes = CASCADE_CLASSIFIER.detectMultiScale(frame,1.3,5)
		if len(facesBoundingBoxes)==0:
			continue

		for face in facesBoundingBoxes:
			cam.drawBoundingBox(frame,face)
			name = predictKNN(cam.getCroppedFace(frame,face))
			cam.displayName(frame,name,face)
		
		display(frame)
		action = cam.checkAction()
		if action=="quit":
			break
	
	webcam.release()
	cv2.destroyAllWindows()

	print("Closing..")

def main():
	print("Welcome to the face reocgnition system!!")
	print("Initializing dataset..")
	processDataset(DATASET_PATH)
	detectFaces()


if __name__ == '__main__':
	main()


