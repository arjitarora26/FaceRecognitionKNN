import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import cam
import os

DATASET_PATH = './data/'
CASCADE_CLASSIFIER = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
SKIP_FRAMES = 10
MAX_SAVED_IMAGES_COUNT = 20
CROP_OFFSET = 20

def getBiggestFace(faceList):
	faceList = sorted(faceList,key = lambda f: f[2]*f[3])
	return faceList[-1]

def display(frame,croppedFace):
	cv2.imshow('Frame',frame)
	cv2.imshow('FaceSection',croppedFace)	

def saveImages(name,faceDataset):
	faceDataset = np.asarray(faceDataset)
	#flatten each frame and save
	faceDataset = faceDataset.reshape((faceDataset.shape[0],-1))
	print(faceDataset.shape)
	
	path = DATASET_PATH + name + '.npy'
	confirm = input(f"Save to {path} ? y/n:  ")
	if confirm != 'y':
		print("Data will not be saved..\n")
		return 
	# create path
	os.makedirs(os.path.dirname(path), exist_ok=True)
	np.save(path,faceDataset)
	print("Data saved!!")


def userSession():
	name = input("Enter your name : ")
	webcam = cam.initCamera()
	
	faceDataset = []
	frameCount  = 1
	savedCount  = 0

	print('Press q to quit camera\n')
	while True:
		success,frame = webcam.read()
		#Succesful Capture	
		if not success:
			print("Picture could not be captured.\n")
			time.sleep(2)
			continue

		facesBoundingBoxes = CASCADE_CLASSIFIER.detectMultiScale(frame,1.1,5)
		if len(facesBoundingBoxes)==0:
			continue

		faceParameters = getBiggestFace(facesBoundingBoxes)
		croppedFace    = cam.getCroppedFace(frame,faceParameters)

		cam.drawBoundingBox(frame,faceParameters)
		
		# store every 10th frame

		if frameCount%SKIP_FRAMES==0:
			faceDataset.append(croppedFace)
			savedCount+=1
			print(savedCount)
			if savedCount==MAX_SAVED_IMAGES_COUNT:
				break

		frameCount+=1

		display(frame,croppedFace)

		action = cam.checkAction()
		if action=="quit":
			break
	
	webcam.release()
	cv2.destroyAllWindows()

	print("Please wait as we save your data...")
	saveImages(name,faceDataset)


def main():
	print("Welcome to the face recognition data collector.")
	while True:
		print("1) Collect data of user")
		print("2) Quit ")
		action = input()
		if(action == '1'):
			userSession()
		if(action == '2'):
			confirm = input("Are you sure you want to exit? y/n : ")
			if(confirm=='y'):
				break


if __name__ == '__main__':
	main()

