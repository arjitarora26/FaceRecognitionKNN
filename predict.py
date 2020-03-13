import os
import numpy as np
import cam
faceData = []
faceLabels = []

train = []
names = {}


def distance(v1,v2):
	return np.sqrt(sum((v1-v2)**2))


def processDataset(path):
	global faceData,faceLabels,names,train
	classId = 0

	for filename in os.listdir(path):
		classId+=1
		names[classId] = filename[:-4]
		if filename.endswith('.npy'):
			data = np.load(path+filename)
			faceData.append(data)

			target = classId*(np.ones(data.shape[0],))
			faceLabels.append(target)
	faceLabels = np.concatenate(faceLabels,axis=0).reshape(-1,1)
	faceData = np.concatenate(faceData,axis=0)
	print(faceLabels.shape)
	print(faceData.shape)
	train = np.concatenate([faceData,faceLabels],axis = 1)

def predictKNN(test,k=3):
	test = test.reshape(-1,)
	dist = []
	for i in range(train.shape[0]):
		x = train[i,:-1]
		y = train[i,-1]
		d = distance(x,test)
		dist.append([d,y])
	dist = sorted(dist,key = lambda x:x[0])
	nearestK = dist[:k]
	nearestK = np.array(nearestK)
	kLabels = nearestK[:,-1]
	output = np.unique(kLabels,return_counts =True)
	index = np.argmax(output[1])
	faceClassId = int(output[0][index]) 
	return names[faceClassId]

