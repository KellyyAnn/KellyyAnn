import cv2
import os
import numpy as np

Info="C:/Users/MINEDUCYT/Proyecto/Data/"
peopleList= os.listdir(Info)
print("Lista de personas: ", peopleList)

labels=[]
facesData=[]
label=0

for nameDir in peopleList:
	personPath=Info+"/"+ nameDir
	print("Leyendo las imágenes")

	for fileName in os.listdir(personPath):
		print("Rostros: ", nameDir + "/" + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+ "/"+fileName,0))
		image= cv2.imread(personPath+ "/"+fileName,0)
		#cv2.imshow("image", image)
		#cv2.waityKey(10)
label= label + 1          

#print("labels= ", labels)
#print("Numero de etiquetas 0: ", np.count_nonzero(np.array(labels)==0))
#print("Numero de etiquetas 1: ", np.count_nonzero(np.array(labels)==1))

reconocedor = cv2.face.LBPHFaceRecognizer_create()
#reconocedor= cv2.face.EigenFaceRecognizer_create()
 
#Entrenando al reconocedor de rostros

print("Entrenando...")
reconocedor.train(facesData, np.array(labels))

#Almacenando al modelo obtenido
#El reconocedor de rostros puede ser XML o YAML
reconocedor.write("ModeloRostros.xml")
print("Modelo almacenado...")