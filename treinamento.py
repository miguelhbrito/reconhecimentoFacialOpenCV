import cv2
import os
import numpy as np


#O eigenface olha para toas as imagens de toas as pessoas de uma vez
#e tenta encontrar componentes de todas elas combinadas
#nao foca nas caracteristicas que distiguem um individuo do outro.
#Ele utiliza a distancia euclidiana para fazer o reconhecimento.
#->quanto maior numero de componentes melhor o reconhecimento
#->quanto menor o numero do threshold mais as faces tem que ser iguais
#as faces dos dados, entao quanto menor melhor.
#Usa o PCA
#eigenface = cv2.face.EigenFaceRecognizer_create(num_components=10, threshold=2)
#Fisherfaces usa o LDA(Linear Discriminant Analysis)

class Treinamento:

    def __init__(self):
        eigenface = cv2.face.EigenFaceRecognizer_create()
        fisherface = cv2.face.FisherFaceRecognizer_create()
        lbph = cv2.face.LBPHFaceRecognizer_create()
        self.salvando(eigenface, fisherface, lbph)


    def pegaImagens(self):
        caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
        print(caminhos)
        faces = []
        ids = []

        for caminhoImagem in caminhos:
            imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
            id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
            ids.append(id)
            faces.append(imagemFace)
            cv2.imshow("Face", imagemFace)
            cv2.waitKey(10)
        return np.array(ids), faces

    def salvando(self, eigenface, fisherface, lbph):
        ids, faces = self.pegaImagens()
        #print(ids)

        print("Treinando !!!")
        #cria os documentos de aprendizagem de cada face com o eigenface.
        eigenface.train(faces, ids)
        eigenface.write('classificadorEigen.yml')

        #cria os documentos de aprendizagem de cada face com o eigenface.
        fisherface.train(faces, ids)
        fisherface.write('classificadorFisher.yml')

        #cria os documentos de aprendizagem de cada face com o lbph.
        lbph.train(faces, ids)
        lbph.write('classificadorLBPH.yml')

        print("Treinamento realizado")

if __name__ == '__main__':
    Treinamento()
