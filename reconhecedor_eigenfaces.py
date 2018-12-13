import cv2
import numpy as np

#pega todas as imagens de cada pessoa do treinamento e gera uma imagem media
#id vetor proprio, eigen vector

#cria um vetor com cada pixel de cada imagem cada pessoa
#cria varios vetores, cada um de cada imagem
#PCA (Principal Component Analysis) seleciona principais caracteristicas desses vetores
#(remove informação inutil, reduz a dimensionalidade)

#Apos esses passos, ele extrai a imagem media
#Quanto as pessoas variam da media
#Mostra os desvios(variações)

#Depois gera os eigen vectors, eigen faces
#Na pratica, ele extrai varios eigen faces para cada pessoas
#Resumindo, utilizando essas eigen faces, voce chega nas faces originais do treinamento

#Projeta a imagem nova(webcam) no espaço de faces
#Extrai componentes eigenfaces da imagem
#Calcula a distancia entra a nova(webcam)
#Quem tiver a menor distancia é a que ele relaciona.
#Algoritmo(KNN)
#Busca nas imagens de treinamento pela mais semelhante


class Reconhecedor_eigenfaces:

    def __init__(self, listaNome):
        self.listaNome = listaNome
        self.reconhecimento_eigenfaces()

    def reconhecimento_eigenfaces(self):

        detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
        reconhecedor = cv2.face.EigenFaceRecognizer_create()
        reconhecedor.read("classificadorEigen.yml")
        largura, altura = 220, 220
        camera = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        while (True):
            conectado, imagem = camera.read()
            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                            scaleFactor=1.5,
                                                            minSize=(30,30))

            for(x, y, l, a) in facesDetectadas:
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

                id, predicao = reconhecedor.predict(imagemFace)

                pessoa = ""
                if predicao > 8000:
                    pessoa = 'Desconhecido'
                else: # np.average(imagemCinza)>100:
                    pessoa = self.listaNome.get( id , "Desconhecido")

                print(pessoa)
                cv2.putText(imagem, pessoa, (x,y +(a+30)), font, 2, (0,0,255))
                cv2.putText(imagem, str(predicao), (x,y + (a+50)), font, 1, (0,0,255))

            cv2.imshow("Face", imagem)
            
            if cv2.waitKey(1) == ord('p'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Reconhecedor_eigenfaces()

