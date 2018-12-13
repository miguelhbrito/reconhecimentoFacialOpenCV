import cv2
import numpy as np

#Eigenfaces olha para todas as imagen de todas as pessoas de uma vez
#e tenta encontrar componentes de todas elas combinadas
#O eigen faces nao foca nas caracteristicas que distinguem um indivio do outro
#(face das pessoas como um todo)

#Fisherfaces utiliza LDA(Linear Discriminant Analysis) que tambem reduz as dimensoes
#LDA não esta interessado na maior variação, mas em maximar a separação entre as classes
#Extrai caracteristicas separadamente
#LDA gera as faces

class Reconhecedor_fisherfaces:

    def __init__(self, listaNome):
        self.listaNome = listaNome
        self.reconhecimento_fisherfaces()

    def reconhecimento_fisherfaces(self):

        detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
        reconhecedor = cv2.face.FisherFaceRecognizer_create()
        reconhecedor.read("classificadorFisher.yml")
        largura, altura = 220, 220
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        camera = cv2.VideoCapture(0)

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
                ##np.average é a luminosidade da foto, valor vai de 0 a 255
                pessoa = ""
                
                if predicao > 550:
                    pessoa = 'Desconhecido'
                else: # np.average(imagemCinza)>100:
                    pessoa = self.listaNome.get( id , "Desconhecido")

                cv2.putText(imagem, pessoa, (x,y +(a+30)), font, 2, (0,0,255))
                cv2.putText(imagem, str(predicao), (x,y + (a+50)), font, 1, (0,0,255))

            cv2.imshow("Face", imagem)
            if cv2.waitKey(1) == ord('p'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Reconhecedor_fisherfaces()
