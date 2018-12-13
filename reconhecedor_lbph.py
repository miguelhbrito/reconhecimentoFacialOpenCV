import cv2
import numpy as np
import pickle

#Local Binary Patterns Histograms
#Pega o elemento central e compara com os vizinhos
#Se >= numero central , entao 1
#se < numero central , entao 0
#Pega esses numeros e gera um binario

#Trabalha melhor com iluminação

#O numero binario é usado pra treinar o sistema,
#gerando um histograma dos valores.
#Encontrar a estrutura local da imagem por meio dos vizinhos.
#Gera o histograma para a imagem nova(webcam) e compara com os
#histogramas da base de dados

#Parametros é muito importante:
#radius = raio maior aumenta a abrangencia mas pode perder bordas finas,
#quanto maior o raio, mais padoroes podem ser codificados, mas gasta mais processamento
#neighbors = numero de pontos da amostra para construir um padrao local,
#quanto maior o numero de vizinhos, maior processamento
#grid x = numero de quadrados horizontal
#grid y = numero de quadrados na vertical
#threshhold = limite de confiança

class Reconhecedor_lbph:

    def __init__(self, listaNome):
        self.listaNome = listaNome
        self.reconhecimento()

    def reconhecimento(self):
        
        detectorFace = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
        reconhecedor = cv2.face.LBPHFaceRecognizer_create()
        reconhecedor.read("classificadorLBPH.yml")
        largura, altura = 220, 220 
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        camera = cv2.VideoCapture(0)

        while (True):
            conectado, imagem = camera.read()
            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
            facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                            scaleFactor=1.5,
                                                            minSize=(100,100),
                                                            maxSize=(500,500))

            for(x, y, l, a) in facesDetectadas: # x , y, largura e algura
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura)) #imagem convertida para 220, 220
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
                id, predicao = reconhecedor.predict(imagemFace)
                ##np.average é a luminosidade da foto, valor vai de 0 a 255

                pessoa = ""
                if predicao > 60:
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
    Reconhecedor_lbph()
