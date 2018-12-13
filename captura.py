from treinamento import Treinamento
import cv2
import numpy as np
import pickle

#pickle.dumps(listaNomes)  seriable
#salvar como txt ou binario

#abrir o arquivo antes
#pickle.loads(file.read())
#dicionario = pickle.loads(file.read())

class Captura:

    def __init__(self, listaNome):
        self.listaNome = listaNome
        self.tirarFoto()

    def tirarFoto(self):
        
        id = int(input('Digite um identificador: '))

        nome = input('Digite o nome da pessoa: ')

        self.listaNome[id] = nome
        
        print("Recomendacoes para tirar as fotos: " +
              "\n-Variações na expressão(feliz, triste, com e sem oculos)" +
              "\n-Variações no angulo(olhando levemente para cima, baixo, esquerda, direita)" +
              "\n-Ambiente bem iluminado e luz incindindo no rosto")
        print("Capturando as faces!"+
              "\nOrientações: "+
              "\nPressione a tecla q para tirar cada uma das fotos"+
              "\nNo total são 25 fotos.")
        
        classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
        classificadorOlho = cv2.CascadeClassifier("haarcascade-eye.xml")
        camera = cv2.VideoCapture(0) ##aciona a camera do notebook

        amostra = 1
        numeroAmostras = 25
        largura, algura = 220, 220

        olhosDetectados = []

        while (True):
            conectado, imagem = camera.read()
            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)##transforma a imagem colorida em cinza
            ##print(np.average(imagemCinza)) ##para testar a luminosidade da imagem
            facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                             scaleFactor = 1.5,
                                                             minSize= (100,100)) ##detecta a face os atributos podem ser configurados

            for(x, y, l, a)in facesDetectadas:
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)##desenha o retangulo
                regiao = imagem[y:y + a, x:x + l]
                regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)##transforma a imagem colorida em cinza
                olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)##detecta os olhos

            for(ox, oy, ol, oa) in olhosDetectados:
                cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)##desenha o retangulo nos olhos

        
                if cv2.waitKey(1) & 0xFF == ord('f'):
                    ##np.average é a luminosidade da foto, valor vai de 0 a 255
                   # if np.average(imagemCinza) > 110 : 
                        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, algura))##pega somente a parte do retangulo para salvar
                        cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)##salva as fotos
                        print("[foto " + str(amostra) + " capturada com sucesso]")
                        amostra += 1
                    #else :
                      #  print('Luminosidade abaixo da esperada')  
            
            cv2.imshow("Face", imagem)
            cv2.waitKey(1)
            if(amostra >= numeroAmostras + 1):
                with open("listaDeNomes","wb") as arquivo:
                    arquivo.write(pickle.dumps(self.listaNome))
                Treinamento()
                break
            if cv2.waitKey(1) and 0xFF == ord('k'):
                print("Saindo da captura !!!")
                break
           
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Captura()
