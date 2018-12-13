import cv2
import numpy as np

class IdadeGenero:

    def __init__(self):
        self.reconhecedorIdadeGenero()

    def reconhecedorIdadeGenero(self):

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

        gender_list = ['Male', 'Female']

        reconhecedor = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        camera = cv2.VideoCapture(0)
        #camera.set(3, 480)
        #camera.set(4, 640)

        age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt',
                                           'age_net.caffemodel')

        gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt',
                                              'gender_net.caffemodel')
        get = 0
        while True:
            conectado, imagem = camera.read()
            if get%1==0:
                imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
                facesDetectadas = reconhecedor.detectMultiScale(imagemCinza,1.1, 5)
                
            if get%1==0: 
                for(x, y, l, a )in facesDetectadas:
                    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
                    
                    imagemFace = imagem[y:y + a, x:x + l]

                    blob = cv2.dnn.blobFromImage(imagemFace,1 ,(227, 227),
                                                 MODEL_MEAN_VALUES,swapRB=False)

                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    gender = gender_list[gender_preds[0].argmax()]
                    
                    age_net.setInput(blob)
                    age_preds = age_net.forward()
                    age = age_list[age_preds[0].argmax()]
                    textoSobrescrito =  "%s %s" % (gender, age)
                    cv2.putText(imagem, textoSobrescrito, (x,y), font, 2, (0,0,255),2, cv2.LINE_AA)
                get = 0
            else:
                get += 1
            cv2.imshow("Face", imagem)
            if cv2.waitKey(1) == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    IdadeGenero()

#fazer metodo que calcula genero e com outra thread manda para a thread que tem a exibição da imagem
