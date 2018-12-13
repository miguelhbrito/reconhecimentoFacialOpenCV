from captura import Captura
from reconhecedor_lbph import Reconhecedor_lbph
from reconhecedor_eigenfaces import Reconhecedor_eigenfaces
from reconhecedor_fisherfaces import Reconhecedor_fisherfaces
from idadeGenero import IdadeGenero
import pickle
import os

#listaNome[1] = miguel
#listaNome[1]
#listaNome.get(id, 'Nao encontrado')

if not os.path.isfile('listaDeNomes'):
    with open('listaDeNomes','wb'):
        pass
        
with open("listaDeNomes", "rb") as arquivo:
    f = arquivo.read()
if len(f) > 0:
    listaNome = pickle.loads(f)
else:
    listaNome = {}

while True:
    
    os.system("clear")
    print("1 - Tirar fotos de um novo usuário")
    print("2 - Iniciar Reconhecimento Eigenfaces")
    print("3 - Iniciar Reconhecimento Fisherfaces")
    print("4 - Iniciar Reconhecimento LBPH")
    print("5 - Iniciar idade e genero")
    print("0 - Sair")

    o = input("Opção: ")

    if o == '0':
        break
    
    elif o == '1':
        Captura(listaNome)
        print(listaNome)

    elif o == '2':
        Reconhecedor_eigenfaces(listaNome)

    elif o == '3':
        Reconhecedor_fisherfaces(listaNome)

    elif o == '4':
        print(listaNome)
        Reconhecedor_lbph(listaNome)

    elif o == '5':
        IdadeGenero()
    
