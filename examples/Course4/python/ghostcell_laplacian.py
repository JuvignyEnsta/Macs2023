from PIL import Image  
import numpy as np
import sys
from mpi4py import MPI

DEBUG=False

def laplacianFilter( image):
    """
    Philtre laplacien qui ne modifie que les cellules internes
    """
    extent = image.shape
    lapl = np.empty((extent[0]-2,extent[1]-2), dtype=np.double)
    # Laplacien sur l'intérieur :
    lapl[:,:] = -4.*image[1:-1,1:-1] + image[1:-1,:-2] + image[1:-1,2:] + image[:-2, 1:-1] + image[2:,1:-1]
    return lapl

# ===================================================================================================
# Programme principal
WIDTH  : int = 1
HEIGHT : int = 0
globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank= globCom.rank

filename = f"output{rank:03d}.txt"
out      = open(filename, 'w')

filename = "lena_gray"
if len(sys.argv) > 1:
    filename = sys.argv[1]

shape = None
if rank == 0:
    img      = Image.open(filename+".png").convert("L") # Convert pour convertir en niveaux de gris
    globData = np.asarray(img,dtype=np.double)/255.
    shape = np.array(globData.shape)
else:
    shape = np.empty((2,), dtype=np.int64)
globCom.Bcast(shape, 0)
out.write(f"Format de l'image globale : {shape}\n")
# Si il n'y a pas d'autres arguments que l'image passé à l'exécutable, on découpe en tranche,
# Sinon on va découper selon un certain nombre de blocs par direction
nbx : int
nby : int
if len(sys.argv) > 3:
    nbx = int(sys.argv[2])
    nby = int(sys.argv[3])
    assert(nbx * nby == nbp)
else:
    nbx = 1
    nby = nbp
out.write(f"Decomposition de l'image en {nbx} x {nby} blocs\n")
# Calcul du bloc (I,J) dont va s'occuper le processus courant (J variant le plus vite)
I : int = rank // nbx 
J : int = rank %  nbx
out.write(f"Je m'occupe du bloc ({I},{J})\n")
# On crée un communicateur pour chaque groupe de processus ayant un I en commun :
rowComm = globCom.Split(I,J)
# De même avec un J en commun :
colComm = globCom.Split(J,I)
# Calcul de la taille de l'image locale (avec les cellules fantômes)
# On va supposer ici, pour simplifier le code, que le nombre de pixels
# est divisible par le nombre de blocs par ligne et colonne (pour la
# décomposition de l'image)
locWidth = shape[WIDTH ]//nbx + 2 # Le +2 c'est pour rajouter une cellule fantôme à gauche et une à droite
locHeight= shape[HEIGHT]//nby + 2 # Le +2 c'est pour rajouter une cellule fantôme en haut et en bas
out.write(f"Taille de l'image locale : ({locHeight} x {locWidth})\n")
out.flush()
# Le processus 0 qui contient l'image complète va distribuer maintenant des morceaux d'images à tous
# les autres processus. Remarquez qu'il aurait été préférable que chaque processus puisse charger la partie
# de l'image le concernant, mais cela demande beaucoup plus de code pour permettre cela !
# On terminera cette distribution par le processus 0 lui-même :
splitImage = np.zeros((locHeight,locWidth), dtype=np.double)
if rank ==0:
    buffer = np.zeros((locHeight,locWidth), dtype=np.double)
    for proc in range(nbp):
        iProc    : int = proc // nbx
        iStartImg: int = iProc*(locHeight-2) - (1 if iProc>0 else 0)
        iStartBuf: int = 0 if iProc>0 else 1
        iEndImg  : int = (iProc+1)*(locHeight-2) + (1 if iProc<nby-1 else 0)
        iEndBuf  : int = locHeight if iProc<nby-1 else locHeight-1

        jProc    : int = proc % nbx
        jStartImg: int = jProc*(locWidth-2) - (1 if jProc>0 else 0)
        jStartBuf: int = 0 if jProc>0 else 1
        jEndImg  : int = (jProc+1)*(locWidth-2) + (1 if jProc<nbx-1 else 0)
        jEndBuf  : int = locWidth if jProc<nbx-1 else locWidth-1

        if DEBUG:
            out.write(f"Pour le proc {proc} = ({iProc},{jProc}) on copie de ({iStartImg},{jStartImg})--({iEndImg},{jEndImg}) vers ({iStartBuf},{jStartBuf})--({iEndBuf},{jEndBuf})\n")

        if proc>0:
            buffer[iStartBuf:iEndBuf, jStartBuf:jEndBuf] = globData[iStartImg:iEndImg, jStartImg:jEndImg]
            globCom.Ssend(buffer, proc)
        else:
            splitImage[iStartBuf:iEndBuf, jStartBuf:jEndBuf] = globData[iStartImg:iEndImg, jStartImg:jEndImg]
else:
    globCom.Recv(splitImage, 0)

if DEBUG:
    minVal = np.min(splitImage)
    maxVal = np.max(splitImage)
    delta  = maxVal - minVal
    red = np.array(np.clip((255.*(splitImage - minVal)/delta).astype(np.uint8), 0, 255))
    partImg = np.stack((red,red,red),axis=-1)
    image = Image.fromarray(partImg, 'RGB')
    image.save(f"part_{filename}_{rank}.png")

lapl = laplacianFilter(splitImage)
rowBuffer = None
if nbx > 1:
    # On collecte les bouts d'images d'une même ligne :
    rowCollect = rowComm.gather(lapl, 0)
    if J==0:
        # Et on les reassemble pour former la bonne image:
        rowBuffer = np.zeros((locHeight-2, shape[WIDTH]), dtype=np.double)
        for i in range(rowComm.size):
            rowBuffer[:,i*(locWidth-2):(i+1)*(locWidth-2)] = rowCollect[i][:,:]
else:
    rowBuffer = lapl
# On reassemble les différentes tranches de l'image à l'aide des processus d'indices J=0 sur le proc de rang 0
globLapl = None
if rank==0:
    globLapl = np.empty(shape,dtype=np.double)
if J==0:
    colComm.Gather(rowBuffer, globLapl, 0)
if rank == 0:
    minVal = np.min(globLapl)
    maxVal = np.max(globLapl)
    delta  = maxVal - minVal
    red = np.array(np.clip((255.*(globLapl - minVal)/delta).astype(np.uint8), 0, 255))
    partImg = np.stack((red,red,red),axis=-1)
    image = Image.fromarray(partImg, 'RGB')
    image.save(f"lapl_{filename}.png")
