from PIL import Image  
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import numpy.linalg as nplin
import sys
from mpi4py import MPI
import math

DEBUG=False

def createLaplacian(extent):
    N = extent[0]*extent[1]
    mainDiag = np.ones(N)*4.
    sideDiag = -np.ones(N-1)
    sideDiag[np.arange(1,N)%extent[WIDTH]==0] = 0
    up_down_diag = -np.ones(N-3)
    diagonals = [mainDiag,sideDiag,sideDiag,up_down_diag,up_down_diag]
    return sparse.diags(diagonals, [0, -1, 1,extent[WIDTH],-extent[WIDTH]], format="csc")
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


# Lecture de la dimension de l'image par le processus 0 et diffusion aux autres processeurs
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
locWidth = shape[WIDTH ]//nbx
locHeight= shape[HEIGHT]//nby
out.write(f"Taille de l'image locale : ({locHeight} x {locWidth})\n")
out.flush()
# Le processus 0 qui contient l'image complète va distribuer maintenant des morceaux d'images à tous
# les autres processus. Remarquez qu'il aurait été préférable que chaque processus puisse charger la partie
# de l'image le concernant, mais cela demande beaucoup plus de code pour permettre cela !
# On terminera cette distribution par le processus 0 lui-même :
locRhs = np.zeros((locHeight,locWidth), dtype=np.double)
if rank ==0:
    rhs = -np.fromfile("lapl_"+filename+".data")
    rhsImg = np.reshape(rhs, shape)
    buffer = np.zeros((locHeight,locWidth), dtype=np.double)
    for proc in range(nbp):
        iProc    : int = proc // nbx
        iStartImg: int = iProc*locHeight
        iEndImg  : int = (iProc+1)*locHeight

        jProc    : int = proc % nbx
        jStartImg: int = jProc*locWidth
        jEndImg  : int = (jProc+1)*locWidth

        if DEBUG:
            out.write(f"Pour le proc {proc} = ({iProc},{jProc}) on copie partie ({iStartImg},{jStartImg})--({iEndImg},{jEndImg})\n")

        if proc>0:
            buffer[:, :] = rhsImg[iStartImg:iEndImg, jStartImg:jEndImg]
            globCom.Ssend(buffer, proc)
        else:
            locRhs[:, :] = rhsImg[iStartImg:iEndImg, jStartImg:jEndImg]
else:
    globCom.Recv(locRhs, 0)

print("Assemblage", flush=True)
ALoc = createLaplacian( (locHeight, locWidth) )
print("Factorisation", flush=True)
LUloc = linalg.splu(ALoc)
# Début de l'algorithme de Schwarz :
hasConverged = False
solLocNew = np.zeros( (locHeight,locWidth) )
leftBC    = np.zeros( locHeight )
rightBC   = np.zeros( locHeight )
botBC     = np.zeros( locWidth  )
topBC     = np.zeros( locWidth  )
iter = 0
res0 = 1.
while not hasConverged:
    # Mise à jour des "conditions limites"
    # On prépare la réception des nouvelles conditions limites :
    lstReq = []
    if I>0: # Si je ne suis pas un processus complètement à gauche dans la grille :
        lstReq.append(colComm.Irecv(topBC, I-1)) # Je reçois la nouvelle condition limite au sommet
    if I<nby-1 : # Si je ne suis pas un processus complètement à droite dans la grille :
        lstReq.append(colComm.Irecv(botBC, I+1)) # Je reçois la nouvelle condition limite en bas
    if J>0: # Si je ne suis pas un processus complètement en haut :
        lstReq.append(rowComm.Irecv(leftBC, J-1)) # Je reçois une nouvelle condition limite à gauche
    if J<nbx-1:# Si je ne suis pas un processus complètement en bas
        lstReq.append(rowComm.Irecv(rightBC, J+1)) # Je reçois une nouvelle condition limite à droite
    # On envoie nos bords aux processus voisins sur la grille :
    if I>0:
        buffer = np.array(solLocNew[0,:], copy=True)
        colComm.Ssend(buffer, I-1)
    if I<nby-1:
        buffer = np.array(solLocNew[-1,:], copy=True)
        colComm.Ssend(buffer, I+1)
    if J>0:
        buffer = np.array(solLocNew[:,0], copy=True)
        rowComm.Ssend(buffer, J-1)
    if J<nbx-1:
        buffer = np.array(solLocNew[:,-1], copy=True)
        rowComm.Ssend(buffer, J+1)
    
    # On attend la fin des réceptions :
    MPI.Request.waitall(lstReq)

    # On calcul la contribution des conditions limites sur le membre de droite (rhs):
    rhs = np.copy(locRhs)
    rhs[:,0]  += leftBC[:]
    rhs[:,-1] += rightBC[:]
    rhs[-1,:]  += botBC[:]
    rhs[0,:] += topBC[:]

    # Résolution du système linéaire ainsi constitué:
    sol = LUloc.solve(rhs.flatten())
    diff = sol - solLocNew.flatten()
    err = diff.dot(diff)
    err = globCom.allreduce(err, MPI.SUM)
    err = math.sqrt(err)/res0
    if iter==0: 
        res0 = err
        err  = 1.
    if rank==0: print(f"||err|| = {err}", flush=True)
    solLocNew = np.reshape(sol, rhs.shape)
    if err < 1.E-3:
        hasConverged = True
    else:
        iter = iter + 1
        out.write(f"Schwarz, iteration {iter} , ||err|| = {err}\n")         


rowBuffer = None
if nbx > 1:
    # On collecte les bouts d'images d'une même ligne :
    rowCollect = rowComm.gather(solLocNew, 0)
    if J==0:
        # Et on les reassemble pour former la bonne image:
        rowBuffer = np.zeros((locHeight, shape[WIDTH]), dtype=np.double)
        for i in range(rowComm.size):
            rowBuffer[:,i*locWidth:(i+1)*locWidth] = rowCollect[i][:,:]
else:
    rowBuffer = solLocNew
# On reassemble les différentes tranches de l'image à l'aide des processus d'indices J=0 sur le proc de rang 0
globSol = None
if rank==0:
    globSol = np.empty(shape,dtype=np.double)
if J==0:
    colComm.Gather(rowBuffer, globSol, 0)
if rank == 0:
    minVal = np.min(globSol)
    maxVal = np.max(globSol)
    delta  = maxVal - minVal
    red = np.array(np.clip((255.*(globSol - minVal)/delta).astype(np.uint8), 0, 255))
    partImg = np.stack((red,red,red),axis=-1)
    image = Image.fromarray(partImg, 'RGB')
    image.save(f"inpainting_{filename}.png")
