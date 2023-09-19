import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import numpy.linalg as nplin
from PIL import Image  
import time
import sys

DEBUG  : bool = True
WIDTH  : int = 1
HEIGHT : int = 0

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

filename = "lena_gray"
if len(sys.argv) > 1:
    filename = sys.argv[1]

useIterSolver = False 
if len(sys.argv) > 2:
    useIterSolver = (sys.argv[2]=="True")

img  = Image.open(filename+".png").convert("L") # Convert pour convertir en niveaux de gris
imgData = np.asarray(img,dtype=np.double)/255.

rhs = -np.fromfile("lapl_"+filename+".data")

extent = imgData.shape
deb = time.time()
lhs = createLaplacian(extent)
fin = time.time()
print(f"Temps pour creer le laplacien : {fin-deb} secondes")

sol = None
if useIterSolver:
    deb = time.time()
    sol, conv = linalg.cg(lhs, rhs, tol=1.E-6, maxiter=500 )
    fin = time.time()
    if not conv:
        print(f"Attention, la methode du gradient conjugue n'a pas converge")
    print(f"Temps pour resoudre le systeme lineaire avec le GC : {fin-deb} secondes")
else:
    deb = time.time()
    LU = linalg.splu(lhs)
    fin = time.time()
    print(f"Temps pour factoriser la matrice : {fin-deb} secondes")

    deb = time.time()
    sol = LU.solve(rhs)
    fin = time.time()
    print(f"Temps pour resoudre le systeme lineaire : {fin-deb} secondes", flush=1)
error = lhs * sol - rhs 
nrmErr = nplin.norm(error)
nrmrhs = nplin.norm(rhs)
print(f"Erreur numerique trouve : {nrmErr/nrmrhs}")

sol = np.reshape(sol, extent)
minVal = np.min(sol)
maxVal = np.max(sol)
delta  = maxVal - minVal
red = np.array(np.clip((255.*(sol - minVal)/delta).astype(np.uint8), 0, 255))
solimag = np.stack((red,red,red),axis=-1)
image = Image.fromarray(solimag, 'RGB')
image.save(f"inpainting_{filename}.png")
