from PIL import Image  
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import numpy.linalg as nplin
import sys
from mpi4py import MPI
import math

DEBUG=True
POSITION=0
TAILLE  =1
CHILDREN=2
OY      = 0
OX      = 1
HEIGHT  = 0
WIDTH   = 1

def createLaplacian(extent):
    N = extent[0]*extent[1]
    mainDiag = np.ones(N)*4.
    sideDiag = -np.ones(N-1)
    sideDiag[np.arange(1,N)%extent[WIDTH]==0] = 0
    up_down_diag = -np.ones(N-3)
    diagonals = [mainDiag,sideDiag,sideDiag,up_down_diag,up_down_diag]
    return sparse.diags(diagonals, [0, -1, 1,extent[WIDTH],-extent[WIDTH]], format="csc")

def splitDomain( domain, level : int ):
    """
    Découpe récursivement un domaine en deux sous-domaines séparés par une interface.
    La récursion se termine lorsque level vaut un (level décroit à chaque appel récursif)

    Retourne sous forme de liste l'historique de la décomposition en dissection emboîtée du domaine initiale

    Chaque sous domaine est sous la forme [ indices du point en haut à gauche du domaine,
                                            dimensions du sous domaine,
                                            [ liste évntuelle des sous-domaines décomposant le sous-domaine]]
    """
    deb = domain[POSITION]
    size= domain[TAILLE]

    if size[OX] > size[OY]:
        ny = size[OY]
        nx = (size[OX]-1)//2
        reste = (size[OX]-1) % 2
        nx1   = nx if reste==0 else nx+1
        nx2   = nx
        domain[CHILDREN].append( [        deb,               (ny, nx1), []])
        domain[CHILDREN].append( [ (deb[OY], deb[OX]+nx1+1), (ny, nx2), []])
        domain[CHILDREN].append( [ (deb[OY], deb[OX]+nx1  ), (ny, 1  ), None])
    else:
        ny = (size[OY]-1)//2
        reste = (size[OY]-1) % 2
        nx = size[OX]
        ny1   = ny if reste==0 else ny+1
        ny2   = ny
        domain[CHILDREN].append( [          deb,             (ny1, nx), []])
        domain[CHILDREN].append( [ (deb[OY]+ny1+1, deb[OX]), (ny2, nx), []])
        domain[CHILDREN].append( [ (deb[OY]+ny1  , deb[OX]), (  1, nx), None])
    if level > 1:
        splitDomain( domain[CHILDREN][0], level-1)
        splitDomain( domain[CHILDREN][1], level-1)

def nestedDecompositionDomain( height : int, width : int, nlevels : int ):
    """
    Crée un domaine de dimension height x width et effectue une dissection emboîtée de
    nlevels niveaux.
    """
    domain = [ (0,0), (height, width), [] ]
    splitDomain( domain, nlevels)
    return domain

def displayDecomposition( domain, tab = '' ) :
    """
    Affiche un domaine et sa décomposition de façon lisible pour un humain
    """
    print(f"{tab}domain beginning at {domain[POSITION]} with size {domain[TAILLE]}");
    if domain[CHILDREN] is not None and len(domain[CHILDREN]) > 0:
        print(f"{tab} is decomposed as : ")
        displayDecomposition( domain[CHILDREN][0], tab + '\t')
        print(f"{tab} and ")
        displayDecomposition( domain[CHILDREN][1], tab + '\t')
        print(f"{tab} with interface ")
        displayDecomposition( domain[CHILDREN][2], tab + '\t')

def buildMatrix( decomposition, rank : int , level : int, matrices ):
    if level > 1:
        if (rank>>(level-1)) > 0:
            buildMatrix( decomposition[CHILDREN][1], rank-(1<<(level-1)), level-1, matrices)
        else:
            buildMatrix( decomposition[CHILDREN][0], rank, level-1, matrices)
        matrices.append(createLaplacian())

def buildMatrices( decomposition, int rank, int nlevels )
    matrices = []


domains = nestedDecompositionDomain( 512, 512, 4)
displayDecomposition(domains)