from PIL import Image  
import numpy as np
import sys

def laplacianFilter( image):
    extent = image.shape
    lapl = np.empty(extent, dtype=np.double)
    # Laplacien sur la première ligne :
    lapl[0,0]    = -4*image[0,0] + image[0,1] + image[1,0]
    lapl[0,1:-1] = -4.*image[0,1:-1] + image[0,:-2] + image[0,2:] + image[1,1:-1]
    lapl[0,-1]   = -4.*image[0,-1] + image[0,-2] + image[1,-1]
    # Laplacien sur la dernière ligne :
    lapl[-1,0]   = -4.*image[-1,0] + image[-2,0] + image[-1,1]
    lapl[-1,1:-1]= -4.*image[-1,1:-1] + image[-2,1:-1] + image[-1,:-2] + image[-1,2:]
    lapl[-1,-1]  = -4.*image[-1,-1] + image[-2,-1] + image[-1,-2]
    # Laplacien sur la première colonne :
    lapl[1:-1,0] = -4.*image[1:-1,0] + image[:-2,0] + image[2:,0] + image[1:-1,1]
    # Laplacien sur la dernière colonne :
    lapl[1:-1,-1]= -4.*image[1:-1,-1] + image[:-2,-1] + image[2:,-1] + image[1:-1,-2]
    # Laplacien sur l'intérieur :
    lapl[1:-1,1:-1] = -4.*image[1:-1,1:-1] + image[1:-1,:-2] + image[1:-1,2:] + image[:-2, 1:-1] + image[2:,1:-1]
    return lapl

# ===================================================================================================
# Programme principal
filename = "lena_gray"
if len(sys.argv) > 1:
    filename = sys.argv[1]

img  = Image.open(filename+".png").convert("L") # Convert pour convertir en niveaux de gris
data = np.asarray(img,dtype=np.double)/255.

laplacian = laplacianFilter(data)
laplacian.tofile(f"lapl_{filename}.data")

minVal = np.min(laplacian)
maxVal = np.max(laplacian)
delta  = maxVal - minVal
red = np.array(np.clip((255.*(laplacian - minVal)/delta).astype(np.uint8), 0, 255))
laplimag = np.stack((red,red,red),axis=-1)
image = Image.fromarray(laplimag, 'RGB')
image.save(f"lapl_{filename}.png")
