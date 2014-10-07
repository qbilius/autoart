import numpy as np
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt


def gabor(
    theta=0,
    gamma=1,
    sigma=2,
    lam=5.6,
    k=10
    ):
    # Mutch and Lowe, 2006
    theta -= np.pi/2
    x,y = np.meshgrid(np.arange(-k,k),np.arange(-k,k))
    X = x*np.cos(theta) - y*np.sin(theta)
    Y = x*np.sin(theta) + y*np.cos(theta)
    g = np.exp( - (X**2 + (gamma*Y)**2) / (2*sigma**2) ) * np.cos( 2*np.pi*X/lam )

    g -= np.mean(g)  # mean 0
    g /= np.sum(g**2) # energy 1
    g[np.abs(g)<.001] = 0
    return g

def get_edges(stim, oris, sf=1):
    gabor_max = stim
    edge_map = np.zeros((len(oris),)+gabor_max.shape)
    sf=1
    for oi, ori in enumerate(oris):
        gab = gabor(theta=ori, sigma=2*sf,lam=5.6*sf,k=10*sf)
        edges = scipy.ndimage.correlate(gabor_max,gab)
        edge_map[oi] = edges

    gabor_max = np.max(edge_map, axis=0)
    gabor_argmax = np.argmax(edge_map, axis=0)

    return gabor_max, gabor_argmax

im = Image.open('dots_input.png').convert('L')
stim = np.asarray(im)*1.
stim = stim[50:125,50:125]
oris = np.pi/8*np.arange(8)
gabor_max, gabor_argmax = get_edges(stim, oris, sf=1)
hist, bin_edges = np.histogram(gabor_max.ravel(),bins=1)
threshold = bin_edges[-2]
inds = gabor_max>threshold
gabor_max[np.logical_not(inds)] = 0
plt.imshow(gabor_max)
plt.axis('off')
#plt.show()
plt.savefig('dots.jpg', dpi=300, format='jpg',
    bbox_inches='tight', pad_inches=0)

