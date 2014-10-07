import sys
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image


def gen_gabor(theta=0, gamma=1, sigma=2, lam=5.6,k=10):
    # Mutch and Lowe, 2006
    theta -= np.pi/2
    x,y = np.meshgrid(np.arange(-k,k),np.arange(-k,k))
    X = x*np.cos(theta) - y*np.sin(theta)
    Y = x*np.sin(theta) + y*np.cos(theta)
    g = np.exp( - (X**2 + (gamma*Y)**2) / (2*sigma**2) ) * np.cos( 2*np.pi*X/lam )

    g -= np.mean(g)  # mean 0
    g /= np.sum(g**2) # energy 1
    return g


def ernst(
    shape=(40,40),
    beta = 0, # orientation
    r0 = 11.6/4,
    sigma_a = .27,  # tolerance to co-circularity; chosen to be optimal from doi:10.1371/journal.pcbi.1002520.g006
    sigma_b = .47,  # curvature; chosen to be optimal from doi:10.1371/journal.pcbi.1002520.g007 assuming a typo in reporting .57
    ):

    x,y=np.meshgrid(np.arange(-shape[0]/2,shape[0]/2),np.arange(-shape[1]/2,shape[1]/2))
#            x, y, beta = ej-ei  # angle between the two elements
    r = np.sqrt(x**2+y**2)
    alpha = np.arctan2(y,x) # angle of a line connecting the two centers

    Ad = np.exp(-r/r0)
    At = np.cosh( -np.cos(beta/2-alpha)/sigma_a**2 + 4*np.cos(beta/2)/sigma_b**2 )+\
         np.cosh( np.cos(beta/2-alpha)/sigma_a**2 + 4*np.cos(beta/2)/sigma_b**2 )

    A = Ad*At

    return A/np.sum(A)


def probmap(edge_map,assoc_field):
    # here the idea is to take the response to an edge at each position
    # and multiply it by association field probabilities
    # then add the resulting posteriors (across positions)
    # and move particles to the highest probability regions
    # Convolution does this trick albeit maybe it's not trivial too see that

    prob_map = scipy.ndimage.convolve(edge_map,assoc_field)# /\
        #scipy.ndimage.correlate(edge_map**2,np.ones(assoc_field.shape))

    return prob_map


im = Image.open('010.png').convert('L')
stim = np.asarray(im)*1.
oris = np.pi/18*np.arange(18)
thres = .002
grid_size = 10

mean = stim
for t in range(5): # loop over time
    print t,
    edge_map = np.zeros((len(oris),)+mean.shape)
    for oi,ori in enumerate(oris):
        gabor = gen_gabor(theta=ori)
        norm = scipy.ndimage.correlate(mean**2,np.ones(gabor.shape))
        edges = scipy.ndimage.correlate(mean,gabor)/np.sqrt(norm)
        edges[edges<thres] = 0
        assoc_field = ernst(beta=-2*ori)
        edges = probmap(edges,assoc_field)
        edge_map[oi] = edges
    mean = np.max(edge_map,0)

plt.imshow(mean,cmap=mpl.cm.gray)
plt.axis('off')
plt.show()
#plt.savefig('metalo_laidai.jpg', dpi=300, format='jpg',
#    bbox_inches='tight', pad_inches=0)



## NOT SURE WHY THIS IS HERE
mean = stim
for t in range(1): # loop over time
    print t,
    edge_map = np.zeros((len(oris),)+mean.shape)
    for oi,ori in enumerate(oris):
        gabor = gen_gabor(theta=ori, sigma=4,lam=11.2)
        norm = scipy.ndimage.correlate(mean**2,np.ones(gabor.shape))
        edges = scipy.ndimage.correlate(mean,gabor)/np.sqrt(norm)
        edges[edges<thres] = 0
        assoc_field = ernst(beta=-2*ori)
        edges = probmap(edges,assoc_field)
        edge_map[oi] = edges
    mean = np.max(edge_map,0)

plt.imshow(mean,cmap=mpl.cm.gray)
plt.show()
