import sys
import numpy as np
from PIL import Image
import scipy.ndimage
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable


class GMin(object):

    def __init__(self):
        self.oris = np.pi/8*np.arange(8)
        self.sf = [23, 27]  #np.arange(7,39,4)

    def get_image(self):
        im = Image.open('the_world_zwei_input.png').convert('L')
        stim = np.asarray(im)*1.
        return stim

    def gabor(self, sf=42, theta=0, gamma=1):
        # Mutch and Lowe, 2006
        sigma = sf/6.
        lam = 2.8*sf/6.
        r = sigma*3  # there is nothing interesting 3 STDs away

        theta -= np.pi/2
        x,y = np.meshgrid(np.arange(-r,r),np.arange(-r,r))
        X = x*np.cos(theta) - y*np.sin(theta)
        Y = x*np.sin(theta) + y*np.cos(theta)
        g = np.exp( - (X**2 + (gamma*Y)**2) / (2*sigma**2) ) * np.cos( 2*np.pi*X/lam )

        g -= np.mean(g)  # mean 0
        g /= np.sum(g**2) # energy 1
        #g[np.abs(g)<.001] = 0  # remove spillover
        return g

    def get_edges(self, stim, sf, bins=5000, nelems=1000):
        gabor_max = stim

        # Step 1: extract edges using Gabor filters
        all_edges = np.zeros((len(self.oris),)+gabor_max.shape)
        for oi, ori in enumerate(self.oris):
            gab = self.gabor(sf=sf, theta=ori)
            #norm = scipy.ndimage.correlate(gabor_max**2,np.ones(gab.shape))
            edges = scipy.ndimage.correlate(gabor_max,gab)#/np.sqrt(norm)
            #edges /= np.max(edges)
            all_edges[oi] = edges
        #edge_map -= np.min(edge_map)
        #edge_map[edge_map<.01]=0

        # Step 2: normalize responses globally
        #edge_map = np.abs(edge_map)
        all_edges /= np.max(all_edges)

        # Step 3: choose the maximally responding orientation for each location
        response = np.max(all_edges, axis=0)
        oris = np.take(self.oris, np.argmax(all_edges, axis=0))
        edge_map = np.dstack((response, oris))
        edge_map = np.rollaxis(edge_map,-1)

        # Step 4: Choose only nelems top responding locations
        hist, bin_edges = np.histogram(edge_map[0].ravel(),bins=5000)
        last = np.nonzero(np.cumsum(hist[::-1]) > nelems)[0][0]
        threshold = bin_edges[bins-last-1]
        #print hist[-1]
        #threshold = bin_edges[-2]
        inds = edge_map[0]>threshold
        print '# elements used: %d' % np.sum(inds)
        edge_map[0][np.logical_not(inds)] = 0
        indices = np.transpose(np.nonzero(inds))

        return edge_map, indices

    def _isinside(self, smallbox, largebox, i, j):
        """Checks if a smaller box centered around (i,j) is inside a larger box
        """
        return (i - smallbox[0] > 0 and i + smallbox[0] + 1 < largebox[0] and
            j - smallbox[1] > 0 and j + smallbox[1] + 1 < largebox[1])

    def _boxinds(self, box, i, j):
        """Returns indices of all box elements centered around (i,j)
        """
        return np.s_[i-box[0]: i+box[0]+1, j-box[1]: j+box[1]+1]

    def plot_arrows(self, edge_map, indices, ax):
        """Plots arrows: length edge_map[0], direction edge-map[1].

        Only edge_map[:][indices] are used for plotting.
        """
        for i,j in indices:
            ax.arrow( j, edge_map.shape[1]-i,
                    np.cos(edge_map[1,i,j]) * edge_map[0,i,j],
                    np.sin(edge_map[1,i,j]) * edge_map[0,i,j],
                    head_width=.5, head_length=1, fc='k', ec='k')
        ax.set_xlim([0, edge_map.shape[2]])
        ax.set_ylim([0, edge_map.shape[1]])

    def compute_fg(self, edge_map, indices, sf):
        field_shape = (39, 39)
        box = (field_shape[0] / 2, field_shape[1] / 2)
        x,y = np.meshgrid(np.arange(-box[0], box[0] + 1),
                          np.arange(-box[1], box[1] + 1))
        # angles of each position wrt f's' center
        angle_field = -np.arctan2(y,x) % (2*np.pi)
        sigma = sf/6
        # gaussian window
        g = np.exp( - (x**2 + y**2) / (2*sigma**2) )

        fg = np.zeros(edge_map.shape)
        for i,j in indices:
            if self._isinside(box, edge_map[0].shape, i,j):
                s = self._boxinds(box, i, j)
                # put a gaussian window so that nearer elements matter more
                window_weight = edge_map[0][s] * g
                angle = edge_map[1,i,j] #+ np.pi
                # determine elements with an angle greater than 'angle'
                sel1 = angle_field >= angle
                sel2 = angle_field < angle + np.pi
                above = np.logical_and(sel1,sel2)
                # elements with an angle smaller than 'angle'
                below = np.logical_not(above)
                # magnitude of the figure-ground signal
                fg[0,i,j] = np.sum(window_weight[below]) - np.sum(window_weight[above])
                # orthogonal 'below' angle
                fg[1,i,j] = angle#-np.pi/2
        return fg

    def plot(self):
        pass

    def run(self):
        stim = self.get_image()
        plt.figure(figsize=(7,3.5), facecolor='white')

        ax = plt.subplot(121)
        edge_map, indices = self.get_edges(stim, self.sf[0], nelems=1000)
        for fgi in range(2):
            edge_map = self.compute_fg(edge_map, indices, self.sf[0])
        edge_map[1] -= np.pi/2
        self.plot_arrows(edge_map, indices, ax)
        plt.axis('off')

        ax = plt.subplot(122)
        edge_map, indices = self.get_edges(stim, self.sf[1], nelems=1000)
        for fgi in range(3):
            edge_map = self.compute_fg(edge_map, indices, self.sf[1])
        edge_map[1] -= np.pi/2
        self.plot_arrows(edge_map, indices, ax)
        plt.axis('off')

        plt.savefig('musu_pasaulis.jpg', dpi=300, format='jpg')
        plt.show()


if __name__ == '__main__':
    gmin = GMin()
    gmin.run()
