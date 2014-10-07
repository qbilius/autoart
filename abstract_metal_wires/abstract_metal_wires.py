import sys
import psychopy
from psychopy import visual, core
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

# import hmax

class Filters(object):
    def gabor(self,
        theta=0,
        gamma=1,
        sigma=2,
        lam=5.6,
        k=10
        ):
#        g = np.exp(-np.pi*((alpha*x)**2+(beta*y)**2) ) * np.exp(-2*np.pi*np.i(u*x+v*y))
#        g = mu**2/sigma**2 * np.exp( - mu**2 (x**2+y**2) / (2*sigma**2) ) * np.exp(np.i*mu*(x*np.cos(theta)+y*np.sin(theta)))
        # Mutch and Lowe, 2006
        theta -= np.pi/2
        x,y = np.meshgrid(np.arange(-k,k),np.arange(-k,k))
        X = x*np.cos(theta) - y*np.sin(theta)
        Y = x*np.sin(theta) + y*np.cos(theta)
        g = np.exp( - (X**2 + (gamma*Y)**2) / (2*sigma**2) ) * np.cos( 2*np.pi*X/lam )

        g -= np.mean(g)  # mean 0
        g /= np.sum(g**2) # energy 1
        return g

    def gabor_pyramid(self):
        pass

    def second_gaussian(self):
        pass

    def gabor_circle(self,
        r = 40,
        k = 10,
        #n = 36,
        #theta=0,
        gamma=1,
        sigma=2,
        lam=5.6,
        ):

        oris = 2*np.pi/r*np.arange(r)
        k += r
        circle = np.zeros((2*k,2*k))

        for ori in oris:
            theta = ori - np.pi/2
            x,y = np.meshgrid(np.arange(-k,k),np.arange(-k,k))
            x -= -r*np.cos(theta)
            y -= r*np.sin(theta)
            X = x*np.cos(theta) - y*np.sin(theta)
            Y = x*np.sin(theta) + y*np.cos(theta)
            g = np.exp( - (X**2 + (gamma*Y)**2) / (2*sigma**2) ) * np.cos( 2*np.pi*X/lam )

            g -= np.mean(g)  # mean 0
            g /= np.sum(g**2) # energy 1
            circle += g

        #
        #circle[circle.shape[0]/2-4:circle.shape[0]/2+4,
            #circle.shape[1]/2-4:circle.shape[1]/2+4] = 4*np.max(circle)
        #import pdb; pdb.set_trace()
        return circle

    def plot(self):
        circle = self.gabor_circle()
        plt.imshow(circle,cmap=mpl.cm.gray)
        plt.show()
        sys.exit()

    class association_field(object):
        def naive(self,
            o1,
            o2,
            a=.6  # from Sigman et al., 2001
            ):
            r = np.linalg.norm(o1,o2)
            theta1 = np.arctan2(o1[1],o1[0])
            theta2 = np.arctan2(o2[1],o2[0])
            optimal_angle = 2*theta1-phi
            # penalty for suboptimal angle times penalty for distance
            w = np.cos(2*(optimal_angle - theta2)) * r**(-a)
            return w

        def watts(self,
            e1,  # one gabor (x,y) position and orientation (in degrees)
            e2,  # the rest of the field
            ds = .1,  # size of the Gaussian envelope (overall size))
            cs = np.pi/9,  # curvature sensitivity
            ts = np.pi/18,  # tolerance for non-circularity (larger values mean that cocircularity is more ignored)
            method = np.sum  # how to calculate the final weight; could also be np.max
            ):

            # R = np.array([np.cos(-t0), -sin ])
            x, y, theta = e2-e1
            xx = x*np.cos(-t0) - y*np.sin(-t0)
            yy = y*np.cos(-t0) + x*np.sin(-t0)

            dd = (xx**2+yy**2)/ds**2
            curvature = yy/(dd*ds)
            spatial_weight = np.exp(-dd) * np.exp(-(curvature/cs)**2 / 2 )

            theta_optimal = 2*np.arctan2(yy,xx) - ei[2]  # ei[2] this was not present in the original
                # presumably ei[2]=0 in those simulations

            theta_difference = theta_optimal-ej[2]  # instead of subtrating theta
            if theta_difference > np.pi/2 or theta_difference < -np.pi/2:
                theta_difference = np.pi - np.abs(theta_difference)
            a=exp(-((theta_difference/ts)**2)/2);


            weight = method(spatial_weight*a)

            return weight

        def ernst_orig(self,
            ei,
            ej,
            r0 = 1.16/4,
            sigma_a = .27,  # tolerance to co-circularity; chosen to be optimal from doi:10.1371/journal.pcbi.1002520.g006
            sigma_b = .47,  # curvature; chosen to be optimal from doi:10.1371/journal.pcbi.1002520.g007 assuming a typo in reporting .57
            ):
            x, y, beta = ej-ei  # angle between the two elements
            r = np.linalg.norm([x,y])
            alpha = np.arctan2(y,x)-ei[2]  # angle of a line connecting the two centers (minus ei orientation)
            # beta = ej[2] - ei[2]

            Ad = np.exp(-r/r0)
            At = np.cosh( np.cos(beta/2-alpha)/sigma_a**2 + 4*np.cos(beta/2)/sigma_b**2 )
            A = Ad*At

            # plt.figure()
            # plt.imshow(Ad)
            # plt.figure()
            # plt.imshow(At)
            # plt.show()
            #import pdb; pdb.set_trace()
            # K = 2*np.sin(beta/2)/r
            return A


        def ernst(self,
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

            A = A/np.sum(A)
            return A-np.mean(A)


        def ernst_half(self,
            shape=(40,40),
            beta = 0, # orientation
            r0 = 11.6/4,
            sigma_a = .27,  # tolerance to co-circularity; chosen to be optimal from doi:10.1371/journal.pcbi.1002520.g006
            sigma_b = .47,  # curvature; chosen to be optimal from doi:10.1371/journal.pcbi.1002520.g007 assuming a typo in reporting .57
            curvature = 'convex',  # 'convex' or 'concave'
            ):

            x,y=np.meshgrid(np.arange(-shape[0]/2,shape[0]/2),np.arange(-shape[1]/2,shape[1]/2))
#            x, y, beta = ej-ei  # angle between the two elements
            r = np.sqrt(x**2+y**2)
            alpha = np.arctan2(y,x) # angle of a line connecting the two centers

            Ad = np.exp(-r/r0)
            At = np.cosh( -np.cos(beta/2-alpha)/sigma_a**2 + 4*np.cos(beta/2)/sigma_b**2 )+\
                 np.cosh( np.cos(beta/2-alpha)/sigma_a**2 + 4*np.cos(beta/2)/sigma_b**2 )

            A = Ad*At
            if curvature == 'convex':
                A[:A.shape[0]/2] = 0
            else:
                A[A.shape[0]/2:] = 0

            return A/np.sum(A)


        def ernst_trans(self,
            shape=(100,100),
            #size = (120,120),
            beta = 0, # orientation
            r0 = 11.6/4,
            sigma_a = .27,  # tolerance to co-circularity; chosen to be optimal from doi:10.1371/journal.pcbi.1002520.g006
            sigma_b = .47,  # curvature; chosen to be optimal from doi:10.1371/journal.pcbi.1002520.g007 assuming a typo in reporting .57
            d = 10,
            ):

            x,y=np.meshgrid(np.arange(-shape[0]/2,shape[0]/2),np.arange(-shape[1]/2,shape[1]/2))
#            x, y, beta = ej-ei  # angle between the two elements
            shift_x = d*np.sin(beta/2)
            shift_y = -d*np.cos(beta/2)
            A = np.zeros(shape)
            for ind in range(-shape[0]/2/d,shape[0]/2/d):
                xn = x-ind*shift_x
                yn = y-ind*shift_y
                r = np.sqrt(xn**2+yn**2)
                alpha = np.arctan2(yn,xn) # angle of a line connecting the two centers

                Ad = np.exp(-r/r0)
                At = np.cosh( -np.cos(beta/2-alpha)/sigma_a**2 + 4*np.cos(beta/2)/sigma_b**2 )+\
                     np.cosh( np.cos(beta/2-alpha)/sigma_a**2 + 4*np.cos(beta/2)/sigma_b**2 )

                A += Ad*At

            A = A/np.sum(A)
            A -= np.mean(A)

            return A

        def plot(self, filter_name=None):
            assoc = self.ernst_trans(beta=-2*np.pi/2)
            plt.imshow(assoc)
            plt.show()
            sys.exit()



class Model(object):
    def __init__(self):
        pass

    def reSample(self,w):
        #RESAMPLE Residual Resampling (Liu et al)
        #   IX = reSample(w), where w is weights and IX is index set of the
        #   resulting particles
        n = len(w)
        w = n*w/np.sum(w) # normalize to sum up to n
        wN = np.floor(w) # integer parts
        wR = w-wN # residual weigths
        wR = wR/np.sum(wR) # normalize

        # filling indexes with integer parts
        k = 1
        IX = np.zeros((n,1))
        for i in range(n):
            for j in range(wN[i]):
                IX[k] = i
                k += 1

        # use residuals to fill rest with roulette wheel selection
        cs = np.cumsum(wR)
        for j in range(k,n):
            ix = np.nonzero(cs > np.random.rand())[0]
            IX[j] = ix

    def weighted_sample(self,weights, n):
        a,b = weights.shape
        weights = weights.ravel()
        total = np.sum(weights)
        i = 0
        w = weights[0]
        v = 0
        out = []
        while n:
            x = total * (1 - np.random.rand() ** (1.0 / n))
            total -= x
            while x > w:
                x -= w
                i += 1
                w = weights[i]
                v = i
            w -= x
            out.append((v/b,v%b))
            n -= 1
        return np.array(out)


    def get_filters(self, filter_sizes, num_orientation, sigDivisor = 4.):
        self.filter_sizes = filter_sizes
        self.num_orientation = num_orientation
        self.gaussFilters = []
        for filter_size in filter_sizes:
            fxx = np.zeros((filter_size,filter_size,num_orientation))
            sigmaq = (filter_size/sigDivisor)**2
            i = np.arange(-filter_size/2+1,filter_size/2+1)
            ii,jj = np.meshgrid(i,i)
            for t in range(num_orientation):

                theta = t*np.pi/num_orientation
                x = ii*np.cos(theta) - jj*np.sin(theta)
                y = ii*np.sin(theta) + jj*np.cos(theta)
                fxx[:,:,t] = (y**2/sigmaq-1)/sigmaq * np.exp(-(x**2+y**2)/(2*sigmaq))
                fxx[:,:,t] -= np.mean(fxx[:,:,t])
                fxx[:,:,t] /= np.sqrt(np.sum(fxx[:,:,t]**2))

            self.gaussFilters.append(fxx)

    def set_filters(self):
        pass

    def S1resp_zeropad(self, stim):

            # function S1 = S1resp_zeropad (stim)
            # This function returns S1 responses with zero-padding,
            # using the difference of the Gaussians as S1 filters.
            # Filters are based on the original HMAX model.

            # filter_sizes = filter_sizes_all[whichBand]
            num_filter = len(self.filter_sizes)
            # make S1 same size as stimulus
            S1 = np.zeros((stim.shape[0], stim.shape[1], num_filter, self.num_orientation))

            for j in range(num_filter):
                S1_filter = self.gaussFilters[j]
                fs = self.filter_sizes[j]
                norm = scipy.ndimage.convolve(stim**2, np.ones((fs,fs)),mode='constant') + sys.float_info.epsilon

                for i in range(self.num_orientation):
                    S1_buf = scipy.ndimage.convolve(stim, S1_filter[:,:,i],mode='constant')
                    S1[:,:,j,i] = S1_buf/np.sqrt(norm)
                    # Riesenhuber states that this 'contrast invariance' is done at C1
                    # and S1 should rather produce outputs in the range [-1,1]

            return S1

    def get_saliency_map(self, stim,n_part=None):
#        xi = np.random.randint(0,stim.shape[0],n_elem)
#        yi = np.random.randint(0,stim.shape[1],n_elem)
        saliency = np.zeros((stim.size,))
        inds = np.random.randint(0,stim.size,n_part)
        saliency[inds] = 1
        return saliency.reshape(stim.shape)


    def detect_edges(self,stim, particle_pos, ori = 0, filter_stack = None):
        def selective_filter(array, cond):
            if cond: return np.dot(filter_flat,array)
            else: return 0

        gabor = Filters().gabor(theta=ori)
        k = gabor.shape[0]/2

        filter_flat = gabor.ravel()
#        edge_map = sp.ndimage.generic_filter(stim,selective_filter,size = gabor.shape)
        edge_map = np.zeros(stim.shape)
        for pos in particle_pos:
            # check that the filter fits within the stim box
            if pos[0]-k>0 and pos[0]+k<stim.shape[0] and \
                pos[1]-k>0 and pos[1]+k<stim.shape[1]:
                neighbors = stim[pos[0]-k:pos[0]+k,pos[1]-k:pos[1]+k]
                edge_map[pos[0],pos[1]] = np.dot(neighbors.ravel(),filter_flat)
            else:
                edge_map[pos[0],pos[1]] = 0

        return np.abs(edge_map)


    def probmap(self,edge_map,assoc_field):
        # here the idea is to take the response to an edge at each position
        # and multiply it by association field probabilities
        # then add the resulting posteriors (across positions)
        # and move particles to the highest probability regions
        # Convolution does this trick albeit maybe it's not trivial too see that

        prob_map = scipy.ndimage.convolve(edge_map,assoc_field, mode='constant')# /\
            #scipy.ndimage.correlate(edge_map**2,np.ones(assoc_field.shape))

        return prob_map


    def run_thres(self):
        im = Image.open('010a.png').convert('L')
        stim = np.asarray(im)*1.
        oris = np.pi/18*np.arange(18)
        rs = np.arange(20,50,2)
        thres = .002
        grid_size = 10
        sf = 1

        mean = stim
        for t in range(1): # loop over time
            print str(t)+':',
            edge_map = np.zeros((len(oris),)+mean.shape)
            #surface_map = np.zeros((len(oris),)+mean.shape)
            #for curno, cur in enumerate(['convex','concave']):
            #for oi,ori in enumerate(oris):
            for ri,r in enumerate(rs):
                print ri,

                #gabor = Filters().gabor(theta=ori, sigma=2*sf,lam=5.6*sf,k=10*sf)
                gabor = Filters().gabor_circle(r=r)
                norm = scipy.ndimage.correlate(mean**2,np.ones(gabor.shape),mode='nearest')
                edges = scipy.ndimage.correlate(mean,gabor,mode='nearest')/np.sqrt(norm)
                edges[edges<thres] = 0
                #assoc_field = Filters().association_field().ernst_trans(beta=-2*ori)
                #assoc_field90 = Filters().association_field().ernst(beta=-2*ori+np.pi/2)
                #assoc_field_s = Filters().association_field().ernst(shape=(40,40),
                    #beta=-2*ori+np.pi/2,
                    #r0=33,
                    #)
                #edges = self.probmap(edges,assoc_field)#-assoc_field90)
                #edges -= np.max(edges)*.3
                edges[edges<0] = 0
                #surface_map[oi] = scipy.ndimage.convolve(mean,assoc_field_s)
                #import pdb; pdb.set_trace()

                edge_map[ri] = edges
            mean = np.max(edge_map, axis=0)

            #import pdb; pdb.set_trace()
            #mean_s = np.max(surface_map,0)


        #for sno in range(len(mean)):
            #plt.subplot(2,2,sno+1)
            #plt.imshow(mean[sno],cmap=mpl.cm.gray)
        #plt.subplot(121)
        #plt.imshow(stim,cmap=mpl.cm.gray)
        #plt.subplot(122)
        plt.imshow(mean,cmap=mpl.cm.gray)
        plt.axis('off')
        #plt.show()
        plt.savefig('plieno_voratinkliai.jpg', dpi=300, format='jpg',
            bbox_inches='tight', pad_inches=0)

        sys.exit()
        #import pdb; pdb.set_trace()


            #k = gabor.shape[0]/2
            #grid = np.meshgrid(
                #np.arange(k,stim.shape[0],grid_size),
                #np.arange(k,stim.shape[1],grid_size)
                #)
            #filter_flat = gabor.ravel()
            #edge_map = np.zeros(stim.shape)
            #for x,y in grid:
                #neighbors = stim[x-k:x+k,y-k:y+k]
                #edge_map[x,y] = np.dot(neighbors.ravel(),filter_flat)
            #edge_map = self.detect_edges(stim,particle_pos,ori= ori)

        sys.exit()
        saliency = self.get_saliency_map(stim,n_part=n_part)
        for t in range(10): # loop over time
            print t,
            prob_map = np.zeros([len(oris),stim.shape[0],stim.shape[1]])
            for oi,ori in enumerate(oris):

                # prob_map[oi]=scipy.ndimage.convolve(stim,Filters().gabor(theta=ori))

                #plt.imshow(saliency);plt.show()
                particle_pos = self.weighted_sample(saliency,n_part)
                # ch=np.zeros(stim.shape)
                # for p in particle_pos:
                    # ch[p[0],p[1]] = 1
                # if t==1:plt.imshow(ch,cmap='gray');plt.show()

                edge_map = self.detect_edges(stim,particle_pos,ori= ori)
#                plt.imshow(edge_map,cmap='gray');plt.show()
                assoc_field = Filters().association_field().ernst(beta=-2*ori)
#                plt.imshow(assoc_field,cmap='gray');plt.show()
                prob_map[oi] = self.probmap(edge_map, assoc_field)
#                plt.imshow(prob_map[oi],cmap='gray');plt.show()
            saliency = np.sum(prob_map,axis=0)
            saliency /= np.sum(saliency)
        plt.imshow(saliency,cmap='gray');plt.colorbar();plt.show()


    def run_partfilt(self):
        im = Image.open('L-POST/images/010.png').convert('L')
        stim = np.asarray(im)*1.
        oris = np.pi/18*np.arange(18)
        n_part = 1000
        saliency = self.get_saliency_map(stim,n_part=n_part)
        for t in range(10): # loop over time
            print t,
            prob_map = np.zeros([len(oris),stim.shape[0],stim.shape[1]])
            for oi,ori in enumerate(oris):

                # prob_map[oi]=scipy.ndimage.convolve(stim,Filters().gabor(theta=ori))

                #plt.imshow(saliency);plt.show()
                particle_pos = self.weighted_sample(saliency,n_part)
                # ch=np.zeros(stim.shape)
                # for p in particle_pos:
                    # ch[p[0],p[1]] = 1
                # if t==1:plt.imshow(ch,cmap='gray');plt.show()

                edge_map = self.detect_edges(stim,particle_pos,ori= ori)
#                plt.imshow(edge_map,cmap='gray');plt.show()
                assoc_field = Filters().association_field().ernst(beta=-2*ori)
#                plt.imshow(assoc_field,cmap='gray');plt.show()
                prob_map[oi] = self.probmap(edge_map, assoc_field)
#                plt.imshow(prob_map[oi],cmap='gray');plt.show()
            saliency = np.sum(prob_map,axis=0)
            saliency /= np.sum(saliency)
        plt.imshow(saliency,cmap='gray');plt.colorbar();plt.show()


#            plt.imshow(stim,cmap='gray');plt.show()




def proximity_rows():
    dot = visual.Circle(win, radius = .03)
    for i in range(5):
        for j in range(2):
            dot.setPos([.1*(i-2),.2*j])
            dot.draw()


def run():
    g = Model()
    g.init_gaussian_filters([7,9,11,13,15], 12)

    win = visual.Window(size = (256,256))
    proximity_rows()


    win.getMovieFrame(buffer='back')
    # win.flip()
    # core.wait(1)
    win.clearBuffer()
    stim = win.movieFrames[0]
    stim = np.asarray(stim.convert('L'))*1.
    win.movieFrames = []
    win.close()

    S1resp = g.S1resp_zeropad(stim)
    plt.imshow(np.sum(np.sum(S1resp,axis=3),axis=2))
    plt.show()


if __name__ == "__main__":
    g = Model()
    g.run_thres()
    #Filters.association_field().plot()
    #Filters().plot()
