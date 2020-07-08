

import subprocess, ctypes
import os, json, gc
import sys, getopt
import cv2
import numpy as np
from glob import glob
import rasterio, tifffile

##  progress bar (bcause te quiero demasiado ...)
from tqdm import tqdm
#from scipy.stats import mode as md

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
from PIL import Image

#from skimage.filters.rank import entropy
from skimage.filters.rank import median
from skimage.morphology import disk, erosion

cv2.setUseOptimized(True)

# =========================================================
def DoCrf(file, config, name, optim):
    """
    Loads imagery and labels from npz file, and calls getCRF
    Input:
        file:
		name:
		config:
    Output:
        res:
    """
    data = np.load(file)

    #img = data['image']
    #Lc = data['label']
    #num_classes = len(config['classes'])
    #fact = config['fact']

    res, p, preds = getCRF(data['image'],
                            data['label'], config, optim)

    if np.all(res)==254:
       res *= 0

    return res, p, preds


# =========================================================
def getCRF(img, Lc, config, optim):
    """
    Uses a dense CRF model to refine labels based on sparse labels and underlying image
    Input:
        img: 3D ndarray image
		Lc: 2D ndarray label image (sparse or dense)
		label_lines: list of class names
	Hard-coded variables:
        kernel_bilateral: DIAG_KERNEL kernel precision matrix for the colour-dependent term
            (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
        normalisation_bilateral: NORMALIZE_SYMMETRIC normalisation for the colour-dependent term
            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
        kernel_gaussian: DIAG_KERNEL kernel precision matrix for the colour-independent
            term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
        normalisation_gaussian: NORMALIZE_SYMMETRIC normalisation for the colour-independent term
            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
    Output:
        res : CRF-refined 2D label image
    """

    if optim is True:
       search = [.25,.33,.5,.75,1,1.33,2,3,4]
       prob = 0.8
    else:
       search = [.25,.5,1,2,4]
       prob = 0.9


    label_lines = config['classes']
    fact = config['fact']

    # initial parameters
    n_iter = 10
    scale = 1+(10 * (np.array(img.shape).max() / 3681))

    compat_col = config['compat_col'] #20
    theta_col = config['theta_col'] #20
    theta_spat = 3
    prob = 0.9
    compat_spat = 3

    #search = [.5,1,2,4,6]
    #search = [1/8,1/4,1/2,1,2,4,8]

    ## relative frequency of annotations
    rel_freq = np.bincount(Lc.flatten())#, minlength=len(label_lines)+1)
    rel_freq[0] = 0
    rel_freq = rel_freq / rel_freq.sum()
    rel_freq[rel_freq<1e-4] = 1e-4
    rel_freq = rel_freq / rel_freq.sum()


    if np.mean(img)<1:
       H = img.shape[0]
       W = img.shape[1]
       res = np.zeros((H,W)) * np.mean(Lc)

    else:

       # if image is 2D, make it 3D by stacking bands
       if np.ndim(img) == 2:
          # decimate by factor first
          img = img[::fact,::fact, :]
          img = np.dstack((img, img, img))

       # get original image shapes
       Horig = img.shape[0]
       Worig = img.shape[1]

       # decimate by factor by taking only every other row and column
       img = img[::fact,::fact, :]
       # do the same for the label image
       Lc = Lc[::fact,::fact]

       # get the new shapes
       H = img.shape[0]
       W = img.shape[1]

       U = unary_from_labels(Lc.astype('int'),
                          len(label_lines) + 1,
                          gt_prob=prob)

       R = []; P = []

       ## loop through the 'theta' values (half, given, and double)
       for mult in tqdm(search):
          d = dcrf.DenseCRF2D(H, W, len(label_lines) + 1)
          d.setUnaryEnergy(U)

          # to add the color-independent term, where features are the locations only:
          d.addPairwiseGaussian(sxy=(theta_spat, theta_spat),
                         compat=compat_spat,
                         kernel=dcrf.DIAG_KERNEL,
                         normalization=dcrf.NORMALIZE_SYMMETRIC)

          feats = create_pairwise_bilateral(
                                      sdims=(theta_col*mult, theta_col*mult),
                                      schan=(scale,
                                             scale,
                                             scale),
                                      img=img,
                                      chdim=2)

          d.addPairwiseEnergy(feats, compat=compat_col,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
          Q = d.inference(n_iter)
          #print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
          R.append(1+np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8))

          preds = np.array(Q, dtype=np.float32).reshape((len(label_lines)+1, H, W)).transpose(1, 2, 0) ##labels+1
          P.append(preds)

          del Q

       ##res = np.round(np.median(R, axis=0))

       R = list(R)

       # search, X, Y, num classes

       if len(label_lines)==2: #<len(search):
          try:
             preds = np.average(P, axis=-1, weights = 1/rel_freq)
          except:
             print("using unweighted average")
             preds = np.median(P, axis=-1)

          probs_per_class = np.median(P, axis=0)

       elif np.asarray(P).shape[0] > np.asarray(P).shape[-1]:
          preds = np.median(np.asarray(P).T, axis=-1).T
          probs_per_class = np.median(P, axis=0)

       else:
          try:
             preds = np.average(np.asarray(P), axis=0, weights = 1/rel_freq)
          except:
             print("using unweighted average")
             preds = np.median(np.asarray(P), axis=-1).T

          probs_per_class = np.median(P, axis=-1)

       del P

       #class is the last channel
       if 1+len(label_lines) != np.asarray(probs_per_class).shape[-1]:
          probs_per_class = np.swapaxes(probs_per_class,0,-1)

       if 1+len(label_lines) != np.asarray(preds).shape[-1]:
          preds = np.swapaxes(preds,0,-1)

       res = np.zeros((H,W))
       counter = 1
       for k in range(len(label_lines)):
         if (len(label_lines)==2):
            res[preds[k,:,:]>=(1/len(label_lines) )] = counter
         else:
            res[preds[:,:,k]>= .5] = counter
         counter += 1

       # _, cnt = md(np.asarray(R, dtype='uint8'),axis=0)
       # res2 = np.squeeze(_)
       # cnt = np.squeeze(cnt)
       # p = cnt/len(R)

       if len(label_lines)==2: #<len(search):
          p = 1-(np.std(preds, axis=0)/len(label_lines))
       else:
          p = np.max(preds, axis=-1)##*len(label_lines) 1-(np.std(preds, axis=-1)/len(label_lines))


       del R, preds


       if fact>1:
          res = np.array(Image.fromarray(1+res.astype(np.uint8)).resize((Worig, Horig),
                resample=1))-1
          if len(label_lines)==2:
             res[res>2] = 0
             res[res==1] = 0
          else:
             res[res>len(label_lines)] = 0
             res[res<0] = 0

          p = np.array(Image.fromarray((100*p).astype(np.uint8)).resize((Worig, Horig),
                resample=1))/100
          p[p>1]=1
          p[p<0]=0

          tmp = np.zeros((Horig,Worig,probs_per_class.shape[-1]))
          for k in range(probs_per_class.shape[-1]):
             tmp[:,:,k] = np.array(
                            Image.fromarray((100*probs_per_class[:,:,k]).astype(np.uint8)).resize((Worig, Horig),
                            resample=1)
                            )/100
          del probs_per_class
          probs_per_class = tmp.copy().astype('float16')
          del tmp
          probs_per_class[probs_per_class>1] = 1
          probs_per_class[probs_per_class<0]= 0

       if (config['medfilt']=="true") or (len(label_lines)==2):
          ## median filter to remove remaining high-freq spatial noise (radius of N pixels)
          N = np.round(5*(Worig/(3681))).astype('int') #11 when ny=3681
          print("Applying median filter of size: %i" % (N))
          res = median(res.astype(np.uint8), disk(N))

       #if len(label_lines)==2:
           # N = np.round(5*(Worig/(3681))).astype('int')
           # res = ~erosion(~res, disk(N))

    return res, p, probs_per_class

# =========================================================
class MaskPainter():
    def __init__(self, image, config, screen_size): #, im_mean):

        if config['im_order']=='RGB':
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #image
        elif config['im_order']=='BGR':
            self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #image

        self.config = config
        self.screen_size = screen_size
        #self.im_mean = im_mean
        self.class_mask = np.zeros((self.image.shape[0],
                                   self.image.shape[1],
                                   int(len(self.config['classes']) + 1)), dtype=np.uint8)
        #self.mask_copy = self.class_mask.copy()
        self.size = self.config['lw']
        self.current_x = 0
        self.current_y = 0

    def WinScales(self, imshape):
        dim = None
        (height, width) = self.screen_size
        (h, w) = imshape
        rh = float(height) / float(h)
        rw = float(width) / float(w)
        rf = min(rh, rw) * self.config['ref_im_scale']
        dim = (int(w * rf), int(h * rf))
        #print("screen size =", self.screen_size)
        #print("im_shape =", imshape)
        #print("min(rh, rw) =", min(rh, rw))
        #print("dim =", dim)
        return dim

    def MakeWindows(self):
        """
        Returns a matrix of x and y values to split the image at

        Returns
        -------
        Z : array
            x and y coordinates (x_min, y_min, x_max, y_max)
            of the sections of the whole image to be labeled
        """
        num_x_steps, num_y_steps = StepCalc(self.image.shape,
                                            self.config['max_x_steps'],
                                            self.config['max_y_steps'])
        ydim, xdim = self.image.shape[:2]
        x_stride = xdim/num_x_steps
        y_stride = ydim/num_y_steps
        for i in range(num_y_steps):
            for j in range(num_x_steps):
                if i == 0 and j == 0:
                    Z = np.array([0, np.int(y_stride), 0, np.int(x_stride)])
                else:
                    Z = np.vstack((Z, [np.int(y_stride * i),
                                       np.int(y_stride * (i + 1)),
                                       np.int(x_stride * j),
                                       np.int(x_stride * (j + 1))]))
        return Z


    def AnnoDraw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            self.current_x, self.current_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                cv2.line(self.im_sect, (self.current_x,
                                        self.current_y),
                         (x, y),
                         (0, 0, 255), param)
                self.current_x = x
                self.current_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False

        return x, y

    def Overlay(self, src, overlay):
        """
        Returns a new image to display, after blending in the pixels that have
            been labeled

        Inputs:
        src : array
            Original image
        overlay : array
            Overlay image, in this case the Labels

        Outputs:
        -------
        new_im : array
            Blended im
        """
        if np.max(overlay) > 0:
            new_im = src.copy()
            vals = np.argmax(overlay, axis=2)
            vals *= 18
            new_im[:, :, 0][vals > 0] = vals[vals > 0]
            return new_im.astype(np.uint8)
        else:
            return src

    def LabelWindow(self):
        print("Initial brush width = %i" % (self.config['lw']))
        print("  decrease using the [1] key")
        print("  increase using the [2] key")
        print("Cycle classes with [ESC] key")
        print("Go back a frame with [b] key")
        print("Skip a frame with [s] key")
        print("Undo the current annotation with [z] key")

        self.draw = False  # True if mouse is pressed
        self.Z = self.MakeWindows()
        lab = False
        ck = 0

        while ck < len(self.Z):
            ref_img = self.image.copy()
            if ck < 1:
                ck = 0
            self.im_sect = self.image[self.Z[ck][0]:self.Z[ck][1],
                                      self.Z[ck][2]:self.Z[ck][3],
                                      :].copy().astype(np.uint8)
            # FIX: See below
            self.im_sect[self.im_sect[:, :, 2] == 255] = 254
            cv2.rectangle(ref_img,
                          (self.Z[ck][2], self.Z[ck][0]),
                          (self.Z[ck][3], self.Z[ck][1]),
                          (255, 255, 0), 20)

            cv2.namedWindow('whole image', cv2.WINDOW_NORMAL)
            cv2.imshow('whole image', ref_img)
            cv2.resizeWindow('whole image',
                             (self.WinScales(ref_img.shape[:2])))
            cv2.moveWindow('whole image', 0, 0)
            nav = False   # Navigator variable
            if not lab:
                counter = 1   # Label number

            #if 2 > 1:
            s = np.shape(self.im_sect[:, :, 2])
            if not lab:
                # TODO: Lc should never be set to zeros!!
                #      It needs to get from class mask, so that it can
                #      keep the labels that have been done
                Lc = np.zeros((s[0], s[1],
                               len(self.config['classes']) + 1), dtype=np.uint8)
            else:
                Lc[counter] = Lc[counter] * 0
            while counter <= len(self.config['classes']):
                label = list(self.config['classes'].keys())[counter - 1]
                if nav:
                    break
                imcopy = self.im_sect.copy()
                cv2.namedWindow(label, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(label,
                                 tuple(self.WinScales(imcopy.shape[:2])))
                cv2.moveWindow(label, 0, 0)  # Move it to (0,0)
                cv2.setMouseCallback(label, self.AnnoDraw, self.size)
                while(1):
                    showim = self.Overlay(self.im_sect, Lc)
                    cv2.imshow(label, showim)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("z"):     # If Z is pressed
                        self.im_sect = imcopy.copy()
                        imcopy = self.im_sect.copy()
                    if k == 27:  # If ESC is pressed, copy labeled pixels
                        #           to Lc and go to next label
                        # TODO: Find a better way to extract the drawing
                        #       inputs Clouds often have a 255 so I made
                        #       everything that was originally 255 in
                        #       blue band == 254
                        try:
                            # This changes the section of the image that
                            #   was drawn on, works well, but if you want
                            #   to go back a label, I don't have a way to
                            #   do that currently
                            Lc[:,
                               :,
                               counter][self.im_sect[:,
                                                     :,
                                                     2] == 255] = counter
                            self.im_sect[self.im_sect[:,
                                                      :,
                                                      2] == 255] = 160
                        except:
                            Lc[:, :, counter][self.im_sect == 255] = \
                                counter
                            self.im_sect[self.im_sect == 255] = 160
                        counter += 1
                        break

                    if k == ord("s"):  # If s is pressed, skip square
                        nav = True
                        break

                    if k == ord('b'):  # If b is pressed go back a square
                        nav = True
                        ck -= 2
                        break

                    if k == ord('2'):  # If 2 is pressed, increase brush wi
                        self.size += 1
                        print("brush width = " + str(self.size))

                    if k == ord('1'):  # If 1 is pressed, decrese brush wid
                        self.size -= 1
                        if self.size < 1:
                            self.size = 1
                        print("brush width = " + str(self.size))

                del showim, imcopy
                cv2.destroyWindow(label)

            if not nav:
                self.class_mask[self.Z[ck][0]:self.Z[ck][1],
                                self.Z[ck][2]:self.Z[ck][3], :] = Lc
                lab = False

            cv2.destroyWindow('whole image')
            ck += 1
        return np.argmax(self.class_mask, axis=2), self.Z

# =========================================================
def Screen():
    """
    gets the screen size.
    Input:
        nothing
    Output:
        screen_size : tuple of screen size
    """

    # get screen size
    if os.name == 'posix':  # true if linux/mac or cygwin on windows
        cmd = ['xrandr']
        cmd2 = ['grep', '*']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
        p.stdout.close()
        resolution_string, junk = p2.communicate()
        resolution = resolution_string.split()[0].decode("utf-8")
        width, height = resolution.split('x')
        screen_size = tuple((int(height), int(width)))
    else:  # windows
        user32 = ctypes.windll.user32
        screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return screen_size

# =========================================================
def OpenImage(image_path, im_order, num_bands):
    """
    Returns the image in numpy array format
    Input:
        image_path : string
            Full or relative path to image
        config : dict
            Dictionary of parameters set in the parameters file
    Output:
        numpy array of image 2D or 3D or 4+D
    """

    if (image_path.lower()[-3:] == 'tif') | (image_path.lower()[-4:] == 'tiff'):
        if num_bands>3:
           try: #4+ band tif file
              img = tifffile.imread(image_path)
              profile = None
           except: ##<=3 band tif file
              img = cv2.imread(image_path)
              if im_order=='RGB':
                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              elif im_order=='BGR':
                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
              profile = None
        else: #3-band geotiff
           img, profile = ReadGeotiff(image_path, im_order)
    else: ###<=3 band non-tif file
        img = cv2.imread(image_path)
        if im_order=='RGB':
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif im_order=='BGR':
           img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        profile = None
    return np.squeeze(img), profile


#===============================================================
def StepCalc(im_shape, max_x_steps=None, max_y_steps=None):
    """
    SplitFlag figures out what size of image it's dealing and sets an
      appropriate step size. Some tif images are over 30,000 pixels
      in one direction.

    Parameters
    ----------
    im_shape : TYPE
        Tuple of image shape

    Returns
    -------
    num_x_steps : TYPE
        how many windows are needed in the x direction
    num_y_steps : TYPE
        how many windows are needed in the x direction
    """
    if max_x_steps == None:
        if im_shape[1] < 2000:
            num_x_steps = 2
        elif im_shape[1] < 4000:
            num_x_steps = 3
        elif im_shape[1] < 6000:
            num_x_steps = 4
        elif im_shape[1] < 10000:
            num_x_steps = 5
        else:
            num_x_steps = 6
    else:
        num_x_steps = max_x_steps

    if max_y_steps == None:
        if im_shape[0] < 2000:
            num_y_steps = 2
        elif im_shape[0] < 4000:
            num_y_steps = 3
        elif im_shape[0] < 6000:
            num_y_steps = 4
        elif im_shape[0] < 10000:
            num_y_steps = 5
        else:
            num_y_steps = 6
    else:
        num_y_steps = max_y_steps

    return num_x_steps, num_y_steps



#===============================================================
def WriteGeotiff(image_path, lab, profile):
    """
    This function writes a 1-band label image in GeoTIFF format.
    TODO: Fill in the doc string better
    Parameters
    ----------
    image_path : string
        full or relative path to the tiff image you'd like to create
    lab : 2D label raster
    profile: rasterio profile of the original geotiff image
    """

    try:
       with rasterio.Env():
          # change the band count to 1, set the
          # dtype to uint8, and specify LZW compression.
          profile.update(
             dtype=rasterio.uint8,
             count=1,
             compress='lzw')

          with rasterio.open(image_path, 'w', **profile) as dst:
             dst.write(lab.astype(rasterio.uint8), 1) #1=1 band

    except:
        print("Geotiff could not be written - was the input image a geotiff?")


#===============================================================
def ReadGeotiff(image_path, rgb):
    """
    This function reads image in GeoTIFF format.
    TODO: Fill in the doc string better
    Parameters
    ----------
    image_path : string
        full or relative path to the tiff image
    rgb : TYPE
        is it RGB or BGR
    Returns
    -------
    img : array
        2D or 3D numpy array of the image
    """
    with rasterio.open(image_path) as src:
        layer = src.read()
        profile = src.profile

    if layer.shape[0] == 3:
        r, g, b = layer
        if rgb == 'RGB':
            img = np.dstack([r, g, b])
        else:
            img = np.dstack([b, g, r])
    elif layer.shape[0] == 4:
        r, g, b, gd = layer
        if rgb == 'RGB':
            img = np.dstack([r, g, b])
        else:
            img = np.dstack([b, g, r])
    else:
        img = layer

    if np.max(img) > 255:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')

    img[img[:,:,2] == 255] = 254

    return img, profile


# =========================================================
def PlotAndSave(img, resr, prob, name, config, class_str, profile):
    """
    Makes plots and save label images
    Input:
        img:
		resr:
		name:
		config:
		class_str:
        profile: rasterio file profile (CRS, etc)
    Output:
        None
    """
    outfile = config['label_folder']+os.sep+name+"_"+class_str+'_label.png'

    if len(config['classes'])==2:
       tmp = np.floor(255*(1-(resr/np.max(resr)))).astype('uint8')
       cv2.imwrite(outfile,tmp)

       if config['create_gtiff']=='true':
          image_path = outfile.replace('.png','.tif')

          WriteGeotiff(image_path, tmp, profile)
       del tmp
    else:
       lab = np.round(255*(resr/len(config['classes'])))
       cv2.imwrite(outfile, lab.astype('uint8')) ##np.max(resr)

       if config['create_gtiff']=='true':
          image_path = outfile.replace('.png','.tif')

          WriteGeotiff(image_path, lab, profile)

       resr = resr.astype('float')
       resr[resr<1] = np.nan
       resr = resr-1

    outfile = config['label_folder']+os.sep+name+"_"+class_str+'_prob.png'

    if len(config['classes'])==2:
       prob[prob<1]= 1-prob[prob<1]

    cv2.imwrite(outfile,
                np.round(255*prob).astype('uint8'))

    if config['create_gtiff']=='true':
       image_path = outfile.replace('.png','.tif')
       WriteGeotiff(image_path, prob, profile)

    try:
       alpha_percent = config['alpha'] #0.75
    except:
       alpha_percent = 0.5

    cols = list(config['classes'].values())
    new_cols = []
    for col in cols:
        if not col.startswith('#'):
            col = '#'+col
        new_cols.append(col)
    cmap = colors.ListedColormap(new_cols)

    if len(config['classes'])==2:
        resr[resr==0] = 1
        resr[resr==2] = 0

    fig = plt.figure()
    ax1 = fig.add_subplot(111) #sp + 1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    if np.ndim(img)==3:
       _ = ax1.imshow(img)
    else:
       img = img.astype('float')
       img = np.round(255*(img/img.max()))
       img[img==0] = np.nan
       _ = ax1.imshow(img)

    im2 = ax1.imshow(resr,
                     cmap=cmap,
                     alpha=alpha_percent, interpolation='nearest',
                     vmin=0, vmax=len(config['classes']))
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%")
    cb=plt.colorbar(im2, cax=cax)
    cb.set_ticks(0.5 + np.arange(len(config['classes']) + 1))
    cb.ax.set_yticklabels(config['classes'], fontsize=6)

    outfile = config['label_folder']+os.sep+name+"_"+class_str+'_mres.png'

    plt.savefig(outfile,
                dpi=300, bbox_inches = 'tight')
    del fig; plt.close()


    fig = plt.figure()
    ax1 = fig.add_subplot(111) #sp + 1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    if np.ndim(img)==3:
       _ = ax1.imshow(img)
    else:
       img = img.astype('float')
       img = np.round(255*(img/img.max()))
       img[img==0] = np.nan
       _ = ax1.imshow(img)

    im2 = ax1.imshow(prob,
                     cmap=plt.cm.RdBu,
                     alpha=alpha_percent, interpolation='nearest',
                     vmin=0, vmax=1)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%")
    cb=plt.colorbar(im2, cax=cax)
    #cb.set_ticks(0.5 + np.arange(len(config['classes']) + 1))
    #cb.ax.set_yticklabels(config['classes'], fontsize=6)

    outfile = config['label_folder']+os.sep+name+"_"+class_str+'_prob_res.png'

    plt.savefig(outfile,
                dpi=300, bbox_inches = 'tight')
    del fig; plt.close()



#
# # =========================================================
# def getCRF_optim(img, Lc, config): #label_lines, fact):
#     """
#     Uses a dense CRF model to refine labels based on sparse labels and underlying image
#     Input:
#         img: 3D ndarray image
# 		Lc: 2D ndarray label image (sparse or dense)
# 		label_lines: list of class names
# 	Hard-coded variables:
#         kernel_bilateral: DIAG_KERNEL kernel precision matrix for the colour-dependent term
#             (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#         normalisation_bilateral: NORMALIZE_SYMMETRIC normalisation for the colour-dependent term
#             (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#         kernel_gaussian: DIAG_KERNEL kernel precision matrix for the colour-independent
#             term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#         normalisation_gaussian: NORMALIZE_SYMMETRIC normalisation for the colour-independent term
#             (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#     Output:
#         res : CRF-refined 2D label image
#     """
#
#     label_lines = config['classes']
#     fact = config['fact']
#
#     # initial parameters
#     n_iter = 10
#     scale = 1+(10 * (np.array(img.shape).max() / 3681))
#
#     compat_col = config['compat_col'] #20
#     theta_col = config['theta_col'] #20
#     theta_spat = 3
#     prob = 0.8
#     compat_spat = 3
#
#     search = [.25,.33,.5,.75,1,1.33,2,3,4]
#
#     ## relative frequency of annotations
#     rel_freq = np.bincount(Lc.flatten())#, minlength=len(label_lines)+1)
#     rel_freq[0] = 0
#     rel_freq = rel_freq / rel_freq.sum()
#     rel_freq[rel_freq<1e-4] = 1e-4
#     rel_freq = rel_freq / rel_freq.sum()
#
#
#     if np.mean(img)<1:
#        H = img.shape[0]
#        W = img.shape[1]
#        res = np.zeros((H,W)) * np.mean(Lc)
#
#     else:
#
#        # if image is 2D, make it 3D by stacking bands
#        if np.ndim(img) == 2:
#           # decimate by factor first
#           img = img[::fact,::fact, :]
#           img = np.dstack((img, img, img))
#
#        # get original image shapes
#        Horig = img.shape[0]
#        Worig = img.shape[1]
#
#        # decimate by factor by taking only every other row and column
#        img = img[::fact,::fact, :]
#        # do the same for the label image
#        Lc = Lc[::fact,::fact]
#
#        # get the new shapes
#        H = img.shape[0]
#        W = img.shape[1]
#
#        U = unary_from_labels(Lc.astype('int'),
#                           len(label_lines) + 1,
#                           gt_prob=prob)
#
#        R = []; P = []
#
#        ## loop through the 'theta' values (half, given, and double)
#        for mult in tqdm(search):
#           d = dcrf.DenseCRF2D(H, W, len(label_lines) + 1)
#           d.setUnaryEnergy(U)
#
#           # to add the color-independent term, where features are the locations only:
#           d.addPairwiseGaussian(sxy=(theta_spat, theta_spat),compat=compat_spat,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#           feats = create_pairwise_bilateral(
#                                       sdims=(theta_col*mult, theta_col*mult),
#                                       schan=(scale,
#                                              scale,
#                                              scale),
#                                       img=img,
#                                       chdim=2)
#
#           d.addPairwiseEnergy(feats, compat=compat_col,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
#           Q = d.inference(n_iter)
#           #print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
#           R.append(1+np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8))
#
#           preds = np.array(Q, dtype=np.float32).reshape((len(label_lines)+1, H, W)).transpose(1, 2, 0) ##labels+1
#           P.append(preds)
#
#           del Q
#
#        ##res = np.round(np.median(R, axis=0))
#
#        R = list(R)
#
#        try:
#           preds = np.average(P, axis=-1, weights = 1/rel_freq)
#        except:
#           print("using unweighted average")
#           preds = np.median(P, axis=-1)
#
#        del P
#
#        res = np.zeros((H,W))
#        counter = 1
#        for k in range(len(label_lines)):
#          res[preds[k,:,:]>=(1/len(label_lines) )] = counter
#          counter += 1
#
#        # _, cnt = md(np.asarray(R, dtype='uint8'),axis=0)
#        # res2 = np.squeeze(_)
#        # cnt = np.squeeze(cnt)
#        # p = cnt/len(R)
#
#        p = 1-(np.std(preds, axis=0)/len(label_lines))
#
#        del R #, cnt
#
#
#        if fact>1:
#           res = np.array(Image.fromarray(res.astype(np.uint8)).resize((Worig, Horig),
#                 resample=1))
#
#           p = np.array(Image.fromarray((100*p).astype(np.uint8)).resize((Worig, Horig),
#                 resample=1))/100
#           if len(label_lines)==2:
#              res[res==1] = 0
#
#           preds2 = np.zeros((preds.shape[0],Horig,Worig))
#           for k in range(preds.shape[0]):
#              preds2[k,:,:] = np.array(
#                             Image.fromarray((100*preds[:,:,k]).astype(np.uint8)).resize((Worig, Horig),
#                             resample=1)
#                             )
#           del preds
#           preds = preds2.copy()
#           del preds2
#           p[p>1] = 1
#           p[p<0]= 0
#           preds[preds>1] = 1
#           preds[preds<0]= 0
#
#        if (config['medfilt']=="true") or (len(label_lines)==2):
#           ## median filter to remove remaining high-freq spatial noise (radius of N pixels)
#           N = np.round(5*(Worig/(3681))).astype('int') #11 when ny=3681
#           print("Applying median filter of size: %i" % (N))
#           res = median(res.astype(np.uint8), disk(N))
#
#        #if len(label_lines)==2:
#            # N = np.round(5*(Worig/(3681))).astype('int')
#            # res = ~erosion(~res, disk(N))
#
#
#     return res, p, preds



# # =========================================================
# def getCRF_optim(img, Lc, num_classes, fact):
#     """
#     Uses a dense CRF model to refine labels based on sparse labels and underlying image
#     Input:
#         img: 3D ndarray image
# 		Lc: 2D ndarray label image (sparse or dense)
# 		label_lines: list of class names
# 	Hard-coded variables:
#         kernel_bilateral: DIAG_KERNEL kernel precision matrix for the colour-dependent term
#             (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#         normalisation_bilateral: NORMALIZE_SYMMETRIC normalisation for the colour-dependent term
#             (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#         kernel_gaussian: DIAG_KERNEL kernel precision matrix for the colour-independent
#             term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#         normalisation_gaussian: NORMALIZE_SYMMETRIC normalisation for the colour-independent term
#             (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#     Output:
#         res : CRF-refined 2D label image
#     """
#
#     # initial parameters
#     theta_col = config["theta_col"] ##20
#     theta_spat = 3
#     n_iter = 10
#
#     scale = 1+(10 * (np.array(img.shape).max() / 3681)) #20
#
#     prob = 0.8 #0.9
#     compat_col = config["theta_col"] ##20
#     compat_spat = 3
#
#     ## relative frequency of annotations
#     rel_freq = np.bincount(Lc.flatten())
#     rel_freq[0] = 0
#     rel_freq = rel_freq / rel_freq.sum()
#     rel_freq[rel_freq<1e-4] = 1e-4
#     rel_freq = rel_freq / rel_freq.sum()
#
#
#     #if the image or label is empty ...
#     if ((np.mean(img)<1.) or (np.std(Lc)==0.)):
#        H = img.shape[0]
#        W = img.shape[1]
#        res = np.zeros((H,W)) * np.mean(Lc)
#        return res, np.nan, np.nan
#
#     else:
#
#        thres = int(config['thres_size_1chunk']/2)
#
#        apply_fact = (img.shape[0] > thres) or (img.shape[1] > thres)
#        #print("apply_fact: %i" % (apply_fact))
#
#        # if image is 2D, make it 3D by stacking bands
#        if np.ndim(img) == 2:
#           # decimate by factor first
#           if apply_fact:
#              img = img[::fact,::fact, :]
#           img = np.dstack((img, img, img))
#
#        # get original image shapes
#        Horig = img.shape[0]
#        Worig = img.shape[1]
#
#        # decimate by factor by taking only every other row and column
#        if apply_fact:
#           img = img[::fact,::fact, :]
#           # do the same for the label image
#           Lc = Lc[::fact,::fact]
#
#        # get the new shapes
#        H = img.shape[0]
#        W = img.shape[1]
#
#        # define some factors to apply to the nominal thetas and mus
#        search = [1/4,1/3,1/2,3/4,1,4/3,2,3,4]
#        #search = [.25,.5,1,2,4]
#
#        R = [] #for label realization
#        #P = [] #for theta parameters
#        #K = [] #for kl-divergence statistic
#        P = []
#
#        mult_spat = 1
#
#        ## loop through the 'theta' values (quarter, half, given, and double, quadruple)
#        for mult_col_compat in tqdm(search):
#           for mult_spat_compat in [1]:
#              for mult_col in search:
#                 for mult_prob in [1]:
#
#                    #create (common to all experiments) unary potentials for each class
#                    U = unary_from_labels(Lc.astype('int'),
#                           num_classes + 1,
#                           gt_prob=np.min((mult_prob*prob,.999)))
#
#                    d = dcrf.DenseCRF2D(H, W, num_classes + 1)
#
#                    d.setUnaryEnergy(U)
#
#                    # to add the color-independent term, where features are the locations only:
#                    d.addPairwiseGaussian(sxy=(int(mult_spat*theta_spat),
#                                 int(mult_spat*theta_spat)), ##scaling per dimension (smoothness)
#                                 compat=mult_spat_compat*compat_spat, ##scaling per channel
#                                 kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#                    #spatial feature extractor
#                    feats = create_pairwise_bilateral(
#                                       sdims=(int(mult_col*theta_col), int(mult_col*theta_col)), #scaling per dimension
#                                       schan=(scale, #theta_beta = scaling per channel (smoothness)
#                                              scale,
#                                              scale),
#                                       img=img,
#                                       chdim=2)
#
#                    #color feature extractor
#                    d.addPairwiseEnergy(feats, compat=mult_col_compat*compat_col,
#                         kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#                    #do inference
#                    Q = d.inference(n_iter)
#
#                    #K.append(d.klDivergence(Q))
#
#                    R.append(np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8))
#                    #P.append([mult_col, mult_col_compat]) #mult_spat, mult_spat_compat, mult_prob
#
#                    preds = np.array(Q, dtype=np.float32).reshape((num_classes+1, H, W)).transpose(1, 2, 0) ##labels+1
#                    P.append(preds)
#
#                    del Q, d
#
#        ##res = np.round(np.median(R, axis=0))
#
#        R = list(R)
#
#        try:
#           preds = np.average(P, axis=-1, weights = np.tile((1/rel_freq), (num_classes+1, len(search) ) ))
#        except:
#           print("using unweighted average")
#           preds = np.median(P, axis=0)
#        del P
#
#        res = np.zeros((H,W))
#        counter = 1
#        for k in range(num_classes):
#            res[preds[:,:,k]>=(1/num_classes )] = counter
#            counter += 1
#
#        _, cnt = md(np.asarray(R, dtype='uint8'),axis=0)
#        #res2 = np.squeeze(res)
#        cnt = np.squeeze(cnt)
#        p = cnt/len(R)
#
#        del cnt, R, U
#
#        if apply_fact:
#           res = np.array(Image.fromarray(res.astype(np.uint8)).resize((Worig, Horig), \
#                          resample=1))
#           p = np.array(Image.fromarray((100*p).astype(np.uint8)).resize((Worig, Horig),
#                 resample=1))/100
#           preds = np.array(Image.fromarray((100*preds).astype(np.uint8)).resize((Worig, Horig),
#                resample=1))
#           p[p>1] = 1
#           p[p<0]= 0
#           preds[preds>1] = 1
#           preds[preds<0]= 0
#
#        res += 1 #avoid averaging over zero class
#
#        # median filter to remove remaining high-freq spatial noise (radius of N pixels)
#        N = np.round(5*(Worig/(7362/fact))).astype('int')
#        #print("median filter size: %i" % (N))
#        res = median(res.astype(np.uint8), disk(N))
#
#        if num_classes==2:
#            N = np.round(5*(Worig/(3681))).astype('int')
#            res = erosion(res, disk(N))
#
#     return res, mult_col*theta_col, mult_col_compat*compat_col, p, preds #, mult_prob*prob
#


#
# # =========================================================
# def DoCrf_optim(file, config, name):
#     """
#     Loads imagery and labels from npz file, and calls getCRF
#     Input:
#         file:
# 		name:
# 		config:
#     Output:
#         res:
#     """
#     data = np.load(file)
#
#     #img = data['image']
#     #Lc = data['label']
#     #num_classes = len(config['classes'])
#     #fact = config['fact']
#
#     res, p, preds = getCRF(data['image'],
#                             data['label'], config, True)
#
#     if np.all(res)==254:
#        res *= 0
#
#     return res, p, preds
#
