#    |               |  | |
#  __|   __   __   __|  | |  _   ,_
# /  |  /  \_/  \_/  |  |/  |/  /  |
# \_/|_/\__/ \__/ \_/|_/|__/|__/   |_/
#

# "Doodle Labeller"
#
# > Daniel Buscombe, Marda Science daniel@mardascience.com
# > USGS Pacific Marine Science Center
#
# Based on the old "dl-tools" labelling code (https://github.com/dbuscombe-usgs/dl_tools/tree/master/create_groundtruth)
# > Incorporating some of the code contribution from LCDR Brodie Wells, Naval Postgraduate school Monterey

import subprocess, ctypes
import os, json, gc
import sys, getopt
import cv2
import numpy as np
from glob import glob
import rasterio, tifffile

##  progress bar (bcause te quiero demasiado ...)
from tqdm import tqdm
from scipy.stats import mode as md

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
from PIL import Image
from skimage.filters.rank import median
from skimage.morphology import disk, erosion

# =========================================================
def DoCrf(file, config, name):
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
                            data['label'],
                            config['classes'], config['fact'])

    if np.all(res)==254:
       res *= 0

    return res, p, preds

# =========================================================
def getCRF(img, Lc, label_lines, fact):
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

    # initial parameters
    n_iter = 10
    scale = 1+(10 * (np.array(img.shape).max() / 3681))

    compat_col = config['compat_col'] #20
    theta_col = config['theta_col'] #20
    theta_spat = 3
    prob = 0.9
    compat_spat = 3

    search = [.25,.5,1,2,4]
    #search = [1/8,1/4,1/2,1,2,4,8]

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
          d.addPairwiseGaussian(
                         sxy=(theta_spat, theta_spat),
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

          d.addPairwiseEnergy(feats, compat=compat_col,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
          Q = d.inference(n_iter)
          #print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
          R.append(1+np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8))

          preds = np.array(Q, dtype=np.float32).reshape((len(label_lines)+1, H, W)).transpose(1, 2, 0) ##labels+1
          P.append(preds)

          del Q

       ##res = np.round(np.median(R, axis=0))

       R = list(R)

       preds = np.median(P, axis=0)
       del P

       res, cnt = md(np.asarray(R, dtype='uint8'),axis=0)
       res = np.squeeze(res)
       cnt = np.squeeze(cnt)
       p = cnt/len(R)

       del cnt, R	   

       if fact>1:
          res = np.array(Image.fromarray(res.astype(np.uint8)).resize((Worig, Horig),
                resample=1))

       if config['medfilt']=="true":
          ## median filter to remove remaining high-freq spatial noise (radius of N pixels)
          N = np.round(11*(Worig/(3681))).astype('int') #11 when ny=3681
          print("Applying median filter of size: %i" % (N))
          res = median(res.astype(np.uint8), disk(N))

       if len(label_lines)==2:
           N = np.round(11*(Worig/(3681))).astype('int') #11 when ny=3681
           res = erosion(res, disk(N))

    return res, p, preds




# =========================================================
class MaskPainter():
    def __init__(self, image, config, screen_size):

        if config['im_order']=='RGB':
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif config['im_order']=='BGR':
            self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.config = config
        self.screen_size = screen_size
        self.class_mask = np.zeros((self.image.shape[0],
                                   self.image.shape[1],
                                   int(len(self.config['classes']) + 1)), dtype=np.uint8)
        self.mask_copy = self.class_mask.copy()
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
            return new_im
        else:
            return src

    def LabelWindow(self):
        print("Initial brush width = %i" % (config['lw']))
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
                                      :].copy()
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
            cv2.moveWindow('whole image', 0, 28)
            nav = False   # Navigator variable
            if not lab:
                counter = 1   # Label number
            sm = 0        # Enhancement variable
            if 2 > 1:
                s = np.shape(self.im_sect[:, :, 2])
                if not lab:
                    # TODO: Lc should never be set to zeros!!
                    #      It needs to get from class mask, so that it can
                    #      keep the labels that have been done
                    Lc = np.zeros((s[0], s[1],
                                   len(self.config['classes']) + 1))
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
                    cv2.moveWindow(label, 0, 28)  # Move it to (0,0)
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
      appropriate step size. Some tif images are are over 30,000 pixels
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

    with rasterio.Env():
       # change the band count to 1, set the
       # dtype to uint8, and specify LZW compression.
       profile.update(
          dtype=rasterio.uint8,
          count=1,
          compress='lzw')

       with rasterio.open(image_path, 'w', **profile) as dst:
          dst.write(lab.astype(rasterio.uint8), 1) #1=1 band


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
       resr[resr==0] = 2 #2 is null class
       resr = resr-1
       cv2.imwrite(outfile,
                np.round(255*((resr)/np.max(resr))).astype('uint8'))

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
                     cmap=plt.cm.Dark2,
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



#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:c:f:")
    except getopt.GetoptError:
        print('python doodler.py -c configfile.json [-f npy_file.npy]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python doodler.py -c \
                   path/to/config.json [-f path/to/npy_file.npy] ')
            sys.exit()
        elif opt in ("-c"):
            configfile = arg
        elif opt in ("-f"):
            npy_file = arg

    # if no npy file is given to the program, set to None
    if 'npy_file' not in locals():
        npy_file = None
    else:
        doodles = np.load(npy_file)

    #configfile = 'config.json'
    # load the user configs
    with open(os.getcwd()+os.sep+configfile) as f:
        config = json.load(f)
        
    # for k in config.keys():
    #     exec(k+'=config["'+k+'"]')

    ## TODO: add error checking on config item dtypes

    ## add defaults for missing items
    if "num_bands" not in config:
       config['num_bands'] = 3
    if "create_gtiff" not in config:
       config['create_gtiff'] = False
    if "alpha" not in config:
       config['alpha'] = 0.5
    if "apply_mask" not in config:
       config['apply_mask'] = None
    if "fact" not in config:
       config['fact'] = 5
    if "compat_col" not in config:
       config['compat_col'] = 20
    if "theta_col" not in config:
       config['theta_col'] = 20
    if "medfilt" not in config:
       config['medfilt'] = "true"
    if "thres_size_1chunk" not in config:
       config['thres_size_1chunk'] = 10000

    #the program needs two classes
    if len(config['classes'])==1:
       print(
       "You must have a minimum of 2 classes, i.e. 1) object of interest \
       and 2) background ... program exiting"
       )
       sys.exist(2)

    class_str = '_'.join(config['classes'].keys())

    ## TODO: add error checking on files items to check if files exist

    files = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*.*')))

    N = []
    if 'doodles' in locals():
       for f in files:
          N.append(os.path.splitext(f)[0].split(os.sep)[-1])

    ## TODO: add error checking on apply-mask items to check if files exist

    # masks are binary labels where the null class is zero
    masks = []
    mask_names = []
    if config["apply_mask"]!='None':
       if type(config["apply_mask"]) is str:
          to_search = glob(config['label_folder']+os.sep+'*'+\
                      config["apply_mask"]+'*label.png')
          for f in to_search:
             tmp, profile = OpenImage(f, None, config['num_bands'])
             tmp = (tmp[:,:,0]==0).astype('uint8')
             masks.append(tmp)
             mask_names.append(f)
             del tmp
       elif type(config["apply_mask"]) is list:
          for k in config["apply_mask"]:
             to_search = glob(config['label_folder']+os.sep+'*'+k+'*label.png')

             for f in to_search:
                tmp, profile = OpenImage(f, None, config['num_bands'])
                if len(np.unique(tmp))==2:
                   tmp = (tmp[:,:,0]==0).astype('uint8')
                else:
                   #assumes the 'null' or masking class is all but the last
                   tmp = (tmp[:,:,0]!=np.max(tmp[:,:,0])).astype('uint8')
                masks.append(tmp)
                mask_names.append(f)
                del tmp

    ## TODO: add error checking on apply-mask items to check if label folder is valid

    if 'doodles' not in locals(): #make annotations

       ##cycle through each file in turn
       for f in tqdm(files):

          screen_size = Screen()
          o_img, profile = OpenImage(f, config['im_order'], config['num_bands'])

          if np.std(o_img[o_img>0]) < 10:
              o_img[o_img==0] = np.min(o_img[o_img>0])
              o_img = o_img-o_img.min()
              o_img[o_img > (o_img.mean() + 2*o_img.std())] = o_img.mean()
              o_img = np.round(rescale(o_img,1.,255.)).astype(np.uint8)

          if masks:
              use = [m for m in mask_names if \
                   os.path.normpath(m).startswith(os.path.splitext(f)[0].replace('images', 'label_images'))]

              for u in use:
                 ind = [i for i in range(len(mask_names)) if mask_names[i]==u][0]
                 o_img[masks[ind]==1] = 255


          ##name = f.split(os.sep)[-1].split('.')[0] #assume no dots in file name
          name = os.path.splitext(f)[0].split(os.sep)[-1]
          N.append(name)

          mp = MaskPainter(o_img.copy(), config, screen_size)
          out, Z = mp.LabelWindow()

          out = out.astype(np.uint8)
          nx, ny = np.shape(out)
          gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))

          counter = 0
          for ind in Z:
             np.savez(config['label_folder']+os.sep+name+"_tmp"+str(counter)+"_"+class_str+".npz",
                    label=out[ind[0]:ind[1], ind[2]:ind[3]],
                    image=o_img[ind[0]:ind[1], ind[2]:ind[3]],
                    grid_x=gridx[ind[0]:ind[1], ind[2]:ind[3]],
                    grid_y=gridy[ind[0]:ind[1], ind[2]:ind[3]])
             counter += 1

          del Z, gridx, gridy, o_img
          outfile = config['label_folder']+os.sep+name+"_"+class_str+".npy"
          np.save(outfile, out)
          print("annotations saved to %s" % (outfile))
          del out

       print("Sparse labelling complete ...")
    else:
       print("Using provided labels ...")

    gc.collect()
    print("Dense labelling ... this may take a while")

    # cycle through each image root name, stored in N
    for name in N:
        print("Working on %s" % (name))

        if 'doodles' not in locals(): #make annotations

           # get a list of the temporary npz files
           label_files = sorted(glob(config['label_folder']+os.sep+name +'*tmp*'+\
                         class_str+'.npz'))
           print("Found %i image chunks" % (len(label_files)))

           #load data to get the size
           l = sorted(glob(config['label_folder']+os.sep+name +'*'+\
               class_str+'.npy'))[0]
           l = np.load(l)
           nx, ny = np.shape(l)
           del l
           apply_fact = (nx > config['thres_size_1chunk']) or \
                     (ny > config['thres_size_1chunk'])

           if apply_fact: #imagery is large and needs to be chunked

              # in parallel, get the CRF prediction for each tile
              # use all but 1 core (njobs=-2) and use pre_dispatch and max_nbytes to control memory usage
              o = Parallel(n_jobs = -2, verbose=1, pre_dispatch='2 * n_jobs', max_nbytes=1e6)\
                       (delayed(DoCrf)(label_files[k], config, name) for k in range(len(label_files)))

              ims, probs, preds = zip(*o)
              del o

              # load the npy file and get the grid coordinates to assign elements of 'ims' below
              l = sorted(glob(config['label_folder']+os.sep+name +'*'+class_str+'.npy'))[0]
              l = np.load(l)
              nx, ny = np.shape(l)
              del l

              # get image file and read it in with the profile of the geotiff, if appropriate
              imfile = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*'+name+'*.*')))[0]
              img, profile = OpenImage(imfile, config['im_order'], config['num_bands'])

              # ==========================================
			  # allocate an empty label image
              resr = np.zeros((nx, ny))

              # assign the CRF tiles to the label image (resr) using the grid positions
              for k in range(len(o)):
                 l = np.load(label_files[k])
                 resr[l['grid_x'], l['grid_y']] =ims[k]#+1
              del ims, l

              #============================================
			  # allocate an empty probablity image
              prob = np.zeros((nx, ny))

              # assign the CRF tiles to the label image (resr) using the grid positions
              for k in range(len(o)):
                 l = np.load(label_files[k])
                 prob[l['grid_x'], l['grid_y']] =probs[k]#+1
              del probs, l


           else: #image is small enough to fit on most memory at once
              print("Imagery is small (<%i px in all dimensions), so not using chunks - they will be deleted", (config['thres_size_1chunk']))
              # get image file and read it in with the profile of the geotiff, if appropriate
              imfile = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*'+name+'*.*')))[0]
              img, profile = OpenImage(imfile, config['im_order'], config['num_bands'])
              l = sorted(glob(config['label_folder']+os.sep+name +'*'+class_str+'.npy'))[0]
              l = np.load(l).astype(np.uint8)

              resr, prob, preds = getCRF(img,l,config['classes'], config['fact'])
              #resr += 1
              del l

        else: #labels provided
           print("computing CRF using provided annotations")

           #nx, ny = np.shape(doodles)
           ## memory management
           #if np.max((nx, ny))>2000:
           #    config['fact'] = np.max((config['fact'], 3))

           # get image file and read it in with the profile of the geotiff, if appropriate
           imfile = sorted(glob(os.path.normpath(config['image_folder']+\
                       os.sep+'*'+name+'*.*')))[0]
           img, profile = OpenImage(imfile, config['im_order'], config['num_bands'])

           resr, prob, preds = getCRF(img,doodles.astype(np.uint8),
                            config['classes'], config['fact']) # mu_col, mu_spat, prob


        outfile = config['label_folder']+os.sep+name+"_probs_per_class.npy"
        np.save(outfile, preds)
        del preds
		
        prob = np.array(Image.fromarray(prob).resize(tuple(np.array(resr.shape)), resample=1))

        gc.collect()

        # if 3-band image, mask out pixels that are [254, 254, 254]
        if np.ndim(img)==3:
           resr[np.sum(img,axis=2)==(254*3)] = 0
           resr[np.sum(img,axis=2)==0] = 0
           resr[np.sum(img,axis=2)==(255*3)] = 0
           prob[resr==0] = 0		   
        elif np.ndim(img)==2: #2-band image
           resr[img==0] = 0 #zero out image pixels that are 0 and 255
           resr[img==255] = 0
           prob[resr==0] = 0		   

        if masks:
              use = [m for m in mask_names if \
                   os.path.normpath(m).startswith(os.path.splitext(f)[0].replace('images', 'label_images'))]
            for u in use:
               ind = [i for i in range(len(mask_names)) if mask_names[i]==u][0]
               resr[masks[ind]==1] = 0
               prob[masks[ind]==1] = 0

        # plot the result and make label image files
        PlotAndSave(img.copy(), resr, prob, name, config, class_str, profile)

        del resr, img, prob

        # #remove the temporary npz files
        # for f in label_files:
        #    os.remove(f)
