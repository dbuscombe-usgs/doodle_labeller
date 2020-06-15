#              o             o
#  __    _ _|_     _  _  _       __   _
# /  \_|/ \_|  |  / |/ |/ |  |  / / _|/
# \__/ |__/ |_/|_/  |  |  |_/|_/ /_/ |__/
#     /|                          /|
#     \|                          \|
#

# "optimize.py"
# Performs an optimization of CRF hyperparameters by minimizing per-class entropy
# > Daniel Buscombe, Marda Science daniel@mardascience.com
# > USGS Pacific Marine Science Center

# do not use this function yet - work in prpgress

import os, json, csv
import sys, getopt
import cv2
import numpy as np
from glob import glob
import rasterio

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import as_strided as ast
from math import gcd

from skimage.filters.rank import entropy
from skimage.morphology import disk

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

# =========================================================
def getCRF(img, Lc, num_classes):
    """
    Uses a dense CRF model to refine labels based on sparse labels and underlying image
    Input:
        img: 3D ndarray image
		Lc: 2D ndarray label image (sparse or dense)
		label_lines: list of class names
	Global parameters used:
		config['n_iter']: number of iterations of MAP inference.
		config['theta_col']: standard deviations for the location component of the colour-dependent term.
		config['theta_spat']: standard deviations for the location component of the colour-independent term.
		config['compat_col']: label compatibilities for the colour-dependent term
		config['compat_spat']: label compatibilities for the colour-independent term
		config['scale']: spatial smoothness parameter
		config['prob']: assumed probability of input labels
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

    if np.mean(img)<1:
       H = img.shape[0]
       W = img.shape[1]
       res = np.zeros((H,W)) * np.mean(Lc)

    else:

       if np.ndim(img) == 2:
          img = np.dstack((img, img, img))
       H = img.shape[0]
       W = img.shape[1]

       R = [] #for label realization
       P = [] #for theta parameters
       ## loop through the 'theta' values (quarter, half, given, and double, quadruple)
       for mult_col in [.25,.5,1,2,4]:
          for mult_spat in [.25,.5,1,2,4]:
             d = dcrf.DenseCRF2D(H, W, num_classes + 1)
             U = unary_from_labels(Lc.astype('int'),
                          num_classes + 1,
                          gt_prob=config['prob'])
             d.setUnaryEnergy(U)

             # to add the color-independent term, where features are the locations only:
             d.addPairwiseGaussian(sxy=(int(mult_spat*config['theta_spat']),
                                int(mult_spat*config['theta_spat'])),
                                compat=config['compat_spat'],
                                kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

             feats = create_pairwise_bilateral(
                                      sdims=(int(mult_col*config['theta_col']), int(mult_col*config['theta_col'])),
                                      schan=(config['scale'],
                                             config['scale'],
                                             config['scale']),
                                      img=img,
                                      chdim=2)

             d.addPairwiseEnergy(feats, compat=config['compat_col'],
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
             Q = d.inference(config['n_iter'])

             R.append(np.argmax(Q, axis=0).reshape((H, W)))
             P.append([mult_col, mult_spat])

       #get the entropy of the first channel with a 10-px disk structural element
       entr_img = entropy(img[:,:,0], disk(10))

       ## binary case
       if num_classes==2:
          #for each label image in R, sum the entropy in the R^1 (no object) class
          scores = [np.sum(entr_img*z) for z in R]
          del R
          #optimal parameters based on minimum summed entropy
          mult_col, mult_spat = P[np.argmin(scores)]
       else:
          #for each label image in R, sum the entropy in the R^1 (no object) class
          S = []
          for z in R:
              scores = [np.sum(entr_img*(z==i).astype('int')) for i in np.unique(z)]
              S.append(scores)
          S = np.sum(np.array(S), axis=0)
          del R
          #optimal parameters based on minimum summed entropy
          mult_col, mult_spat = P[np.argmin(S)]

       print("Optimal color theta hyperparameter: %f" % (mult_col))
       print("Optimal spatial theta hyperparameter: %f" % (mult_spat))

       # do the CRF again with optimal parameters
       d = dcrf.DenseCRF2D(H, W, num_classes + 1)
       U = unary_from_labels(Lc.astype('int'),
              num_classes + 1,
              gt_prob=config['prob'])
       d.setUnaryEnergy(U)

       # to add the color-independent term, where features are the locations only:
       d.addPairwiseGaussian(sxy=(int(mult_spat*config['theta_spat']),
                    int(mult_spat*config['theta_spat'])),
                    compat=config['compat_spat'],
                    kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

       feats = create_pairwise_bilateral(
                          sdims=(int(mult_col*config['theta_col']), int(mult_col*config['theta_col'])),
                          schan=(config['scale'],
                                 config['scale'],
                                 config['scale']),
                          img=img,
                          chdim=2)

       d.addPairwiseEnergy(feats, compat=config['compat_col'],
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC)
       Q = d.inference(config['n_iter'])

    return np.argmax(Q, axis=0).reshape((H, W)) #np.round(np.median(), axis=0))


# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')


# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape


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
    # TODO: I have not tested any of the rest of this project for one layer
    else:
        img = layer

    if np.max(img) > 255:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')

    img[img[:,:,2] == 255] = 254

    return img

#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:c:")
    except getopt.GetoptError:
        print('python optimize.py -c configfile.json')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python optimize.py -c config.json')
            sys.exit()
        elif opt in ("-c"):
            configfile = arg

    #configfile = 'config.json'
    # load the user configs
    with open(os.getcwd()+os.sep+configfile) as f:
        config = json.load(f)

    # for k in config.keys():
    #     exec(k+'=config["'+k+'"]')
    num_classes = len(config['classes'])

    if num_classes==1:
       print("You must have a minimum of 2 classes, i.e. 1) object of interest and 2) background")
       sys.exist(2)

    class_str = '_'.join(config['classes'].keys())

    files = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*.*')))
    names = [f.split(os.sep)[-1].split('.')[0] for f in files] #assumes no dots in file names

    for f, name in zip(files, names):

       ##read image
       img, profile = OpenImage(f, config['im_order'], config['num_bands'])

       outfile = config['label_folder']+os.sep+name+"_"+class_str+'_label.png'

       msk_flat = OpenImage(outfile, None)[:,:,0]
       msk_flat = np.round(msk_flat/255.)+1

       ##50% overlap between tiles
       overlap = 1.
       nx, ny, nz = np.shape(img)
       ##number of chunks in each dimension
       num_chunks = gcd(nx, ny)

       if (num_chunks==nx) | (num_chunks==ny):
          ## do this without windowing/overlap
          res = getCRF(img, msk_flat+1, num_classes)

       elif num_chunks>100:

          res = getCRF(img, msk_flat+1, num_classes)

       else:

          print("working on %i chunks" % (int(num_chunks)))

          ##size of each chunk
          sx = int(nx/num_chunks)
          # if (sx % 2) != 0: #odd
          #    sx = sx-1
          sy = int(ny/num_chunks)
          # if (sy % 2) != 0: #odd
          #    sy = sy-1
          ssx = int(overlap*sx)
          ssy = int(overlap*sy)

          ##gets small overlapped image windows
          Z, _ = sliding_window(img, (sx, sy, nz), (ssx, ssy, nz))
          del img
          ##gets small overlapped label windows
          L, _ = sliding_window(msk_flat, (sx, sy), (ssx, ssy))
          del msk_flat

          print("%i chunks" % (len(Z)))

          try:
             # in parallel, get the CRF prediction for each tile
             o = Parallel(n_jobs = -1, verbose=1, pre_dispatch='2 * n_jobs', max_nbytes=None)\
                         (delayed(getCRF)(Z[k], L[k], num_classes) for k in range(len(Z)))
          except:
             print("Something went wrong with parallel - trying 2 cores ...")
             o = Parallel(n_jobs = 2, verbose=1, pre_dispatch='2 * n_jobs', max_nbytes=None)\
                         (delayed(getCRF)(Z[k], L[k], num_classes) for k in range(len(Z)))
          finally:
             print("Something went really wrong with parallel - trying 1 core ...")
             o = Parallel(n_jobs = 1, verbose=1, pre_dispatch='2 * n_jobs', max_nbytes=None)\
                          (delayed(getCRF)(Z[k], L[k], num_classes) for k in range(len(Z)))

          ## process each small image chunk in parallel

          ## get grids to deal with overlapping values
          gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))
          ## get sliding windows
          Zx,_ = sliding_window(gridx, (sx, sy), (ssx, ssy))
          Zy,_ = sliding_window(gridy, (sx, sy), (ssx, ssy))

          ## get two grids, one for division and one for accumulation
          av = np.zeros((nx, ny))
          out = np.zeros((nx, ny))
          for k in range(len(o)):
             av[Zx[k], Zy[k]] += 1
             out[Zx[k], Zy[k]] += o[k]

          ## delete what we no longer need
          del Zx, Zy, o
          ## make the grids by taking an average, plus one, and floor
          res = np.floor(1+(out/av))
          del out, av

       ## median filter to remove remaining high-freq spatial noise (radius of N pixels)
       N = np.round(11*(ny/7362)).astype('int') #11 when ny=7362

       res = median(res.astype(np.uint8), disk(N))

       outfile2 = outfile.replace('_label.png', '_label_optim.png')
       ## write out the refined flattened label image
       print("Writing our 2D label image to %s" % (outfile2))
       if num_classes==2:
          res[res==0] = 2 #2 is null class
          res = res-1
          cv2.imwrite(outfile2,
                np.round(255*((res)/np.max(res))).astype('uint8'))
       else:
          lab = np.round(255*(res/num_classes )).astype('uint8')
          cv2.imwrite(outfile2,lab)

          if config['create_gtiff']=='true':
             image_path = outfile.replace('.png','.tif')

             WriteGeotiff(image_path, lab, profile)
















       # b
