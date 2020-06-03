# merge label rasters created using doodler.py
#
# > Daniel Buscombe, Marda Science daniel@mardascience.com
# > USGS Pacific Marine Science Center
#
import os, json
import sys, getopt
import cv2
import numpy as np
from glob import glob

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import as_strided as ast
from math import gcd

# =========================================================
def getCRF(img, Lc, label_lines):
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

       R = []
    
       ## loop through the 'theta' values (half, given, and double)	
       for mult in [.5,1,2]: 
          d = dcrf.DenseCRF2D(H, W, len(label_lines) + 1)
          U = unary_from_labels(Lc.astype('int'),
                          len(label_lines) + 1,
                          gt_prob=config['prob'])
          d.setUnaryEnergy(U)

          # to add the color-independent term, where features are the locations only:
          d.addPairwiseGaussian(sxy=(int(mult*config['theta_spat']), int(mult*config['theta_spat'])), compat=config['compat_spat'], kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

          feats = create_pairwise_bilateral(sdims=(int(mult*config['theta_col']), int(mult*config['theta_col'])),
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

       res = np.round(np.median(R, axis=0))
       del R

    return res 

#===============================================================
def merge_labels(msk, img, classes):
    """
    Flattens a 3D (RGB) label image into 2D (integer) according to 
	hex color codes
    Input:
        msk: 3D ndarray label image
		img: 2D ndarray label image
		classes: dictionary of class names and hex color codes
    Output:
        msk : 3D label image, coded according to RGB color codes
		classes_names: list of class strings
		rgb: list of RGB tuples
    """
    classes_colors = [classes[k] for k in classes.keys() if 'no_' not in k]
    classes_codes = [i for i,k in enumerate(classes) if 'no_' not in k]
    classes_names = [k for i,k in enumerate(classes) if 'no_' not in k]	
	
    rgb = []
    for c in classes_colors:
       rgb.append(tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
	
    counter = 0
    for k in rgb:
       ind = (img[:,:,0]==classes_codes[counter])
       msk[ind] = k 
       counter += 1	   
    return msk, classes_names, rgb

# =========================================================
def OpenImage(image_path, im_order):
    """
    Returns the image in numpy array format
    Input:
        image_path : string
            Full or relative path to image
        config : dict
            Dictionary of parameters set in the parameters file
    Output:
        numpy array of image 2D or 3D #NOTE: want this to do multispectral
    """
    if image_path.lower()[-3:] == 'tif':
        img = WF.ReadGeotiff(image_path, im_order) #need to implemten WF
    else:
        img = cv2.imread(image_path)
        if im_order=='RGB':
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif im_order=='BGR':
           img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

#===============================================================
def flatten_labels(msk, cols):
    """
    Flattens a 3D (RGB) label image into 2D (integer)
    Input:
        msk: 3D ndarray label image
		cols: list of rgb color tuples
    Output:
        msk_flat : 2D label image, coded according to integers in sequence of cols
    """
    M = []
    for k in range(len(cols)):
       col = list(cols[k])
       msk_flat = ((msk[:,:,0]==col[0])==1) & ((msk[:,:,1]==col[1])==1) & ((msk[:,:,2]==col[2])==1)
       msk_flat = (msk_flat).astype('int')
       M.append(msk_flat)
       del msk_flat

    M2 = [(M[counter]==1)*(1+counter) for counter in range(len(M))]
    msk_flat = np.sum(M2, axis=0)
    return msk_flat


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
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:c:")
    except getopt.GetoptError:
        print('python merge.py -c configfile.json')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python merge.py -c config_merge.json')
            sys.exit()
        elif opt in ("-c"):
            configfile = arg

    #configfile = 'config_merge.json'
	
    # load the user configs
    with open(os.getcwd()+os.sep+configfile) as f:
        config = json.load(f)
		
    ## list of label images to combine
    to_merge = [config['to_merge'][k] for k in config['to_merge'].keys()]	

    ##list of associated class sets 
    class_sets = [c for c in config.keys() if c.startswith('class')]	

    ## get first mask in list
    img = cv2.imread(to_merge[0])
	
    ## allocate empty array of same dimensions
    msk = np.zeros((img.shape), dtype=np.uint8)
	
    ##allocate empty dictionary for classes and rgb colors
    class_dict = {}
		
    ## get rgb label for first label image
    msk, classes_names, rgb = merge_labels(msk, img, config[class_sets[0]])
	
    ##update dictionary	
    for c,r in zip(classes_names, rgb):
       class_dict[c] = r
    
    ## do same for rest of class sets
    for ii,cc in zip(to_merge[1:], class_sets[1:]):
       msk, classes_names, rgb = merge_labels(msk, cv2.imread(ii), config[cc])
       for c,r in zip(classes_names, rgb):
          class_dict[c] = r

    ## write out rgb image
    print("Writing our RGB image to %s" % (config['outfile']))
    cv2.imwrite(config['outfile'], cv2.cvtColor(msk, cv2.COLOR_RGB2BGR) )

    ##get rgb colors
    cols = [class_dict[c] for c in class_dict.keys()]
    ## flatten 3d label image to 2d	
    msk_flat = flatten_labels(msk, cols )

    ## get image files list
    files = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*.*')))
    ##get image names	
    names = [f.split(os.sep)[-1].split('.')[0] for f in files]
    ## the file to use 	
    imfile = [n for n in names if config['outfile'].split('/')[-1].startswith(n)][0]
    ##the file to use with full path
    to_use = [f for f in files if imfile in f][0]
	
    ##read image	
    img = OpenImage(to_use, config['im_order'])

    ## get CRF label refinement
    print("Dense labelling ... this may take a while")
	
    ##50% overlap between tiles
    overlap = .5
    nx, ny, nz = np.shape(img)
    ##number of chunks in each dimension	
    num_chunks = gcd(nx, ny)
    ##size of each chunk
    sx = int(nx/num_chunks)
    sy = int(ny/num_chunks)
    ssx = int(overlap*sx)
    ssy = int(overlap*sy)
	
    ##gets small overlapped image windows	
    Z, indZ = sliding_window(img, (sx, sy, nz), (ssx, ssy, nz))
    del img
    ##gets small overlapped label windows	
    L, indL = sliding_window(msk_flat, (sx, sy), (ssx, ssy))
    del msk_flat	

    ## process each small image chunk in parallel 
    o = Parallel(n_jobs = -1, verbose=1, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(getCRF)(Z[k], L[k], class_dict) for k in range(len(Z)))

    ## get grids to deal with overlapping values
    gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))
    ## get sliding windows
    Zx,_ = sliding_window(gridx, (sx, sy), (ssx, ssy))
    Zy,_ = sliding_window(gridy, (sx, sy), (ssx, ssy))
	
    ## get two grids, one for averaging and one for accumulation
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

    ## to do this without windowing/overlap, we would do	
    #res = getCRF(img, msk_flat, class_dict)+1

    ## median filter to remove remaining high-freq spatial noise
    res = median(res.astype(np.uint8), disk(11))
    res[res==0] = np.argmax(np.bincount(res.flatten()))	
	
    ## write out the refined flattened label image
    print("Writing our 2D label image to %s" % (config['outfile'].replace('.png', '_flat.png')))
    cv2.imwrite(config['outfile'].replace('.png', '_flat.png'), 
	            np.round(255*(res/np.max(res))).astype('uint8'))    
		
    del res
    ## allocate empty 3D array of same x-y dimensions
    msk = np.zeros((res.shape)+(3,), dtype=np.uint8)
    	
    for k in np.unique(res):
       ind = (res==k)
       msk[ind] = cols[k-1]
       
    ## write out smoothed rgb image
    print("Writing our RGB image to %s" % (config['outfile'].replace('.png', '_crf.png')))
    cv2.imwrite(config['outfile'].replace('.png', '_crf.png'), cv2.cvtColor(msk, cv2.COLOR_RGB2BGR) )














