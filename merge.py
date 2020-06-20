#  _  _  _    _   ,_    __,  _
# / |/ |/ |  |/  /  |  /  | |/
#   |  |  |_/|__/   |_/\_/|/|__/
#                        /|
#                        \|

# merge label rasters created using doodler.py
#
# > Daniel Buscombe, Marda Science daniel@mardascience.com
# > USGS Pacific Marine Science Center
#
import os, json, csv
import sys, getopt
import cv2
import numpy as np
from glob import glob
import rasterio
from PIL import Image

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk, erosion
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import as_strided as ast
from math import gcd
from skimage.filters.rank import median
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

       R = []

       ## loop through the 'theta' values (half, given, and double)
       for mult in search:
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
          R.append(1+np.argmax(Q, axis=0).reshape((H, W)))
          del Q

       res = np.round(np.median(R, axis=0))
       del R

       if fact>1:
          res = np.array(Image.fromarray(res.astype(np.uint8)).resize((Worig, Horig), resample=1))

       ## median filter to remove remaining high-freq spatial noise (radius of N pixels)
       N = np.round(11*(Worig/(3681))).astype('int') #11 when ny=3681
       #print("median filter size: %i" % (N))

       res = median(res.astype(np.uint8), disk(N))

       if len(label_lines)==2:
           res = erosion(res, disk(N))

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
        msk : 2D label image, coded according to RGB color codes
		classes_names: list of class strings
		rgb: list of RGB tuples
    """
    classes_colors = [classes[k] for k in classes.keys() if 'no_' not in k]
    classes_codes = [i for i,k in enumerate(classes) if 'no_' not in k]
    classes_names = [k for i,k in enumerate(classes) if 'no_' not in k]

    rgb = []
    for c in classes_colors:
       rgb.append(tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))

    hist_class = []

    if len(classes_codes)==1: #binary
       counter = 0
       for k in rgb:
          ind = (img[:,:,0]==classes_codes[counter])
          msk[ind] = k
          hist_class.append(len(np.where(ind==1)[0]))
          counter += 1
    else:
       msk2 = msk.copy()
       tmp = np.round(len(classes)*(img[:,:,0]/255.)).astype('int')-1

       counter = 0
       for k in rgb:
          ind = (tmp==classes_codes[counter])
          msk2[ind] = k
          hist_class.append(len(np.where(ind==1)[0]))
          counter += 1
       msk2[msk>0] = msk[msk>0]
       msk=msk2.copy()
       del msk2, tmp

    return msk, classes_names, rgb, classes_colors, hist_class


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
       msk_flat = (msk_flat).astype(np.uint8)
       M.append(msk_flat)
       del msk_flat

    del msk
    M2 = [(M[counter]==1)*(1+counter) for counter in range(len(M))]
    del M
    msk_flat = np.sum(M2, axis=0)
    del M2
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
    else:
        img = layer

    if np.max(img) > 255:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')

    img[img[:,:,2] == 255] = 254

    return img

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


    ##===================================================

    ## list of label images to combine
    #to_merge = [config['to_merge'][k] for k in config['to_merge'].keys()]

    to_merge = []
    if type(config["to_merge"]) is str:
       to_search = glob(config['label_folder']+os.sep+'*'+config["apply_mask"]+'*label.png')
       to_merge.append(to_search)
    elif type(config["to_merge"]) is list:
       for k in config["to_merge"]:
          to_search = glob(config['label_folder']+os.sep+'*'+k+'*label.png')[0]
          to_merge.append(to_search)


    ##list of associated class sets
    class_sets = [c for c in config.keys() if c.startswith('class')]

    ## get first mask in list
    img = cv2.imread(to_merge[0])

    ## allocate empty array of same dimensions
    msk = np.zeros((img.shape), dtype=np.uint8)
    del img

    ##allocate empty dictionary for classes and rgb colors
    class_dict = {}

    ## get rgb label for first label image
    msk, classes_names, rgb, classes_colors, _ = merge_labels(msk, cv2.imread(to_merge[0]), config[class_sets[0]])

    H = []; #NH = []
    ##update dictionary
    for c,r,h in zip(classes_names, rgb, classes_colors):
       class_dict[c] = r
       H.append(h)

    ## do same for rest of class sets
    for ii,cc in zip(to_merge[1:], class_sets[1:]):
       msk, classes_names, rgb, classes_colors, _ = merge_labels(msk, cv2.imread(ii), config[cc])
       for c,r,h in zip(classes_names, rgb, classes_colors):
          class_dict[c] = r
          H.append(h)


    outfile = to_merge[0].split('label.png')[0].split(list(config['classes1'])[0])[0]+'rgb.csv'

    ##write class dict to csv file
    with open(outfile, 'w') as f:
       f.write("%s,%s,%s,%s\n" % ('class', 'r', 'g', 'b' ))
       for key in class_dict.keys():
          f.write("%s,%s\n" % (key, str(class_dict[key]).replace(')','').replace('(','')) )

    num_classes = len(class_dict)

    ##===================================================

    outfile = outfile.replace('rgb.csv','merged_rgb_label.png')

    ## write out initial rgb image (this will be revised by CRF later)
    print("Writing our RGB image to %s" % (outfile))
    cv2.imwrite(outfile, cv2.cvtColor(msk, cv2.COLOR_RGB2BGR) )

    ##get rgb colors
    cols = [class_dict[c] for c in class_dict.keys()]
    ## flatten 3d label image to 2d
    msk_flat = flatten_labels(msk, cols )
    msk_flat = msk_flat.astype('uint8')
    del msk

    ## get image files list
    files = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*.*')))

    ##get image names
    names = [os.path.splitext(f)[0].split(os.sep)[-1] for f in files]

    ## the file to use
    imfile = [n for n in names if outfile.split('/')[-1].startswith(n)][0]
    ##the file to use with full path
    to_use = [f for f in files if imfile in f][0]

    ##===================================================

    ##read image
    img, profile = OpenImage(to_use, config['im_order'], config['num_bands'])

    ## get CRF label refinement
    print("Dense labelling ... this may take a while")

    ##50% overlap between tiles = .5, no overlap = 1.
    overlap = 1.
    nx, ny, nz = np.shape(img)
    ##number of chunks in each dimension
    num_chunks = gcd(nx, ny)

    ##===================================================

    if (num_chunks==nx) | (num_chunks==ny):
       ## do this without windowing/overlap
       res = getCRF(img, msk_flat+1, class_dict, config['fact'])-1

    elif num_chunks>100:

       res = getCRF(img, msk_flat+1, class_dict, config['fact'])-1

    else:

       apply_fact = (nx > 8000) or (ny > 8000)
       
       if apply_fact:

          ##size of each chunk
          sx = int(nx/(num_chunks))
          sy = int(ny/(num_chunks))
          ssx = int(overlap*sx)
          ssy = int(overlap*sy)

          ##gets small overlapped image windows
          Z, indZ = sliding_window(img, (sx, sy, nz), (ssx, ssy, nz))
          ##gets small overlapped label windows
          L, indL = sliding_window(msk_flat, (sx, sy), (ssx, ssy))
          del msk_flat

          print("working on %i chunks, each %i x %i pixels" % (len(Z), sx, sy))

          ## process each small image chunk in parallel, with one core left over
          o = Parallel(n_jobs = -2, verbose=1, pre_dispatch='2 * n_jobs', max_nbytes=1e6)\
                   (delayed(getCRF_optim)(Z[k], L[k], num_classes, config['fact']) for k in range(len(Z)))

          ims, theta_cols, compat_cols  = zip(*o)  ##, compat_spats, theta_spats, probs
          del o, Z, L

          #get the median of each as the global best for the image
          theta_col = np.nanmedian(theta_cols)
          compat_col = np.nanmedian(compat_cols)

          print("======================================")
          print("Optimal color theta for this image: %f" % (theta_col))
          print("Optimal color compat for this image: %f" % (compat_col))
          print("======================================")

          ## get grids to deal with overlapping values
          gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))
          ## get sliding windows
          Zx,_ = sliding_window(gridx, (sx, sy), (ssx, ssy))
          Zy,_ = sliding_window(gridy, (sx, sy), (ssx, ssy))
          del gridx, gridy

          ## get two grids, one for division and one for accumulation
          ## this is to handle any amount of overlap between tiles
          av = np.zeros((nx, ny))
          out = np.zeros((nx, ny))
          for k in range(len(ims)):
             av[Zx[k], Zy[k]] += 1
             out[Zx[k], Zy[k]] += ims[k]

          ## delete what we no longer need
          del Zx, Zy, ims
          ## make the grids by taking an average, plus one, and floor
          res = np.floor(1+(out/av))-1
          del out, av
          

       else: #imagery < 8000 pixels
          res = getCRF(img, msk_flat+1, class_dict, config['fact'])-1
          del msk_flat
          

    ##===================================================
    ## replace large with most populous value
    res = res.astype(np.uint8)
    res[res>len(cols)] = np.argmax(np.bincount(res.flatten()))

    ##===================================================
    ## write out the refined flattened label image
    
    outfile = outfile.replace('rgb_label.png', 'label_crf.png')
    
    print("Writing our 2D label image to %s" % (outfile))
    cv2.imwrite(outfile,
	            np.round(255*(res/len(class_dict) )).astype('uint8')) 

    #write out geotiff label if requested
    if config['create_gtiff']=='true':
       image_path = outfile.replace('.png','.tif')

       WriteGeotiff(image_path, np.round(255*(res/len(class_dict) )).astype('uint8'), profile)

    ## allocate empty 3D array of same x-y dimensions
    msk = np.zeros((res.shape)+(3,), dtype=np.uint8)

    ##do the rgb allocation
    for k in np.unique(res):
       ind = (res==k)
       msk[ind] = cols[int(k)-1]

    ##mask out null portions of image
    msk[np.sum(img,axis=2)==(254*3)] = 0

    outfile = outfile.replace('label_crf.png', 'rgb_label_crf.png')

    ## write out smoothed rgb image
    print("Writing our RGB image to %s" % (outfile))
    cv2.imwrite(outfile, cv2.cvtColor(msk, cv2.COLOR_RGB2BGR) )

    ##===================================================
    ## make a matplotlib overlay plot
    resr = res.astype('float')
    del res
    resr[resr<1] = np.nan
    resr = resr-1

    try:
       alpha_percent = config['alpha'] #0.75
    except:
       alpha_percent = 0.5

    new_cols = []
    for col in H:
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
       # img = img.astype('float')
       # img = np.round(255*(img/img.max()))
       # img[img==0] = np.nan
       _ = ax1.imshow(img)

    im2 = ax1.imshow(resr,
                     cmap=cmap,
                     alpha=alpha_percent, interpolation='nearest',
                     vmin=0, vmax=len(new_cols))
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%")
    cb=plt.colorbar(im2, cax=cax)
    cb.set_ticks(0.5 + np.arange(len(new_cols) + 1))
    cb.ax.set_yticklabels(list(class_dict.keys()) , fontsize=6)

    #name = os.path.splitext(config['outfile'])[0].split(os.sep)[-1]
    class_str = '_'.join(list(class_dict.keys()))

    #outfile = config['image_folder'].replace('images','label_images')+os.sep+name+"_"+class_str+'_mres_merged.png'
    
    outfile = outfile.replace('merged_rgb_label_crf.png', class_str+'.png')

    plt.savefig(outfile,
                dpi=300, bbox_inches = 'tight')
    del fig; plt.close()
    
    
    
    
