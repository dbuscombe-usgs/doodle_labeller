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

from funcs import *

#===============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:c:f:")
    except getopt.GetoptError:
        print('python doodler_optim.py -c configfile.json [-f npy_file.npy]')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python doodler_optim.py -c \
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
       config['fact'] = 3
    if "medfilt" not in config:
       config['medfilt'] = "true"
    if "compat_col" not in config:
       config['compat_col'] = 100
    if "theta_col" not in config:
       config['theta_col'] = 60
    if "thres_size_1chunk" not in config:
       config['thres_size_1chunk'] = 10000

    # for k in config.keys():
    #     exec(k+'=config["'+k+'"]')

    if len(config['classes'])==1:
       print("You must have a minimum of 2 classes, i.e. \
             1) object of interest and 2) background")
       sys.exist(2)

    class_str = '_'.join(config['classes'].keys())

    files = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*.*')))

    files = [f for f in files if '.txt' not in f]

    N = []
    if 'doodles' in locals():
       for f in files:
          N.append(os.path.splitext(f)[0].split(os.sep)[-1])

    # for f in files:
    #   N.append(os.path.splitext(f)[0].split(os.sep)[-1])
    #

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

    if 'doodles' not in locals(): #make annotations

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


          name = os.path.splitext(f)[0].split(os.sep)[-1]
          N.append(name)

          mp = MaskPainter(o_img.copy(), config, screen_size)
          out, Z = mp.LabelWindow()

          out = out.astype(np.uint8)
          nx, ny = np.shape(out)
          gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))

          counter = 0
          for ind in Z:
             np.savez(config['label_folder']+os.sep+name+"_tmp"+str(counter)+\
                    "_"+class_str+".npz",
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

    print("Dense labelling ... this may take a while")

    #gc.collect()

    # cycle through each image root name, stored in N
    for name in N:
        print("Working on %s" % (name))

        if 'doodles' not in locals(): #make annotations

           # get a list of the temporary npz files
           label_files = sorted(glob(config['label_folder']+os.sep+name \
                         +'*tmp*'+class_str+'.npz'))
           print("Found %i image chunks" % (len(label_files)))

           #load data to get the size
           l = sorted(glob(config['label_folder']+os.sep+name +'*'+ \
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
                       (delayed(DoCrf)(label_files[k], config, name)\
                        for k in range(len(label_files)))

              #parse the object 'o' into images and a list of parameters
              ims, probs, preds  = zip(*o)  ##, compat_spats, theta_spats, probs # theta_cols, compat_cols,
              del o

              #get the median of each as the global best for the image
              #theta_col = np.nanmedian(theta_cols)
              #theta_spat = np.nanmedian(theta_spats)
              #prob = np.nanmedian(probs)
              #compat_col = np.nanmedian(compat_cols)
              #compat_spat = np.nanmedian(compat_spats)

              # load the npy file and get the grid coordinates to assign elements of 'ims' below
              l = sorted(glob(config['label_folder']+os.sep+name +'*'+\
                         class_str+'.npy'))[0]
              l = np.load(l).astype(np.uint8)
              nx, ny = np.shape(l)
              del l

              # get image file and read it in with the profile of the geotiff, if appropriate
              imfile = sorted(glob(os.path.normpath(config['image_folder']+\
                       os.sep+'*'+name+'*.*')))[0]
              img, profile = OpenImage(imfile, config['im_order'], config['num_bands'])

              # allocate an empty label image
              resr = np.zeros((nx, ny))

              # assign the CRF tiles to the label image (resr) using the grid positions
              for k in range(len(ims)):
                 l = np.load(label_files[k])#.astype(np.uint8)
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
              print("Imagery is small (<%i px), so not using chunks - they will be deleted" % \
                   (config['thres_size_1chunk']))
              # get image file and read it in with the profile of the geotiff, if appropriate
              imfile = sorted(glob(os.path.normpath(config['image_folder']+\
                       os.sep+'*'+name+'*.*')))[0]
              img, profile = OpenImage(imfile, config['im_order'], config['num_bands'])
              l = sorted(glob(config['label_folder']+os.sep+name +'*'+\
                         class_str+'.npy'))[0]
              l = np.load(l).astype(np.uint8)

              resr, prob, preds = getCRF(img,l,config, True)
                            #len(config['classes']), config['fact']) # mu_col, mu_spat, prob
              #resr += 1

        # print("======================================")
        # print("Optimal color theta for this image: %f" % (theta_col))
        # #print("Optimal space theta for this image: %f" % (theta_spat))
        # print("Optimal color compat for this image: %f" % (compat_col))
        # #print("Optimal space compat for this image: %f" % (compat_spat))
        # #print("Optimal prior for label prob for this image: %f" % (prob))
        # print("======================================")

        else: #labels provided
           print("computing CRF using provided annotations")

           nx, ny = np.shape(doodles)
           ## memory management
           if np.max((nx, ny))>2000:
               config['fact'] = np.max((config['fact'], 3))

           # get image file and read it in with the profile of the geotiff, if appropriate
           imfile = sorted(glob(os.path.normpath(config['image_folder']+\
                       os.sep+'*'+name+'*.*')))[0]
           img, profile = OpenImage(imfile, config['im_order'], config['num_bands'])

           resr, prob, preds = getCRF(img,doodles.astype(np.uint8),config, True)
                            #len(config['classes']), config['fact']) # mu_col, mu_spat, prob


        outfile = config['label_folder']+os.sep+name+"_probs_per_class.npy"
        np.save(outfile, preds)
        del preds

        #gc.collect()

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

        #remove the temporary npz files
        try:
            for f in label_files:
               os.remove(f)
        except:
            pass

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
#
#     data = np.load(file)
#
#     #img = data['image']
#     #Lc = data['label']
#     #num_classes = len(config['classes'])
#
#     res, theta_col, compat_col, p, preds = getCRF_optim(data['image'],
#                             data['label'],
#                             len(config['classes']), config['fact']) # mu_col, mu_spat, prob
#
#     if np.all(res)==254:
#        res *= 0
#
#     return res, theta_col, compat_col, p, preds #, prob #mu_col, mu_spat, theta_spat



       #get the entropy of the first channel with a 10-px disk structural element
       #entr_img = entropy(img[:,:,0], disk(10))

       # ## binary case
       # if num_classes==2:
       #    #for each label image in R, sum the entropy in the R^1 (no object) class
       #    scores = [np.sum(entr_img*z) for z in R]
       #    del R
       #    #optimal parameters based on minimum summed entropy
       #    mult_col, mult_spat, mult_col_compat, mult_spat_compat = P[np.argmin(scores)]
       # else:
       #    #for each label image in R, sum the entropy in the R^1 (no object) class
       #    S = []
       #    for z in R:
       #        #score = sum of entropy in each class
       #        scores = [np.sum(entr_img*(z==i).astype('int')) for i in np.unique(z)]
       #        S.append(scores)
       #    # if the length of R is the same as length of S
       #    if len(R)==len(np.sum(np.array(S), axis=1)): #axis=0
       #       S = np.sum(np.array(S), axis=1)
       #    else: #if the lengths are unequal, it is because the number of classes change
       #       #from iteration to iteration
       #       #get the number of predicted classes in each realization of S
       #       ind = [len(s) for s in S]
       #       #find those that correspond to the maximum
       #       ind = np.where(ind==np.argmax(np.bincount(ind)))[0]
       #       #count only those
       #       P = [P[i] for i in ind]
       #       S = [S[i] for i in ind]
       #       K2 = [K[i] for i in ind]
       #       #now sum
       #       S = np.sum(np.array(S), axis=0)
       #
       #    del R
       #    #optimal parameters based on minimum summed entropy
       #    mult_col, mult_spat, mult_col_compat, mult_spat_compat = P[np.argmin(S)]

       #mult_col, mult_col_compat = P[np.argmin(K)] #mult_col_compat, mult_spat_compat, mult_prob

       # print("================================")
       # print("Optimal color theta hyperparameter: %f" % (mult_col*theta_col))
       # #print("Optimal spatial theta hyperparameter: %f" % (mult_spat*theta_spat))
       # print("Optimal color compat hyperparameter: %f" % (mult_col_compat*compat_col))
       # #print("Optimal spatial compat hyperparameter: %f" % (mult_spat_compat*compat_spat))
       # #print("Optimal prior for label probability: %f" % (mult_prob*prob))
       # print("================================")

       # make new images scaled back up to the correct size
       #img = np.array(Image.fromarray(img).resize((Worig, Horig), resample=1))
       #Lc = np.array(Image.fromarray(Lc).resize((Worig, Horig), resample=1))

       # #common unary potentials for each class
       # U = unary_from_labels(Lc.astype('int'),
       #                    num_classes + 1,
       #                    gt_prob=np.min((mult_prob*prob,.999)))
       #
       # # do the CRF again with optimal parameters
       # d = dcrf.DenseCRF2D(H, W, num_classes + 1)
       #
       # d.setUnaryEnergy(U)
       #
       # # to add the color-independent term, where features are the locations only:
       # d.addPairwiseGaussian(sxy=(int(theta_spat), int(theta_spat)), compat=compat_spat, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
       #
       # # color-dependent features
       # feats = create_pairwise_bilateral(
       #                    sdims=(int(mult_col*theta_col), int(mult_col*theta_col)),
       #                    schan=(scale,
       #                           scale,
       #                           scale),
       #                    img=img,
       #                    chdim=2)
       #
       # d.addPairwiseEnergy(feats, compat=mult_col_compat*compat_col,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
       # Q = d.inference(n_iter*2)

       #kl = d.klDivergence(Q)

       # res = np.argmax(Q, axis=0).reshape((H, W))
       #del Q, d, U


#
# #===============================================================
# def WriteGeotiff(image_path, lab, profile):
#     """
#     This function writes a 1-band label image in GeoTIFF format.
#     TODO: Fill in the doc string better
#     Parameters
#     ----------
#     image_path : string
#         full or relative path to the tiff image you'd like to create
#     lab : 2D label raster
#     profile: rasterio profile of the original geotiff image
#     """
#
#     with rasterio.Env():
#        # change the band count to 1, set the
#        # dtype to uint8, and specify LZW compression.
#        profile.update(
#           dtype=rasterio.uint8,
#           count=1,
#           compress='lzw')
#
#        with rasterio.open(image_path, 'w', **profile) as dst:
#           dst.write(lab.astype(rasterio.uint8), 1) #1=1 band
#
#
# #===============================================================
# def ReadGeotiff(image_path, rgb):
#     """
#     This function reads image in GeoTIFF format.
#     TODO: Fill in the doc string better
#     Parameters
#     ----------
#     image_path : string
#         full or relative path to the tiff image
#     rgb : TYPE
#         is it RGB or BGR
#     Returns
#     -------
#     img : array
#         2D or 3D numpy array of the image
#     """
#     with rasterio.open(image_path) as src:
#         layer = src.read()
#         profile = src.profile
#
#     if layer.shape[0] == 3:
#         r, g, b = layer
#         if rgb == 'RGB':
#             img = np.dstack([r, g, b])
#         else:
#             img = np.dstack([b, g, r])
#     elif layer.shape[0] == 4:
#         r, g, b, gd = layer
#         if rgb == 'RGB':
#             img = np.dstack([r, g, b])
#         else:
#             img = np.dstack([b, g, r])
#     else:
#         img = layer
#
#     if np.max(img) > 255:
#         img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#         img = img.astype('uint8')
#
#     img[img[:,:,2] == 255] = 254
#
#     return img, profile


# # =========================================================
# def PlotAndSave(img, resr, lab, name, config, class_str, profile):
#     """
#     Makes plots and save label images
#     Input:
#         img:
# 		resr:
# 		name:
# 		config:
# 		class_str:
#         profile: rasterio file profile (CRS, etc)
#     Output:
#         None
#     """
#     outfile = config['label_folder']+os.sep+name+"_"+class_str+'_label.png'
#
#     if len(config['classes'])==2:
#        resr[resr==0] = 2 #2 is null class
#        resr = resr-1
#        cv2.imwrite(outfile,
#                 np.round(255*((resr)/np.max(resr))).astype('uint8'))
#
#     else:
#        lab = np.round(255*(resr/len(config['classes'])))
#        cv2.imwrite(outfile, lab.astype('uint8')) ##np.max(resr)
#
#        if config['create_gtiff']=='true':
#           image_path = outfile.replace('.png','.tif')
#
#           WriteGeotiff(image_path, lab, profile)
#
#        resr = resr.astype('float')
#        resr[resr<1] = np.nan
#        resr = resr-1
#
#
#     outfile = config['label_folder']+os.sep+name+"_"+class_str+'_prob.png'
#
#     cv2.imwrite(outfile,
#                 np.round(255*prob).astype('uint8'))
#
#     if config['create_gtiff']=='true':
#        image_path = outfile.replace('.png','.tif')
#        WriteGeotiff(image_path, prob, profile)
#
#
#     try:
#        alpha_percent = config['alpha'] #0.75
#     except:
#        alpha_percent = 0.5
#
#     cols = list(config['classes'].values())
#     new_cols = []
#     for col in cols:
#         if not col.startswith('#'):
#             col = '#'+col
#         new_cols.append(col)
#     cmap = colors.ListedColormap(new_cols)
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111) #sp + 1)
#     ax1.get_xaxis().set_visible(False)
#     ax1.get_yaxis().set_visible(False)
#
#     if np.ndim(img)==3:
#        _ = ax1.imshow(img)
#     else:
#        img = img.astype('float')
#        img = np.round(255*(img/img.max()))
#        img[img==0] = np.nan
#        _ = ax1.imshow(img)
#
#     im2 = ax1.imshow(resr,
#                      cmap=cmap,
#                      alpha=alpha_percent, interpolation='nearest',
#                      vmin=0, vmax=len(config['classes']))
#     divider = make_axes_locatable(ax1)
#     cax = divider.append_axes("right", size="5%")
#     cb=plt.colorbar(im2, cax=cax)
#     cb.set_ticks(0.5 + np.arange(len(config['classes']) + 1))
#     cb.ax.set_yticklabels(config['classes'], fontsize=6)
#
#     outfile = config['label_folder']+os.sep+name+"_"+class_str+'_mres.png'
#
#     plt.savefig(outfile,
#                 dpi=300, bbox_inches = 'tight')
#     del fig; plt.close()
#
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111) #sp + 1)
#     ax1.get_xaxis().set_visible(False)
#     ax1.get_yaxis().set_visible(False)
#
#     if np.ndim(img)==3:
#        _ = ax1.imshow(img)
#     else:
#        img = img.astype('float')
#        img = np.round(255*(img/img.max()))
#        img[img==0] = np.nan
#        _ = ax1.imshow(img)
#
#     im2 = ax1.imshow(prob,
#                      cmap=plt.cm.Dark2,
#                      alpha=alpha_percent, interpolation='nearest',
#                      vmin=0, vmax=1)
#     divider = make_axes_locatable(ax1)
#     cax = divider.append_axes("right", size="5%")
#     cb=plt.colorbar(im2, cax=cax)
#     #cb.set_ticks(0.5 + np.arange(len(config['classes']) + 1))
#     #cb.ax.set_yticklabels(config['classes'], fontsize=6)
#
#     outfile = config['label_folder']+os.sep+name+"_"+class_str+'_prob_res.png'
#
#     plt.savefig(outfile,
#                 dpi=300, bbox_inches = 'tight')
#     del fig; plt.close()

#
# import subprocess, ctypes
# import os, json, gc
# import sys, getopt
# import cv2
# import numpy as np
# from glob import glob
# import rasterio, tifffile
#
# ##  progress bar (bcause te quiero demasiado ...)
# from tqdm import tqdm
# from scipy.stats import mode as md
#
#
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from joblib import Parallel, delayed
# from PIL import Image
#
# #from skimage.filters.rank import entropy
# from skimage.filters.rank import median
# from skimage.morphology import disk, erosion
