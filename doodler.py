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
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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

    if 'configfile' not in locals():
        root = Tk()
        configfile =  askopenfilename(initialdir = "/config",
                                                    title = "choose your config file",
                                                    filetypes = (("json files","*.json"),("all files","*.*")))
        print("[INFO] Configuration file selected : %s" % (configfile))
        root.withdraw()
        del root

    #configfile = 'config.json'
    # load the user configs
    try:
        with open(os.getcwd()+os.sep+configfile) as f:
            config = json.load(f)
    except:
        with open(configfile) as f:
            config = json.load(f)

    # for k in config.keys():
    #     exec(k+'=config["'+k+'"]')

    ## TODO: add error checking on config item dtypes

    ## add defaults for missing items
    if "create_gtiff" not in config:
       config['create_gtiff'] = False
    if "alpha" not in config:
       config['alpha'] = 0.5
    if "apply_mask" not in config:
       config['apply_mask'] = None
    if "fact" not in config:
       config['fact'] = 3
    if "compat_col" not in config:
       config['compat_col'] = 120
    if "theta_col" not in config:
       config['theta_col'] = 100
    if "medfilt" not in config:
       config['medfilt'] = "true"
    if "thres_size_1chunk" not in config:
       config['thres_size_1chunk'] = 10000
    if "optim" not in config:
       config['optim'] = False
    if "do_stack" not in config:
       config['do_stack'] = "true"

    #the program needs two classes
    if len(config['classes'])==1:
       print(
       "[ERROR] You must have a minimum of 2 classes, i.e. 1) object of interest \
       and 2) background ... program exiting"
       )
       sys.exit(2)

    class_str = '_'.join(config['classes'].keys())

    ## TODO: add error checking on files items to check if files exist

    files = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*.*')))

    files = [f for f in files if '.txt' not in f]

    N = []
    if 'doodles' in locals():
       match = os.path.splitext(npy_file)[0].split(os.sep)[-1]
       for f in files:
          tmp = os.path.splitext(f)[0].split(os.sep)[-1]
          if match.startswith(tmp):
             N.append(tmp)

    # for f in files:
    #   N.append(os.path.splitext(f)[0].split(os.sep)[-1])

    ## TODO: add error checking on apply-mask items to check if files exist

    # masks are binary labels where the null class is zero
    masks = []
    mask_names = []
    try:
        if config["apply_mask"]!='None':
           if type(config["apply_mask"]) is str:
              to_search = glob(config['label_folder']+os.sep+'*'+\
                          config["apply_mask"]+'*label.png')
              for f in to_search:
                 tmp, profile = OpenImage(f, None)
                 tmp = (tmp[:,:,0]==0).astype('uint8')
                 masks.append(tmp)
                 mask_names.append(f)
                 del tmp
           elif type(config["apply_mask"]) is list:
              for k in config["apply_mask"]:
                 to_search = glob(config['label_folder']+os.sep+'*'+k+'*label.png')

                 for f in to_search:
                    tmp, profile = OpenImage(f, None)
                    if len(np.unique(tmp))==2:
                       tmp = (tmp[:,:,0]==0).astype('uint8')
                    else:
                       #assumes the 'null' or masking class is all but the last
                       tmp = (tmp[:,:,0]!=np.max(tmp[:,:,0])).astype('uint8')
                    masks.append(tmp)
                    mask_names.append(f)
                    del tmp
    except:
        print("Something went wrong with the image masking - proceeding without")


    ## TODO: add error checking on apply-mask items to check if label folder is valid

    if 'doodles' not in locals(): #make annotations

       ##cycle through each file in turn
       for f in tqdm(files):

          screen_size = Screen()
          o_img, profile = OpenImage(f, config['im_order'])

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

          del Z, gridx, gridy, o_img, mp
          outfile = config['label_folder']+os.sep+name+"_"+class_str+".npy"
          np.save(outfile, out)
          print("[INFO] annotations saved to %s" % (outfile))
          del out

       print("[INFO] Sparse labelling complete ...")
    else:
       print("[INFO] Using provided labels ...")

    #gc.collect()
    print("[INFO] Dense labelling ... this may take a while")

    # cycle through each image root name, stored in N
    for name in N:
        print("[INFO] Working on %s" % (name))

        if 'doodles' not in locals(): #make annotations

           # get a list of the temporary npz files
           label_files = sorted(glob(config['label_folder']+os.sep+name +'*tmp*'+\
                         class_str+'.npz'))
           print("[INFO] Found %i image chunks" % (len(label_files)))

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
              img, profile = OpenImage(imfile, config['im_order'])

              # ==========================================
			  # allocate an empty label image
              resr = np.zeros((nx, ny))

              # assign the CRF tiles to the label image (resr) using the grid positions
              for k in range(len(label_files)):
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
              del probs#, l


           else: #image is small enough to fit on most memory at once
              print("[INFO] Imagery is small (<%i px in all dimensions), so not using chunks", (config['thres_size_1chunk']))
              # get image file and read it in with the profile of the geotiff, if appropriate
              imfile = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*'+name+'*.*')))[0]
              img, profile = OpenImage(imfile, config['im_order'])
              l = sorted(glob(config['label_folder']+os.sep+name +'*'+class_str+'.npy'))[0]
              l = np.load(l).astype(np.uint8)
              resr, prob, preds = getCRF(img, l, config)

        else: #labels provided
           print("[INFO] computing CRF using provided annotations")

           # get image file and read it in with the profile of the geotiff, if appropriate
           imfile = sorted(glob(os.path.normpath(config['image_folder']+\
                       os.sep+'*'+name+'*.*')))[0]
           img, profile = OpenImage(imfile, config['im_order'])

           resr, prob, preds = getCRF(img,doodles.astype(np.uint8), config) ##, False)

        #outfile = config['label_folder']+os.sep+name+"_probs_per_class.npy"
        #np.save(outfile, preds)
        #del preds

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
                   os.path.normpath(m).startswith(os.path.splitext(imfile)[0].replace('images', 'label_images'))]
            for u in use:
               ind = [i for i in range(len(mask_names)) if mask_names[i]==u][0]
               resr[masks[ind]==1] = 0
               prob[masks[ind]==1] = 0

        # plot the result and make label image files
        if 'doodles' not in locals(): #make annotations
            PlotAndSave(img.copy(), resr, l.astype(np.uint8), prob, name, config, class_str, profile)
        else:
            PlotAndSave(img.copy(), resr, doodles.astype(np.uint8), prob, name, config, class_str, profile)

        del resr, img, prob

        #remove the temporary npz files
        try:
            for f in label_files:
               os.remove(f)
        except:
            pass
