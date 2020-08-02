#  _  _  _    _   ,_    __,  _
# / |/ |/ |  |/  /  |  /  | |/
#   |  |  |_/|__/   |_/\_/|/|__/
#                        /|
#                        \|

# merge label rasters created using doodler.py
#
# > Daniel Buscombe, Marda Science daniel@mardascience.com
# > USGS Pacific Marine Science Center

from funcs import *
import itertools

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
       msk_flat = ((msk[:,:,0]==col[0])==1) & ((msk[:,:,1]==col[1])==1) & \
                  ((msk[:,:,2]==col[2])==1)
       msk_flat = (msk_flat).astype(np.uint8)
       M.append(msk_flat)
       del msk_flat

    del msk
    M2 = [(M[counter]==1)*(1+counter) for counter in range(len(M))]
    del M
    msk_flat = np.sum(M2, axis=0)
    del M2
    return msk_flat

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

    if 'configfile' not in locals():
        root = Tk()
        configfile =  askopenfilename(initialdir = "/config",
                                                    title = "choose your config file",
                                                    filetypes = (("json files","*.json"),("all files","*.*")))
        print("[INFO] Configuration file selected : %s" % (configfile))
        root.withdraw()

    #configfile = 'config.json'
    # load the user configs
    try:
        with open(os.getcwd()+os.sep+configfile) as f:
            config = json.load(f)
    except:
        with open(configfile) as f:
            config = json.load(f)

    if "name" not in config:
       print("[ERROR] Variable 'name' not in config file ... exiting")
       sys.exit(2)

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

    #=======================================================================
    #=========================SORT =========================
    #=======================================================================

    ## list of label images to combine
    to_merge = []
    if type(config["to_merge"]) is str:
       to_search = glob(config['label_folder']+os.sep+'*'+\
                   config["apply_mask"]+'*_label.png')
       to_merge.append(to_search)
    elif type(config["to_merge"]) is list:
       for k in config["to_merge"]:
          to_search = glob(config['label_folder']+os.sep+'*'+k+'*label.png')
          to_search = [s for s in to_search if s.startswith(config['label_folder']+\
                       os.sep+config['name'])]
          to_merge.append(to_search)

    ##to_merge is a list of list. nested lists are per class, not per site
    to_merge = [sorted(m) for m in to_merge]

    # sort out how many images and labels per image
    num_class_sets = len(to_merge)
    num_ims_per_set = len(to_merge[0])

    ##list of associated class sets
    class_sets = [c for c in config.keys() if c.startswith('class')]

    class_str = ['_'.join(config[cc].keys()) for cc in class_sets]

    all_names = [os.path.splitext(n)[0].split(os.sep)[-1] \
                 for n in list(itertools.chain(*to_merge)) ]

    all_stripped_names = []
    for name in all_names:
       for c in class_str:
          if c in name:
             all_stripped_names.append(name.split(c)[0])

    to_merge_per_set = []
    for k in all_stripped_names:
       to_merge_per_set.append([n for n in all_names if n.startswith(k)])
       if len(to_merge_per_set[0])==len(config['to_merge']):
          break

    #=======================================================================
    #=========================DO =========================
    #=======================================================================
    counter = 0
    #main loop for merging masks
    for this_set in to_merge_per_set:

       img = cv2.imread(config['label_folder']+os.sep+this_set[0]+'.png')
       msk = np.zeros((img.shape), dtype=np.uint8)
       del img

       ##allocate empty dictionary for classes and rgb colors
       class_dict = {}
       H = []

       msk, classes_names, rgb, classes_colors, _ = merge_labels(msk, \
                 cv2.imread(config['label_folder']+os.sep+this_set[0]+'.png'), config[class_sets[0]])

       ##update dictionary
       for c,r,h in zip(classes_names, rgb, classes_colors):
          class_dict[c] = r
          H.append(h)

       xcounter = 1
       for ii,cc in zip(this_set[1:], class_sets[1:]):
          msk, classes_names, rgb, classes_colors, _ = \
                  merge_labels(msk, cv2.imread(config['label_folder']+os.sep+ii+'.png'), config[cc])
          for c,r,h in zip(classes_names, rgb, classes_colors):
             class_dict[c] = r
             H.append(h)
          xcounter += 1

       class_str2 = ''
       for cc in class_str:
          class_str2+=cc

       outfile = config['label_folder']+os.sep+all_stripped_names[counter]+\
                 class_str2+'_rgb.csv'

       ##write class dict to csv file
       with open(outfile, 'w') as f:
          f.write("%s,%s,%s,%s\n" % ('class', 'r', 'g', 'b' ))
          for key in class_dict.keys():
             f.write("%s,%s\n" % (key, str(class_dict[key]).replace(')','').replace('(','')) )

       print("[INFO] Writing RGB image to %s" % (outfile))
       cv2.imwrite(outfile.replace('_rgb.csv','_label_rgb.png'),
                   cv2.cvtColor(msk , cv2.COLOR_RGB2BGR) )


       ##get rgb colors
       cols = [class_dict[c] for c in class_dict.keys()]
       ## flatten 3d label image to 2d
       msk_flat = flatten_labels(msk, cols )
       msk_flat = msk_flat.astype('uint8')
       del msk

       ##===================================================
       imfile = glob(config['image_folder']+os.sep+\
                all_stripped_names[counter][:-1]+'*.*')[0]
       ##read image
       img, profile = OpenImage(imfile, config['im_order'])

       ##===================================================
       ## make a matplotlib overlay plot
       resr = msk_flat.astype('float')
       resr[resr<1] = np.nan
       resr = resr-1

       try:
          alpha_percent = config['alpha']
       except:
          alpha_percent = 0.5

       new_cols = []
       for col in H:
          if not col.startswith('#'):
             col = '#'+col
          new_cols.append(col)
       cmap = colors.ListedColormap(new_cols)

       fig = plt.figure()
       ax1 = fig.add_subplot(111)
       ax1.get_xaxis().set_visible(False)
       ax1.get_yaxis().set_visible(False)

       if np.ndim(img)==3:
          _ = ax1.imshow(img)
       else:
          _ = ax1.imshow(img)

       im2 = ax1.imshow(resr,
                     cmap=cmap,
                     alpha=alpha_percent, interpolation='nearest',
                     vmin=0, vmax=len(new_cols))
       divider = make_axes_locatable(ax1)
       cax = divider.append_axes("right", size="5%", pad="2%")
       cb=plt.colorbar(im2, cax=cax)
       cb.set_ticks(0.5 + np.arange(len(new_cols) + 1))
       cb.ax.set_yticklabels(list(class_dict.keys()) , fontsize=6)

       outfile = outfile.replace('_rgb.csv', '_merged_mres.png')

       plt.savefig(outfile,
                dpi=300, bbox_inches = 'tight')
       del fig; plt.close()

       counter += 1
