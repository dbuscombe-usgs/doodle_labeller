
import time
import subprocess, ctypes
import os, json
import sys, getopt
import cv2
import numpy as np
from glob import glob
import matplotlib

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =========================================================

def FlipAxForMPL(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    newim = np.dstack([b, g, r])
    return newim

# =========================================================
def PlotAndSave(img, resr, crf_output, name, config, counter):
    mpl_im = FlipAxForMPL(img.copy())

    sp = 120

    a_label = 'a) Input'
    b_label = 'b) CRF prediction'
    c_label = 'b) CRF prediction'
    alpha_percent = 0.4

    cmap = colors.ListedColormap(list(config['classes'].values()))

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.4)
    ax1 = fig.add_subplot(sp + 1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    _ = ax1.imshow(mpl_im)
    plt.title(a_label, loc='left', fontsize=12)

    ax1 = fig.add_subplot(122)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    _ = ax1.imshow(mpl_im)
    plt.title(c_label,
              loc='left',
              fontsize=12)
    im2 = ax1.imshow(resr,
                     cmap=cmap,
                     alpha=alpha_percent,
                     vmin=0, vmax=len(config['classes']))
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%")
    cb=plt.colorbar(im2, cax=cax)
    cb.set_ticks(0.5 + np.arange(len(config['classes']) + 1))
    cb.ax.set_yticklabels(config['classes'])
    #cb.ax.tick_config(labelsize=4)

    outfile = config['label_folder']+os.sep+name + '_split_'+ \
                   str(counter)+'_of_'+str(len(Z))+'_mres.png'

    plt.savefig(outfile,
                dpi=600, bbox_inches = 'tight')
    del fig; plt.close()

    outfile = config['label_folder']+os.sep+name + '_split_'+ \
                   str(counter)+'_of_'+str(len(Z))+'_label.png'

    cv2.imwrite(outfile,
                np.round(255*(resr/np.max(resr))).astype('uint8'))

# =========================================================

def DoCrf(o_img, out, config, name, counter):
    im = o_img.copy()
    print('Generating dense scene from sparse labels...')
    res, _ = getCRF_justcol(im,
                            out.astype('int'),
                            config['classes'])

    res = median(res, disk(6))

    Lcorig = out.copy().astype('float')
    Lcorig[Lcorig<1] = np.nan

    PlotAndSave(o_img, res, out.astype('int'), name, config, counter)

# =========================================================

def getCRF_justcol(img, Lc, label_lines):
    if np.ndim(img) == 2:
         img = np.dstack((img, img, img))
    H = img.shape[0]
    W = img.shape[1]
    d = dcrf.DenseCRF2D(H, W, len(label_lines) + 1)
    U = unary_from_labels(Lc.astype('int'),
                          len(label_lines) + 1,
                          gt_prob=config['prob'])
    d.setUnaryEnergy(U)
    feats = create_pairwise_bilateral(sdims=(config['theta'], config['theta']),
                                      schan=(config['scale'],
                                             config['scale'],
                                             config['scale']),
                                      img=img,
                                      chdim=2)
    d.addPairwiseEnergy(feats, compat=config['compat_col'],
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(config['n_iter'])
    preds = np.array(Q, dtype=np.float32).reshape(
                     (len(label_lines) + 1, H, W)).transpose(1, 2, 0)
    preds = np.expand_dims(preds, 0)
    preds = np.squeeze(preds)

    return np.argmax(Q, axis=0).reshape((H, W)), preds

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
                                   int(len(self.config['classes']) + 1)))
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
        print("screen size =", self.screen_size)
        print("im_shape =", imshape)
        print("min(rh, rw) =", min(rh, rw))
        print("dim =", dim)
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
        #                 (src , overlay):
        """
        Returns a new image to display, after blending in the pixels that have
            been labeled

        Takes
        src : array
            Original image
        overlay : array
            Overlay image, in this case the Labels

        Returns
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
        print("Initial brush width = 5")
        print("  -change using the +/- keys")
        print("Cycle classes with [ESC]")
        #print("Subtract mean with [Space]")
        print("Go back a frame with [b]")
        print("Skip a frame with [s]")
        print("\nTo navigate labels use:\nButton: Label")

        nav_b = "123456789qwerty"
        for labl, button in enumerate(self.config['classes'].keys()):
            print(button + ':', nav_b[labl])
        nav_b = nav_b[:len(self.config['classes'])]
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
                          (255, 255, 0), 2)

            cv2.namedWindow('whole image', cv2.WINDOW_NORMAL)
            cv2.imshow('whole image', ref_img)
            cv2.resizeWindow('whole image',
                             (self.WinScales(ref_img.shape[:2])))
            cv2.moveWindow('whole image', 0, 28)
            nav = False   # Navigator variable
            if not lab:
                counter = 1   # Label number
            sm = 0        # Enhancement variable
            if np.std(self.im_sect) > 0:
                s = np.shape(self.im_sect[:, :, 2])
                if not lab:
                    # TODO: Lc should never be set to zeros!!
                    #      It needs to get from class mask, so that it can
                    #      keep the labels that have been done
                    #      Need to figure out what that means for lab and nav
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

                        if chr(k) in nav_b:  # if label number pressd, go to it
                            nav = True
                            lab = True
                            ck -= 1
                            counter = nav_b.find(chr(k)) + 1
                            break

                        if k == ord("s"):  # If s is pressed, skip square
                            nav = True
                            break

                        if k == ord('b'):  # If b is pressed go back a square
                            nav = True
                            ck -= 2
                            break

                        if k == ord('-'):  # If + is pressed, increase brush wi
                            self.size += 1
                            print("brush width = " + str(self.size))

                        if k == ord('-'):  # If - is pressed, decrese brush wid
                            self.size -= 1
                            if self.size < 1:
                                self.size = 1
                            print("brush width = " + str(self.size))

                    cv2.destroyWindow(label)

                if not nav:
                    self.class_mask[self.Z[ck][0]:self.Z[ck][1],
                                    self.Z[ck][2]:self.Z[ck][3], :] = Lc
                    lab = False
            else:
                if self.config['auto_class'] == "No":
                    ac = 0
                else:
                    ac = list(self.config['classes'].keys()).index('BackGrnd') + 1
                self.class_mask[self.Z[ck][0]:self.Z[ck][1],
                                self.Z[ck][2]:self.Z[ck][3], :] = \
                    np.ones((self.im_sect.shape[0],
                            self.im_sect.shape[1],
                            self.class_mask.shape[2])) * ac

            cv2.destroyWindow('whole image')
            ck += 1
        return np.argmax(self.class_mask, axis=2)

# =========================================================

def TimeScreen():
    """
    Starts a timer and gets the screen size. I realize these should be seperate
      functions, but the os name is here, so might as well. Maybe this should
      seperated in the future.
    Takes:
        nothing
    Returns:
        start : datetime stamp
        screen_size : tuple of screen size
    """

    # start timer and get screen size
    if os.name == 'posix':  # true if linux/mac or cygwin on windows
        start = time.time()
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
        start = time.clock()
        user32 = ctypes.windll.user32
        screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return start, screen_size

# =========================================================

def OpenImage(image_path, im_order):
    """
    Opens the image of any type I know of
    Returns the image in numpy array format
    Takes:
        image_path : string
            Full or relative path to image
        config : dict
            Dictionary of parameters set in the parameters file
    Returns:
        numpy array of image 2D or 3D #NOTE: want this to do multispectral
    """
    if image_path.lower()[-3:] == 'tif':
        img = WF.ReadGeotiff(image_path, im_order)
    else:
        img = cv2.imread(image_path)
        if im_order=='RGB':
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif im_order=='BGR':
           img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


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
if __name__ == '__main__':

    # argv = sys.argv[1:]
    # try:
    #     opts, args = getopt.getopt(argv,"h:c:")
    # except getopt.GetoptError:
    #     print('python doodler.py -c configfile.json')
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == '-h':
    #         print('Example usage: python doodler.py -c config.json')
    #         sys.exit()
    #     elif opt in ("-c"):
    #         configfile = arg

    configfile = 'config.json'
    # load the user configs
    with open(os.getcwd()+os.sep+configfile) as f:
        config = json.load(f)

    # for k in config.keys():
    #     exec(k+'=config["'+k+'"]')

    files = sorted(glob(os.path.normpath(config['image_folder']+os.sep+'*.*')))

    N = []
    counter = 1
    for f in files:

        start, screen_size = TimeScreen()
        o_img = OpenImage(f, config['im_order'])

        name = f.split(os.sep)[-1].split('.')[0]
        N.append(name)

        mp = MaskPainter(o_img.copy(), config, screen_size)
        out = mp.LabelWindow()

        outfile = config['label_folder']+os.sep+name + '_split_'+ \
                       str(counter)+'_of_'+str(len(Z))+'.npy'
        np.save(outfile, out)
        print("annotations saved to %s" % (outfile))

        outfile = config['label_folder']+os.sep+name + '_split_'+ \
                       str(counter)+'_of_'+str(len(Z))+'_im.npy'
        np.save(outfile, o_img)
        print("annotations saved to %s" % (outfile))

    counter += 1

    for name in N:
        print("Sparse labelling complete ...")
        print("Dense labelling ... this may take a while")
        label_files = sorted(glob(config['label_folder']+os.sep+name +'*.npy'))
        im_files = sorted(glob(config['label_folder']+os.sep+name +'*_im.npy'))

        counter = 1
        for l,f in zip(label_files, im_files):
            out = np.load(l)
            im = np.load(f)
            DoCrf(im, out, config, name, counter)
            counter += 1

        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # x
