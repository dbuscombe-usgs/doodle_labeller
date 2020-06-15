import tifffile
import cv2
import numpy as np

# read RGB image
im1 = cv2.imread('data/images/4_rgb.png')

#read elevation and get just the first band (if this is 3-band)
im2 = cv2.imread('data/images/4_elev.png')[:,:,0]

#(if you had a 1-band elevation image, it would be ...)
#im2 = cv2.imread('data/images/4_elev.png')

#merge bands
merged = np.dstack((im1, im2)) # creates a numpy array with 4 channels

#write file
tifffile.imsave('test.tiff', merged)

#read back in
merged = tifffile.imread('test.tiff')

# verify with 'shape' - should be 4 bands
merged.shape


gdal_translate -b 1 data/images/4_rgb.png red.png
gdal_translate -b 2 data/images/4_rgb.png green.png
gdal_translate -b 3 data/images/4_rgb.png blue.png
gdal_merge.py -separate  -o merged.tiff -co PHOTOMETRIC=MINISBLACK red.png green.png blue.png data/images/4_elev.png
