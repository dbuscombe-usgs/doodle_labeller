# merge label rasters
#
# > Daniel Buscombe, Marda Science daniel@mardascience.com
# > USGS Pacific Marine Science Center
#
import os, json
import sys, getopt
import cv2
import numpy as np

#===============================================================
def merge_labels(msk, img, classes):

    classes_colors = [classes[k] for k in classes.keys() if 'no_' not in k]
    classes_codes = [i for i,k in enumerate(classes) if 'no_' not in k]
	
    rgb = []
    for c in classes_colors:
       rgb.append(tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
	
    counter = 0
    for k in rgb:
       ind = (img[:,:,0]==classes_codes[counter])
       msk[ind] = k 
       counter += 1	   
    return msk



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
		
    to_merge = [config['to_merge'][k] for k in config['to_merge'].keys()]	

    class_sets = [c for c in config.keys() if c.startswith('class')]	

    img = cv2.imread(to_merge[0])
    msk = np.zeros((img.shape), dtype=np.uint8)
		
    msk = merge_labels(msk, img, config[class_sets[0]])
	
    for ii,cc in zip(to_merge[1:], class_sets[1:]):
       msk = merge_labels(msk, cv2.imread(ii), config[cc])
	
    cv2.imwrite(config['outfile'], cv2.cvtColor(msk, cv2.COLOR_RGB2BGR) )
	
	
	
















