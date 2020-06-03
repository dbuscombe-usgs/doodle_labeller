# "Doodle Labeller"

> Daniel Buscombe, Marda Science daniel@mardascience.com

> Significant code contribution from LCDR Brodie Wells, Naval Postgraduate school Monterey

> Sample image provided by Christine Kranenburg , USGS St. Petersburg Coastal and Marine Science Center

This is a tool for partially supervised image segmentation and is based on code previously contained in the "dl tools" repository, https://github.com/dbuscombe-usgs/dl_tools

## Rationale
There are many great tools for exhaustive image labeling for segmentation tasks, using polygons. Examples include [makesense.ai](www.makesense.ai) and [cvat](https://cvat.org). However, for high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools is prohibitively time-consuming.

What is generally required is a semi-supervised tool for efficient image labeling, based on sparse examples provided by a human annotator.

The approach taken here is to freehand label only some of the scene, then use a model to complete the scene. Sparse annotations are provided to a Conditional Random Field (CRF) model, that develops a scene-specific model for each class and creates a dense (i.e. per pixel) label image based on the information you provide it. This approach can reduce the time required for detailed labeling of large and complex scenes by an order of magnitude or more.

This tool is also set up to tackle image labelling in stages, using minimal annotations. For example, by labeling individual classes then using the resulting binary label images as masks for the imagery to be labeled for subsequent classes. Labeling is acheived using the `doodler.py` script

Label images that are outputted by `doodler.py` can be merged using `merge.py`, which uses a CRF approach again to refine labels based on a windowing approach with 50% overlap. This refines labels further based on the underlying image.

## How to use

This is python software that is designed to be used from within a `conda` environment. After setting up that environment, the user places imagery in a folder and creates a `config` file that tells the program where the imagery is and what classes will be labeled. The minimum number of classes is 2. There is no limit to the maximum number of classes.

### create environment
```
conda env create -f doodler.yml
```

### Activate environment
```
conda activate doodler
```

### Add pictures
You will need separate folders for:
* Where to get the images that you want to label `data/images`. The program will assume you want to label all these images in one go. It's usually best to put one image in there at a time 
* Where to put a labeled images (annotation files in `.npy` format, the label images in png format, and a plot showing the image and label in png format)

### Make a config.json file
Several example config files are provided. A generic multi-label case would be similar to

	{
	  "image_folder" : "data/images",
	  "label_folder": "data/label_images",
	  "max_x_steps": 4,
	  "max_y_steps": 4,
	  "ref_im_scale": 0.8,
	  "lw": 5,
	  "im_order": "RGB",
	  "theta_col": 40,
	  "theta_spat": 5,
	  "n_iter": 10,
	  "compat_col": 40,
	  "compat_spat": 5,
	  "scale": 5,
	  "prob": 0.4,
	  "apply_mask": "None",
	  "classes": {
	   "Surf_Swash": "#fcf9f9",
	   "Water": "#3b81d0",
	   "BareSand": "#c0d03b",
	   "Grass_Shrub": "#62894d",
	   "WoodyVeg": "#395c27",
	   "MarshPlatform": "#94925c",
	   "Road": "#888c8c",
	   "Anthro": "#23ad96"
	 }
	}

where

* "image_folder" : ordinarily this would be "data/images" but could be a relative path to any relative directory
* "label_folder": see the above but for "data/label_images"
* "max_x_steps": number of tiles to split the image up in x direction (suggested range: 1- 6, depending on image size/complexity)
* "max_y_steps": number of tiles to split the image up in y direction (suggested range: 1- 6, depending on image size/complexity)
* "ref_im_scale": the proportion of the detected screen size to make the window (suggested: 0.5 -- 1.0)
* "lw": initial pen width in pixels. this can be changed on the fly (suggested range: 5 - 10, depending on image size)
* "im_order": either "RGB" or "BGR", depending on what type of imagery you have (RGB is more common)
* "theta_col": standard deviations for the location component of the colour-dependent term. (suggested range: 20 -- 60)
* "theta_spat": standard deviations for the location component of the colour-independent term. (suggested range: 1--10)
* "compat_col": label compatibilities for the colour-dependent term
* "compat_spat": label compatibilities for the colour-independent term
* "scale": spatial smoothness parameter (pixels)
* "prob": assumed probability of input labels	
* "n_iter": number of iterations of CRF inference.
* "classes": a dictionary of class names and associated colors as hex codes. There are various online color pickers including [this one](https://htmlcolorcodes.com/)
* "apply_mask": either `None` (if no pre-masking) or a list of binary label images with which to mask the image 

This file can be saved anywhere and called anything, but it must have the `.json` format extension.

Note that if you get errors reading your config file, it is probably because you have put commas where you shouldn't, or have left commas out

Also note that the program uses an average of label image estimated using the specified `theta_col` and `theta_spat`, 2 x `theta_col` and 2 x `theta_spat`, and 0.5 x `theta_col` and 0.5 x `theta_spat`. This increases performance, and lessens the pressure on the user to specify those parameters exactly. The suggested values work for most cases.


## Run doodler.py
Assuming you have already activated the conda environment (`conda activate doodler` - see above) ...

you will need to cd to the directory you will be working in, for example:

```
cd /Documents/doodle_labeller
```

Then run it with:

```
python doodler.py -c config_file.json
```

### Draw on the image
The title of the window is the label that will be associated with the pixels
you draw on, by holding down the left mouse button. After you are done with label press `escape` (Esc key, usually the top left corner on your keyboard). You can increase and decrease the brush width with `+ / -` keys. You can also undo a mistake with the `z` key.

* Use s to skip a square
* Use b to go back a square
* Use number keys to switch label

If your mouse has a scroll wheel, you can use that to zoom in and out. Other navigation options (zoom, pan, etc) are available with a right mouse click. 

### Dense labeling happens after a manual annotation session
After you have labeled each image tile for each image, the program will automatically use the CRF algorithm to
carry out dense (i.e. per-pixel) labelling based on the labels you provided and the underlying image

## Run merge.py

This script takes each individual mask image and merges them into one, keeping track of classes and class colors. This script takes a while to run, because it splits the merged image into small pieces that overlap by 50%, carries out a task-specific CRF-based label refinement on each, then averages the results. The idea is to refine the image by over-sampling. Finally, a median filter is applied with an 11-pixel radius circular structural element, to smooth high-frequency noise.

The amount of time the program takes is a function of the size of the image, and the number of CPU cores. The program uses all available cores, so machines with many cores will be faster. The program takes about 7 mins for a > 7000 x 4000 image using 8 Intel i7 2.9 GHz cores. Use of 16 cores of equivalent specification would approximately half the execution time, as would an image of half as many pixels.   

Assuming you have already activated the conda environment (`conda activate doodler` - see above) ...

you will need to cd to the directory you will be working in, for example:

```
cd /Documents/doodle_labeller
```

Then run it with:

```
python merge.py -c config_merge.json
```

An example config file is provided:

	{
	{
	  "image_folder" : "data/images",
	  "im_order": "RGB",
	  "theta_col": 40,
	  "theta_spat": 5,
	  "n_iter": 10,
	  "compat_col": 40,
	  "compat_spat": 5,
	  "scale": 5,
	  "prob": 0.4,  
	  "outfile": "data/label_images/2019-0830-195300-DSC04880-N7251F_merged_label.png",
	  
	  "to_merge": {
	  "water":  "data/label_images/2019-0830-195300-DSC04880-N7251F_water_no_water_label.png",
	  "veg": "data/label_images/2019-0830-195300-DSC04880-N7251F_vegetation_no_vegetation_label.png",
	  "anthro": "data/label_images/2019-0830-195300-DSC04880-N7251F_anthro_no_anthro_label.png",
	  "substrate": "data/label_images/2019-0830-195300-DSC04880-N7251F_sand_mud_gravel_boulders_snow_ice_wrack_peat_indeterminate_label.png"
	   },
	   
	  "classes1": 
	  {
	   "water": "#3b81d0",
	   "no_water": "#e35259"
	  },
	  
	  "classes2":
	  {
	   "vegetation": "#29f120",
	   "no_vegetation": "#e35259"
	  },  
	  
	  "classes3":
	  {
	   "anthro": "#f12063",
	   "no_anthro": "#e35259"
	  },  
	  
	  "classes4":
	  {
	   "sand": "#c0d03b",
	   "mud": "#62894d",
	   "gravel": "#395c27",
	   "boulders": "#94925c",
	   "snow_ice": "#888c8c",
	   "wrack_peat": "#23ad96",
	   "indeterminate": "#9e289c"
	   }
	}

where most of the fields are the same as above, except

* "to_merge": a list of label images (generated by doodler.py) to merge
* "classes1", "classes2", etc: these are the class names and hex color codes associated with each label image in "to_merge", in order


<!-- 
## compiling doodler.py
- conda activate doodler
- pip install --upgrade 'setuptools<45.0.0'
- pip install python-opencv-headless
- pip install pyinstaller
- pyinstaller --onefile --noconfirm doodler.py --clean --hidden-import pydensecrf.eigen
- conda deactivate
- ./dist/doodler -->

## Improvements coming soon
* support for geotiffs
* compiled executables
