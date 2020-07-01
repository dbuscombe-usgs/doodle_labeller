# "Doodle Labeller"

> Daniel Buscombe, Marda Science daniel@mardascience.com

> Developed for the USGS Coastal Marine Geology program, as part of the Florence Supplemental project

> This is a "Human-In-The-Loop" machine learning tool for partially supervised image segmentation and is based on code previously contained in the "dl tools" [repository](https://github.com/dbuscombe-usgs/dl_tools)

> Significant code contribution from LCDR Brodie Wells, Naval Postgraduate school Monterey

> Sample images provided by 1) Christine Kranenburg, USGS St. Petersburg Coastal and Marine Science Center, 2) Chris Sherwood, USGS Woods Hole Coastal and Marine Science Center, and 3) Jenna Brown, USGS MD-DE-DC Water Science Center

> The Conditional Random Field (CRF) model used by this tool is described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244)


## Rationale
There are many great tools for exhaustive (i.e. whole image) image labeling for segmentation tasks, using polygons. Examples include [makesense.ai](www.makesense.ai) and [cvat](https://cvat.org). However, for high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. This is especially true of scenes with many classes of interest, and covering relatively small, spatially discontinuous regions of the image.

What is generally required in the above case is a semi-supervised tool for efficient image labeling, based on sparse examples provided by a human annotator. Those sparse annotations are used by a secondary automated process to estimate the class of every pixel in the image. The number of pixels annotated by the human annotator is typically a small fraction of the total pixels in the image.  

`Doodler` is a tool for "exemplative", not exhaustive, labeling. The approach taken here is to freehand label only some of the scene, then use a model to complete the scene. Sparse annotations are provided to a Conditional Random Field (CRF) model, that develops a scene-specific model for each class and creates a dense (i.e. per pixel) label image based on the information you provide it. This approach can reduce the time required for detailed labeling of large and complex scenes by an order of magnitude or more.

This tool is also set up to tackle image labeling in stages, using minimal annotations. For example, by labeling individual classes then using the resulting binary label images as masks for the imagery to be labeled for subsequent classes. Labeling is achieved using the `doodler.py` script

Label images that are outputted by `doodler.py` can be merged using `merge.py`, which uses a CRF approach again to refine labels based on a windowing approach with 50% overlap. This refines labels further based on the underlying image.


## Terminology

* image: a 1 or 3-band geotiff, or 1 or 3-band jpg/jpeg/JPG/JPEG, png/PNG, or tiff/tif/TIF/TIFF
* class: a pre-determined category
* label: an annotation made on an image attributed to a class (either manually or automatically - in this program, annotations are manual)
* label image: an image where each pixel has been labeled with a class (either manually or automatically - in this program, the label image is generated automatically using a machine learning algorithm called a CRF)
* binary label image: a label image of 2 classes (the class of interest and everything else)

## How to use

1. Put a single image or multiple images in the "data/images" folder

2. (optionally) Clear the contents of the "data/label_images" folder

3. make a `config` file (see below)

4. make 'sparse' annotations on the imagery with a mouse or stylus, and use keyboard keys to advance through classes

This is python software that is designed to be used from within a `conda` environment. After setting up that environment, the user places imagery in a folder and creates a `config` file that tells the program where the imagery is and what classes will be labeled. The minimum number of classes is 2. There is no limit to the maximum number of classes.

This tool can be used in a few different ways, including:

1. Create a label image in one go, by defining all classes at once in a single `config` file

2. Create a label image in stages, by defining subsets of classes in multiple `config` files (then optionally merging them afterwards using `merge.py`)

The second option is possibly more time-efficient for complex scenes. It might also be advantageous for examining the error of labels individually.

The software can be used to:

1. label single or multiple images at once, with 2 or more classes, with and without pre-masking
2. merge multiple label images together, each with with 2 or more classes

The program will read 1- and 3-band imagery in all common file formats supported by [opencv](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html). 4+ band imagery must be in formats supported by the [tifffile](https://pypi.org/project/tifffile/) library. Geotiff imagery is read and written using [rasterio](https://rasterio.readthedocs.io/en/latest/)


### Getting set up video

[This video](https://drive.google.com/file/d/111Ezrz0lU_UAFt6J9RMo8LLaVbEGgW-_/view?usp=sharing) demonstrates how to download the software and install the conda environment

### Labeling videos

I made a series of videos that demonstrate how to use the software using example data provided in this repository.

1. [Video 1](https://drive.google.com/file/d/184XUSYgHKD2QRjyR1oUnzEDfeBmKDNaK/view?usp=sharing): Creating a binary mask of water / no water, using no pre-masking (on two sample images). No audio
2. [Video 2](https://drive.google.com/file/d/1oiIBWeANwr23qKnJqOf6fpFUZSuQTVq4/view?usp=sharing): Creating a binary mask of vegetation / no vegetation, using pre-masking (on two sample images). No audio
3. [Video 3](https://drive.google.com/file/d/1TkU2cxy7sLdEPAF9cXee4Eo7VtDE9RZs/view?usp=sharing): Creating a multiclass label images of substrates, using pre-masking (on two sample images). No audio
4. [Video 4](https://drive.google.com/file/d/1QwKWklMF8lop0uVqWxWX1NEtvDcjAnrU/view?usp=sharing): Merging multiclass label images of substrates with binary masks of water, and vegetation (on two sample images)
5. [Video 5](https://drive.google.com/file/d/1_z2reRdOnjdhaRworvf9J8-vIGIJt9ht/view?usp=sharing): Using `doodler_optim.py` to redo a set of class labels (on one sample image). Audio


### Clone the github repo

```
git clone --depth 1 https://github.com/dbuscombe-usgs/doodle_labeller.git
```

### create environment

If you are a regular conda user, now would be a good time to

```
conda clean --all
conda update conda
conda update anaconda
```

Issue the following command from your Anaconda shell, power shell, or terminal window:

```
conda env create -f doodler.yml
```

### Activate environment

Use this command to activate the environment, in order to use it

```
conda activate doodler
```

If you are a Windows user (only) who wishes to use unix style commands, install `m2-base`

```
conda install m2-base
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
	  "lw": 15,
	  "im_order": "RGB",
	  "alpha": 0.5,
	  "num_bands": 3,
	  "create_gtiff": "false",
	  "apply_mask": ["water", "veg"],
	  "classes": {
	   "sand": "#c0d03b",
	   "roads": "#696669",
	   "buildings": "#de3e5d",
	   "people": "#74346b",
	   "cars": "#dd9c1a"
	   }
	}

where the required arguments are:

* `image_folder` : ordinarily this would be "data/images" but could be a relative path to any relative directory
* `label_folder`: see the above but for "data/label_images"
* `max_x_steps`: number of tiles to split the image up in x direction (suggested range: 1- 6, depending on image size/complexity)
* `max_y_steps`: number of tiles to split the image up in y direction (suggested range: 1- 6, depending on image size/complexity)
* `ref_im_scale`: the proportion of the detected screen size to make the window (suggested: 0.5 -- 1.0)
* `lw`: initial pen width in pixels. this can be changed on the fly (suggested range: 5 - 10, depending on image size)
* `im_order`: either "RGB" or "BGR", depending on what type of imagery you have (RGB is more common)
* `classes`: a dictionary of class names and associated colours as hex codes. There are various online colour pickers including [this one](https://htmlcolorcodes.com/)


and the optional arguments and their default values:

* `apply_mask`: either `None` (if no pre-masking) or a list of label names with which to mask the image. These label images should already exist *Default = None*
* `num_bands`: the number of bands in the input imagery (e.g. 1, 3, or 4). *Default = 3*
* `create_gtiff`: if "true", the program will create a geotiff label image - only appropriate if the input image is also geotiff. Otherwise, "false" *Default = false*
* `alpha`: the degree of transparency in merged image-output plots *Default = 0.5*
* `fact`: the factor by which to downsample by imagery. *Default = 5* (this might seem large, but trust me the CRF is very cpu and memory intensive otherwise, and the results work with a large `fact` turn out ok. Reduce to get finer resolution results by be warned, it will take a lot longer. )
* `compat_col`: compatibility value for the color/space-dependent term in the model, initial value for search. *Default = 20*
* `theta_col`: standard deviation value for the color/space-dependent term in the model, initial value for search. *Default = 20*

This file can be saved anywhere and called anything, but it must have the `.json` format extension.

Note that if you get errors reading your config file, it is probably because you have put commas where you shouldn't, or have left commas out

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

or optionally

```
python doodler.py -c path/to/config_file.json -f path/to/npy_file
```

Command line arguments:

* `-h`: print a help message to screen, then exit
* `-c`: for specification of a `config` file. Required.
* `-f`: pass a `.npy` file to the program (made using `doodler.py` or `doodler_optim.py` during a previous session). Optional


### Draw on the image
The title of the window is the label that will be associated with the pixels
you draw on, by holding down the left mouse button.

* After you are done with label (or if you need to skip a class because it is not present in the current tile) press `escape` (`Esc` key, usually the top left corner on your keyboard).
* You can increase and decrease the brush width with (number) `1 / 2` keys.
* You can also undo a mistake with the `z` key.
* Use `s` to skip forward a square (note that this will not record any labels done on the current square - this feature is for squares with no labels to make)
* Use `b` to go back a square

If your mouse has a scroll wheel, you can use that to zoom in and out. Other navigation options (zoom, pan, etc) are available with a right mouse click.

### Dense labeling happens after a manual annotation session
After you have labeled each image tile for each image, the program will automatically use the CRF algorithm to carry out dense (i.e. per-pixel) labelling based on the labels you provided and the underlying image

## Run merge.py

This script takes each individual mask image and merges them into one, keeping track of classes and class colours. It optimizes the CRF parameters for each chunk. This script takes a while to run, because it splits the merged image into small pieces, carries out a task-specific CRF-based label refinement on each, then averages the results. The idea is to refine the image by over-sampling.

The amount of time the program takes is a function of the size of the image, and the number of CPU cores. The program uses all available cores, so machines with many cores will be faster.

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
	  "image_folder" : "data/images",
	  "label_folder": "data/label_images",
	  "im_order": "RGB",
	  "alpha": 0.5,
	  "num_bands": 3,
	  "create_gtiff": "false",

	  "to_merge": ["water","veg","sand"],

	  "classes1":
	  {
	   "water": "#3b81d0",
	   "no_water": "#e35259"
	  },

	  "classes2":
	  {
	    "veg_dune_grass": "#6eaf29",
	    "veg_other_grass":"#13dc19",
	    "veg_woody": "#1a7120",
	    "no_veg": "#e35259"
	  },

	  "classes3":
	  {
	    "sand": "#c0d03b",
	    "roads": "#696669",
	    "buildings": "#de3e5d",
	    "people": "#74346b",
	    "cars": "#dd9c1a"
	   }
	}

The required arguments are:

* `to_merge`: a list of label image sets (generated by `doodler.py`) to merge
* `classes1`, `classes2`, etc: these are the class names and hex colour codes associated with each label sets in `to_merge`, in order.  There are various online colour pickers including [this one](https://htmlcolorcodes.com/)
* `im_order`: either "RGB" or "BGR", depending on what type of imagery you have (RGB is more common)
* `image_folder` : ordinarily this would be "data/images" but could be a relative path to any relative directory
* `label_folder`: see the above but for "data/label_images"
* `ref_im_scale`: scalar to control the size of the image and label windows *Default = 0.8*

and the optional arguments and their default values:

* `num_bands`: the number of bands in the input imagery (e.g. 1, 3, or 4). *Default = 3*
* `create_gtiff`: if "true", the program will create a geotiff label image - only appropriate if the input image is also geotiff. Otherwise, "false" *Default = false*
* `alpha`: the degree of transparency in merged image-output plots *Default = 0.5*
* `fact`: the factor by which to downsample by imagery. *Default = 5* (this might seem large, but trust me the CRF is very cpu and memory intensive otherwise, and the results work with a large `fact` turn out ok. Reduce to get finer resolution results by be warned, it will take a lot longer. )
* `compat_col`: compatibility value for the color/space-dependent term in the model, initial value for search. *Default = 20*
* `theta_col`: standard deviation value for the color/space-dependent term in the model, initial value for search. *Default = 20*
* `medfilt`: whether or not to apply a median filter to smooth the results *Default = "true"*


## Got a bad result?

Did you get a disappointing result from `doodler.py`?

You could try using `doodler_optim.py`. This script works in the same way as `doodler.py` but attempts to help with problematic imagery. It performs a much wider hyperparameter search

You can redo the annotations:

```
python doodler_optim.py -c config_file.json
```

or optionally you can pass your previous annotations instead of having to redo annotations

```
python doodler_optim.py -c path/to/config_file.json -f path/to/npy_file
```

Command line arguments:

* `-h`: print a help message to screen, then exit
* `-c`: for specification of a `config` file. Required.
* `-f`: pass a `.npy` file to the program (made using `doodler.py` or `doodler_optim.py` during a previous session). Optional

Note that `doodler_optim.py` takes a lot longer to estimate the CRF solution compared to `doodler.py`, but the result should be better.

<!-- Why redo the labelling? You'll be more careful this time :)

Just joking - it's top to the following list ... -->


## Improvements coming soon
* fix the line width issue on the current image tile
* compiled executables
* lookup table for consistent hex colours for common classes
* config file generator (need a GUI?)
* move labelling window to different screen option?
* output image of your doodles, optionally

### Known issues

This software is undergoing development and is subject to frequent change. In particular, 1d and 4d imagery has not been extensively tested so please expect bugs. Proper documentation is forthcoming.

I know the brush thickness buttons doesn't change on the present tile - I am working on a fix


## FAQS

* *Why are my results poor?*
> Some things to check: 1) the default 'fact' is set quite high to deal with really large imagery and/or small computer RAM (<< 16 GB) and/or large imagery (>8,000 pixels in any dimension). If your system can handle it (it probably can), set the number low in the config file e.g. `'fact': 2,` for a downsize factor of 2, and `'fact': 1,` for no downsizing. 2) did you run `doodler_optim.py`? 3) Increase one or both of the CRF parameters in the config file, e.g. change from the default `'compat_col': 20,` and/or `'theta_col': 20,` to something like `'compat_col': 40,` and/or `'theta_col': 40,`

* *Why is my window so small?*
> Try using a large number for `ref_im_scale` in the config file, increasing it from the default value of 0.8 to, say, 1.0 or even 1.2

* *What to I do with a `json.decoder.JSONDecodeError`?*
> This relates to the config file. You either have a comma at the end of a list (bounded by `{...}`) where it shouldn't be, or you have a variable outside of `"..."`


To ask a different question, please use the `Issues` tab (see above) - please do not email me

## Contributing
If you'd like to discuss a new feature, use the `Issues` tab. If you'd like to submit code improvements, please submit a `pull request`



## Version history

### 7/1/20
1. The probability of the label is now computed and outputted in the same formats as the label images. This probability will be useful for purposes such as further label refinement, error propagation, etc
2. Bug fixes, and more tests into default CRF hyperparameters


### 6/24/20

1. now you can pass annotation files (`.npy`) to both `doodler_optim.py` and `doodler.py` if you want to 'redo' the CRF inference. This might be 1) using `doodler_optim.py` with a `.npy` file previously `doodler.py`, or 2) using `doodler.py`, but this time overriding any defaults or with different `config` parameters
2. progress bar (!) in all scripts
3. more memory/efficiency improvements

### 6/20/20

1. CRF inference is now faster and hopefully more robust
2. better use of memory
3. smaller output file sizes
4. masking is now specified by type/name rather than filename, allowing for merging of multiple sets of image labels that conform to a common pattern
5. config files are now easier to construct because no hard-coding of file paths, and no CRF parameter specification by default (although you can still specify the parameters)
6. labeling still happens in chunks, but inference uses merged chunks for smaller imagery (<10000 px in all dimensions)
7. `doodler.py` now automatically computes the optimal CRF hyperparameters for each image chunk. I have done a lot of research into the sensitivity of results to input hyperparameters. The variation can be massive; therefore I have hard-coded some values in, implemented formulas for others, and allow the program to attempt to search for the remaining hyperparameters values. Seems to work ok in tests but please report - work still in progress
8. new `doodler_optim.py` is for redoing imagery, this time with an expanded hyperparameter search. Use in an emergency (requires redoing sparse labels for the offending image/class set) - work still in progress (subject to change)
9. the optimization of the hyperparameters happens on subsampled imagery (unless any image chunk dimension is less than 2000 pixels), which is faster and less memory intensive and still results in sensible outputs
10. periods (other than to specify the file extension) are now allowed in input image file names
11. `merge.py` will only use chunks if any image dimension is less than 10000 pixels
12. `merge.py` now makes a semi-transparent overlay figure (like `doodler.py` does)




<!--
## compiling doodler.py
- conda activate doodler
- pip install --upgrade 'setuptools<45.0.0'
- pip install python-opencv-headless
- pip install pyinstaller
- pyinstaller --onefile --noconfirm doodler.py --clean --hidden-import pydensecrf.eigen --hidden-import rasterio._shim --hidden-import rasterio.control --hidden-import pkg_resources.py2_warn
- conda deactivate
- ./dist/doodler -->
