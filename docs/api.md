---
id: api
title: User guide
---

## Reference

The Conditional Random Field (CRF) model used by this tool is described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244)

## Terminology

* image: a 1 or 3-band geotiff, or 1 or 3-band jpg/jpeg/JPG/JPEG, png/PNG, or tiff/tif/TIF/TIFF
* class: a pre-determined category
* label: an annotation made on an image attributed to a class (either manually or automatically - in this program, annotations are manual)
* label image: an image where each pixel has been labeled with a class (either manually or automatically - in this program, the label image is generated automatically using a machine learning algorithm called a CRF)
* binary label image: a label image of 2 classes (the class of interest and everything else)


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
	  "alpha": 0.5,
	  "create_gtiff": "false",
	  "apply_mask": "None",
	  "classes": {
	   "vegetation": "#0bec0b",
	   "anthro": "#f12063",
	   "road_pavement": "#7c7878",
	   "sand": "#c0d03b",
	   "mud": "#62894d",
	   "marsh": "#1c6d1d",
	   "gravel": "#395c27",
	   "boulders": "#94925c",
	   "snow_ice": "#f6faf6",
	   "wrack_peat": "#23ad96",
	   "indeterminate": "#9e289c"
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
* `create_gtiff`: if "true", the program will create a geotiff label image - only appropriate if the input image is also geotiff. Otherwise, "false" *Default = false*
* `alpha`: the degree of transparency in merged image-output plots *Default = 0.5*
* `fact`: the factor by which to downsample by imagery. *Default = 5* (this might seem large, but trust me the CRF is very cpu and memory intensive otherwise, and the results work with a large `fact` turn out ok. Reduce to get finer resolution results by be warned, it will take a lot longer. )
* `compat_col`: compatibility value for the color/space-dependent term in the model, initial value for search. *Default = 120*
* `theta_col`: standard deviation value for the color/space-dependent term in the model, initial value for search. *Default = 100*
* `optim`: if `True`, will search through more hyperparameters and will take longer. For problem imagery

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
* `-f`: pass a `.npy` file to the program (made using `doodler.py` during a previous session). Optional


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

## [Optionally] Run merge.py

This script takes each individual mask image and merges them into one, keeping track of classes and class colours.

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
	  "create_gtiff": "true",
	  "name": "2018_12_ldi_15cm_utm_0_0",

	  "to_merge": ["deep","anthro"],

	  "classesA": {
	    "deep": "#2f36aa",
	    "whitewater": "#E317CE",
	    "shallow": "#b3946b",
	    "no_water": "#e35259"
	  },

	  "classesB": {
	    "anthro": "#aa352f",
	    "vegetation": "#2faa58",
	    "mud": "#aa8b2f",
	    "sand": "#a0aa2f",
	    "submerged_sediment": "#84aa2f",
	    "gravel": "#186074",
	    "boulder": "#4f7373",
	    "wrack_peat": "#aa2f9c",
	    "snow_ice": "#aa2f69",
	    "shell": "#84b1b4"
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

* `create_gtiff`: if "true", the program will create a geotiff label image - only appropriate if the input image is also geotiff. Otherwise, "false" *Default = false*
* `alpha`: the degree of transparency in merged image-output plots *Default = 0.5*
* `fact`: the factor by which to downsample by imagery. *Default = 5* (this might seem large, but trust me the CRF is very cpu and memory intensive otherwise, and the results work with a large `fact` turn out ok. Reduce to get finer resolution results by be warned, it will take a lot longer. )
* `compat_col`: compatibility value for the color/space-dependent term in the model, initial value for search. *Default = 20*
* `theta_col`: standard deviation value for the color/space-dependent term in the model, initial value for search. *Default = 20*
* `medfilt`: whether or not to apply a median filter to smooth the results *Default = "true"*

## Got file spaces in your file names?

That's a problem for doodler. On unix based file systems:

`find -name "* *" -type f | rename 's/ /_/g'`


## Got a bad result?

Did you get a disappointing result from `doodler.py`?

You could try using `optim = True` in the config file, to help with problematic imagery. It performs a much wider hyperparameter search

You can redo the annotations:

```
python doodler.py -c config_file.json
```

or optionally you can pass your previous annotations instead of having to redo annotations

```
python doodler.py -c path/to/config_file.json -f path/to/npy_file
```

Command line arguments:

* `-h`: print a help message to screen, then exit
* `-c`: for specification of a `config` file. Optional - program will prompt you for it if it is missing from command line arguments
* `-f`: pass a `.npy` file to the program (made using `doodler.py` during a previous session). Optional


## CRF implementation details

1. if `optim` is `True`, the probability of your annotation is set to 0.8, and a set of 10 factors are defined that modify `theta_col` and `compat_col`. If `optim` is `False`, the probability of your annotation is set to 0.9, and a set of 5 factors are defined that modify `theta_col` and `compat_col`.
2. per-class weights are computed as inverse relative frequencies. The number of each pixel in each annotation class is computed and normalized. If the maximum relative frequency is more than 5 times the minimum, the maximum is halved and the relative frequencies re-normalized before their inverses are used as weights in the CRF model.
3. the CRF `scale` parameter is determined as 1+(5 * (`M` / 3681)) where `M` is the maximum image dimension
4. the hyperparameters `theta_col` and `compat_col` are modified by the factors described above. The 5 factors are .5, .75, 1, 1.25, and 1.5, therefore if `theta_col` is 100, the program would cycle through creating a CRF realization based on  `theta_col` = 50, `compat_col` = 50, then 75, 100, and 140. The 10 `optim` factors are .5, .66, .75, 1, 1.25, 1.33, and 2.
5. The predictions (softmax scores based on normalized logits) per class are computed as the weighted average of the per-class predictions compiled over the number of hyperparameter factors.
6. Each per-class prediction raster is then median-filtered using a disk-shaped structural element with a radius of 10*(M/(3681)) pixels
7. To make the final classification, if the per-class prediction is > .5 (for binary) or > 2*(1/`N`) where `N`=number of classes (for multiclass), the pixel is encoded that label. This works in order of classes listed in the config file, so a latter class could override a former one. Where no grid cell is encoded with a label, such as where the above criteria are not met, the label is the argument maximum for for that cell in the stack.


## Improvements coming soon
* fix the line width issue on the current image tile
* compiled executables or web version
* lookup table for consistent hex colours for common classes, or assign colors based on name?
* move labelling window to different screen option?

### Known issues

This software is undergoing development and is subject to frequent change. In particular, 1d and 4d imagery has not been extensively tested so please expect bugs. Proper documentation is forthcoming.

I know the brush thickness buttons doesn't change on the present tile - I am working on a fix


## FAQS

* *Why are my results poor?*
> Some things to check: 1) the default 'fact' is set quite high to deal with really large imagery and/or small computer RAM (<< 16 GB) and/or large imagery (>8,000 pixels in any dimension). If your system can handle it (it probably can), set the number low in the config file e.g. `'fact': 2,` for a downsize factor of 2, and `'fact': 1,` for no downsizing. 2) did you run `doodler.py` using `optim=True`? 3) Increase one or both of the CRF parameters in the config file, e.g. change from the default `'compat_col': 120,` and/or `'theta_col': 120,` to something like `'compat_col': 140,` and/or `'theta_col': 140,`

* *Why is my window so small?*
> Try using a large number for `ref_im_scale` in the config file, increasing it from the default value of 0.8 to, say, 1.0 or even 1.2

* *What to I do with a `json.decoder.JSONDecodeError`?*
> This relates to the config file. You either have a comma at the end of a list (bounded by `{...}`) where it shouldn't be, or you have a variable outside of `"..."`


To ask a different question, please use the `Issues` tab (see above) - please do not email me

## Contributing
If you'd like to discuss a new feature, use the `Issues` tab. If you'd like to submit code improvements, please submit a `pull request`



## Version history

### 7/31/2020
1. number of bands is now detected automatically and no longer needs to be specified in the config file. Involved implementing an elaborate function to determine whether or not the 4th band in an image is an alpha mask
2. started website at https://dbuscombe-usgs.github.io/doodle_labeller/
3. remove the `utils` script and folder (creating a 4 band image) and replaced it with a blog post on the new website


### 7/29/20
1. removed `doodler_optim.py` (optim can now be specified as a config file parameter, `optim=True`)
2. good general parameters now hard-coded: `compat_col` = 120, `theta_col` = 100
3. fixed plotting errors so 1-class labels show in mres file, and zeros in final label image dealt with so they don't skew color scale
4. smaller `search` range
5. simplified `merge.py` and made it more consistent with `doodler.py`, using same function calls. Now it doesn't do an additional CRF inference step, just the merging of existing labels
6. tidied example images and labels
7. changed algo that converts probabilities per class to final label image. A class is assigned if its probability is 2(1/N) where N is the number of classes, or 1/N if N=2
8. if `-c` flag is missing, it prompts you to pick a config file manually/graphically
9. no longer makes a `_probs_per_class.npy` file (not necessary going forward)

### 7/24/20

1. the per class predictions are now median-filtered before used in MAP estimation
2. defaults should be `theta_col` = `compat_col` = 120 (?)
3. updated README with explanation about the implementation of the CRF (see `CRF implementation details`	above)
4. probabilities per class now float16 to save disk space
5. the mres file now shows your annotations as well as the label image, side by side. Numerous uses...
6. got rid of some annoying warning message so the screen printout is easier to read


### 7/5/20

1. moved all common functions to `funcs.py`, and consolidated several functions
2. implemented a better and quicker way to final label( now is the weighted average of each class). Works much better in general and particularly for small isolated regions. The weights are the relative frequencies of the classes in the annotations
4. fixed the probabilities per class bugs (many)
5. disabled garbage collector to help troubleshoot program slow-down over successive classes
6. smaller probability per class npy file
7. now automatically removes the `npz` files at the end

### 7/5/20
1. Implemented a better way to estimate the final label from the stack or probabilities associated with each class


### 7/1/20
1. The probability of the label is now computed and outputted in the same formats as the label images. This probability will be useful for purposes such as further label refinement, error propagation, etc
2. Bug fixes, and more tests into default CRF hyperparameters


### 6/24/20

1. now you can pass annotation files (`.npy`) to `doodler.py` if you want to 'redo' the CRF inference.
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
