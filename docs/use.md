---
id: use
title: How to Use
sidebar_label: How to Use
---


### Add images to `data/images`
The program will assume you want to label all these images in one go. It's usually best to put 1-10 images in there at a time, depending on how much time you have in your doodler session

### Make a `config.json` file
Several example config files are provided in the `config/` folder. A generic multi-label case would be similar to below:

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
	  "num_bands": 3,
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

where the *required* arguments are:

* `image_folder` : ordinarily this would be "data/images" but could be a relative path to any relative directory
* `label_folder`: see the above but for "data/label_images"
* `max_x_steps`: number of tiles to split the image up in x direction (suggested range: 1- 6, depending on image size/complexity)
* `max_y_steps`: number of tiles to split the image up in y direction (suggested range: 1- 6, depending on image size/complexity)
* `ref_im_scale`: the proportion of the detected screen size to make the window (suggested: 0.5 -- 1.0)
* `lw`: initial pen width in pixels. this can be changed on the fly (suggested range: 5 - 10, depending on image size)
* `im_order`: either "RGB" or "BGR", depending on what type of imagery you have (RGB is more common)
* `classes`: a dictionary of class names and associated colours as hex codes (starting with `#`). There are various online colour pickers including [this one](https://htmlcolorcodes.com/)


and the optional arguments and their default values:

* `apply_mask`: either `None` (if no pre-masking) or a list of label names with which to mask the image. These label images should already exist *Default = None*
* `num_bands`: the number of bands in the input imagery (e.g. 1, 3, or 4). *Default = 3*
* `create_gtiff`: if "true", the program will create a geotiff label image - only appropriate if the input image is also geotiff. Otherwise, "false" *Default = false*
* `alpha`: the degree of transparency in merged image-output plots *Default = 0.5*
* `fact`: the factor by which to downsample by imagery. *Default = 5* (this might seem large, but trust me the CRF is very cpu and memory intensive otherwise, and the results work with a large `fact` turn out ok. Reduce to get finer resolution results by be warned, it will take a lot longer. )
* `compat_col`: compatibility value for the color/space-dependent term in the model, initial value for search. *Default = 120*
* `theta_col`: standard deviation value for the color/space-dependent term in the model, initial value for search. *Default = 100*
* `optim`: if `True`, will search through more hyperparameters and will take longer. For problem imagery

This file can be saved anywhere and called anything, but it must have the `.json` format extension.

### Run `doodler.py`
Assuming you have already activated the conda environment (`conda activate doodler` - see above) ...

you will need to cd to the directory you will be working in, for example:

```
cd /my_python_programs/doodle_labeller
```

Then run it with:

```
python doodler.py -c config_file.json
```

or optionally (to manually navigate to and select a config file)

```
python doodler.py
```

or optionally (to pass a previous annotation filr in `.npy` format)

```
python doodler.py -c path/to/config_file.json -f path/to/npy_file
```

Command line arguments:

* `-h`: print a help message to screen, then exit
* `-c`: for specification of a `config` file. Optional - if this is missing, the program will prompt you to select a config file
* `-f`: pass a `.npy` file to the program (made using `doodler.py` during a previous session). Optional
