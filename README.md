# "Doodle Labeller"

> Daniel Buscombe, Marda Science daniel@mardascience.com

> Developed for the USGS Coastal Marine Geology program, as part of the Florence Supplemental project

> This is a "Human-In-The-Loop" machine learning tool for partially supervised image segmentation and is based on code previously contained in the "dl tools" [repository](https://github.com/dbuscombe-usgs/dl_tools)

> Significant code contribution from LCDR Brodie Wells, Naval Postgraduate school Monterey

> Sample images provided by 1) Christine Kranenburg, USGS St. Petersburg Coastal and Marine Science Center, 2) Chris Sherwood, USGS Woods Hole Coastal and Marine Science Center, and 3) Jenna Brown, USGS MD-DE-DC Water Science Center

> The Conditional Random Field (CRF) model used by this tool is described by [Buscombe and Ritchie (2018)](https://www.mdpi.com/2076-3263/8/7/244)

Please go to the [project website](https://dbuscombe-usgs.github.io/doodle_labeller/) for more details and documentation

## Rationale
There are many great tools for exhaustive (i.e. whole image) image labeling for segmentation tasks, using polygons. Examples include [makesense.ai](www.makesense.ai) and [cvat](https://cvat.org). However, for high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. This is especially true of scenes with many classes of interest, and covering relatively small, spatially discontinuous regions of the image.

What is generally required in the above case is a semi-supervised tool for efficient image labeling, based on sparse examples provided by a human annotator. Those sparse annotations are used by a secondary automated process to estimate the class of every pixel in the image. The number of pixels annotated by the human annotator is typically a small fraction of the total pixels in the image.  

`Doodler` is a tool for "exemplative", not exhaustive, labeling. The approach taken here is to freehand label only some of the scene, then use a model to complete the scene. Sparse annotations are provided to a Conditional Random Field (CRF) model, that develops a scene-specific model for each class and creates a dense (i.e. per pixel) label image based on the information you provide it. This approach can reduce the time required for detailed labeling of large and complex scenes by an order of magnitude or more.

This tool is also set up to tackle image labeling in stages, using minimal annotations. For example, by labeling individual classes then using the resulting binary label images as masks for the imagery to be labeled for subsequent classes. Labeling is achieved using the `doodler.py` script

Label images that are outputted by `doodler.py` can be merged using `merge.py`, which uses a CRF approach again to refine labels based on a windowing approach with 50% overlap. This refines labels further based on the underlying image.

This is python software that is designed to be used from within a `conda` environment. After setting up that environment, the user places imagery in a folder and creates a `config` file that tells the program where the imagery is and what classes will be labeled. The minimum number of classes is 2. There is no limit to the maximum number of classes.

## How to use

1. Put a single image or multiple images in the "data/images" folder

2. (optionally) Clear the contents of the "data/label_images" folder

3. make a `config` file (see below)

4. make 'sparse' annotations on the imagery with a mouse or stylus, and use keyboard keys to advance through classes


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


<!--
## compiling doodler.py
- conda activate doodler
- pip install --upgrade 'setuptools<45.0.0'
- pip install python-opencv-headless
- pip install pyinstaller
- pyinstaller --onefile --noconfirm doodler.py --clean --hidden-import pydensecrf.eigen --hidden-import rasterio._shim --hidden-import rasterio.control --hidden-import pkg_resources.py2_warn
- conda deactivate
- ./dist/doodler -->
