---
id: install
title: Getting Started
sidebar_label: Installation
---

**Warning**: `Doodler` is still in active development and `beta` version. Check back later or listen to announcements on [twitter](https://twitter.com/magic_walnut) for the first official release.

`Doodler` is a [python](https://www.python.org/) program that is designed to run within a [conda](https://docs.conda.io/en/latest/) environment, accessed from the command line (terminal). It is designed to work on all modern Windows, Mac OS X, and Linux distributions, with python 3.6 or greater. It therefore requires some familiarity with process of navigating to a directory, and running a python script from a command line interface such as a [Anaconda prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/) terminal window, [Powershell](https://docs.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7) terminal window, [git bash](https://gitforwindows.org/) shell, or other terminal/command line interface.

The following instructions assume you are using the popular (and free) [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) python distribution.

### Clone the github repo

```
git clone --depth 1 https://github.com/dbuscombe-usgs/doodle_labeller.git
```

### Create a conda environment

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

### Activate the conda environment

Use this command to activate the environment, in order to use it

```
conda activate doodler
```

If you are a Windows user (only) who wishes to use unix style commands, install `m2-base`

```
conda install m2-base
```
