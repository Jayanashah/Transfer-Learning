Transfer Learning in Cochlear Implant (CI) Users
-----------------------------------------------------------------------------------------------------
Overview
-------------

This project investigates the application of transfer learning and convolutional neural networks (CNNs) to enhance auditory attention decoding (AAD) for cochlear implant (CI) users. The research aims to improve speech comprehension in noisy environments by adapting pre-trained models on normal-hearing (NH) datasets to CI users.


Getting started
------------

**Setting up the environment**

I recommend using a virtual conda environment for the repository. [Here's a tutorial](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/)

Under /conda_env you find .yml files for Linux and Linux with which you you can [create an environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

As a second option there is also  requirements.txt file.

If there are issues with creating the environment from the files try to install libraries in the following order:
1. PyTorch
2. mne-base (Make sure not to install full mne - this takes for ever)
3. gitpython
4. h5py
5. pickleshare

**Installing src as module**

Run `python setup.py install` in the root directory


Workflow
------------
**Download hdf5 database file**

The dataset was created with the src/data/create_dataset.py file.
For that you need th raw_input data, which can be made available upon reasonable request.

We provide a dataset as hdf5 file on [zenodo](10.5281/zenodo.10980117).
It contains preprocessed data as well as raw data.
The downloaded file should be put in the data/processed folder.

**Plotting**

All plots were created from jupyter notebooks in the Plots folder.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
