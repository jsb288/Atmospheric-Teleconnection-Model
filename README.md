# Atmospheric Teleconnection Model

## Getting Started (Installation):

1) Install the project from Github  
Click the green "<>Code" button, click "Download ZIP", find the project in your downloads, and unzip the folder (extract all). You can move the entire Atmosperic-Teleconnection-Model folder, but moving individual files within that folder may cause problems if all scripts' folder path variables are not changed accordingly.

2) Make sure you have the correct environment  
The Environments folder includes yml files you can use to create a python environment for the project. For more on creating a conda environment from a yml file, refer to the [Conda User Guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). Use the agcm_environment_windows.yml file for environments on Windows machines or the agcm_environment_mac.yml file for MAC machines. agcm_environment.yml is a generic version which can be used for either.

3) Some users have reported difficulties using the yml environment files. The following is an alternative strategy:

   (i) conda install -n agcm_environment xarray netcdf4 scipy matplotlib jupyter pytorch;

   (ii) conda install -n agcm_environment -c conda-forge xesmf;

   (iii) conda active agcm_environment;

   (iv) pip3 install torch-harmonics==0.6.3 

5) Edit Preprocess.ipynb variables  
Edit Preprocess.ipynb to choose your resolution, number of months to run, and make sure your folder paths are correctly set - that is, where to write the output. Documentation in the notebooks should help in doing this.

6) Run Preprocess.ipynb as a jupyter notebook  
You can easily modify the topography, background state or the heating in this script, but we suggest running without modification first.

7) Edit RunModel.beta.ipynb (or RunModel.PrescribedMean.ipynb) variables  
Choose which model you are using and edit its variables according to your requirements. Variables included in both postprocess and the model must match.

8) Run RunModel.beta.ipynb (or RunModel.PrescribedMean.ipynb)  
The RunModel.beta.ipynb file is the weakly prescribed mean version and the RunModel.PrescribedMean.ipynb file is the strongly prescribed mean version of the model. See manuscript for detailed discussion of the two versions of the model.

9) Edit and run a Postprocess script  
Before running the postprocess file, edit the variables to match your data from the preprocess and model files. There are two post-processing scripts in the Postprocess folder for vertical interpolation for sigma to pressure coordinates. The preferred post-processing uses metpy as indicated in the filename. The raw model output is in the native sigma coordinate in the vertical and is on the Gaussian grid for the horizontal.

10) Issues, questions or concerns please contact Ben Kirtman at bkirtman@miami.edu.


