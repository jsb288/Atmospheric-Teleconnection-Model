# Atmospheric Teleconnection Model


## Getting Started (Installation):

1) Install the project from Github  
Click the green "<>Code" button, click "Download ZIP", find the project in your downloads, and unzip the folder (extract all). You can move the entire Atmosperic-Teleconnection-Model folder, but moving individual files within that folder may cause problems if all scripts' folder path variables are not changed accordingly.

2) Make sure you have the correct environment  
2A. The Environments folder includes yml files you can use to create a python environment for the project. For more on creating a conda environment from a yml file, refer to the [Conda User Guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). Use the agcm_environment_windows.yml file for environments on Windows machines or the agcm_environment_mac.yml file for MAC machines. agcm_environment.yml is a generic version which can be used for either.  
2B. Some users have reported difficulties using the yml environment files. Using the following commands in a terminal is an alternative strategy:
   - conda install -n agcm_environment xarray netcdf4 scipy matplotlib jupyter pytorch;
   - conda install -n agcm_environment -c conda-forge xesmf;
   - conda activate agcm_environment;
   - pip3 install torch-harmonics==0.6.3

3) Edit Preprocess.ipynb variables  
Edit Preprocess.ipynb to choose your resolution, number of months to run, and make sure your folder paths are correctly set - that is, where to write the output. Documentation in the notebooks should help in doing this.

4) Run Preprocess.ipynb as a jupyter notebook  
You can easily modify the topography, background state or the heating in this script, but we suggest running without modification first. You will need to download observed gridded topography data from the TopogData folder (topog.gridded.nc).

5) Edit RunModel.beta.ipynb (or RunModel.PrescribedMean.ipynb) variables  
Choose which model you are using and edit its variables according to your requirements. Variables included in both postprocess and the model must match.

6) Run RunModel.beta.ipynb (or RunModel.PrescribedMean.ipynb)  
The RunModel.beta.ipynb file is the weakly prescribed mean version and the RunModel.PrescribedMean.ipynb file is the strongly prescribed mean version of the model. See manuscript for detailed discussion of the two versions of the model.

7) Edit and run a Postprocess script  
Before running the postprocess file, edit the variables to match your data from the preprocess and model files. There are two post-processing scripts in the Postprocess folder for vertical interpolation for sigma to pressure coordinates. The preferred post-processing uses metpy as indicated in the filename. The raw model output is in the native sigma coordinate in the vertical and is on the Gaussian grid for the horizontal.

8) Troubleshooting  
For any issues, questions or concerns please contact Ben Kirtman at bkirtman@miami.edu.

9) To verify that you have installed the preprocessor and run the model correctly see the readme file in the benchmark folder.

10) The Held and Suarez (1994) dynamical core test is in the HeldSuarez folder. See Readme there in.


## Variable Glossary
Towards the top of each preprocess, model, and postprocess file you can set the values of the variables relevant to the model. The details of each variable are included below.


### Standard Variables
In most cases it is only necessary to set values for the standard variables.

**zw** is the zonal wave number. For standard use, zw should be set to the value of either 42, 63, or 124. Setting the value for zw also sets default values for the following variables: mw, jmax, imax, and steps_per_day.
<br>Set zw = 42, to set mw = 42, jmax = 64, imax = 128, and steps_per_day = 216.
<br>Set zw = 63, to set mw = 63, jmax = 96, imax = 192, and steps_per_day = 324.
<br>Set zw = 124, to set mw = 124, jmax = 188, imax = 376, and steps_per_day = 648.
<br>Each of these variables (mw, jmax, imax, and steps_per_day) that is given a value in the advanced variables section will instead use that value.

**kmax** is the number of vertical levels. The value of kmax should be 11 or 26 for standard use.

**expname** is the name you want to be given to your experiment. When the data is saved to your computer, it will be saved in a folder with this name. Note that if you run this program twice with the same expname, your first experiment's data will be overwritten.

**toffset** is the number of days that have already run when restarting.

**datapath_init** is the directory in which files are saved in the case of a restart. Set this equal to the datapath if restarting in the same directory.

**DataSetname** is the name of the data set. This will be used to name files that are created by this program.

**Dataname** is the name of the data. This is what will be used to label the data in the charts generated by this program.

**dayst** is the number of days over which the data spans. This number will be used to label the data as it's output to files and will also be used to run this program and display the data.


### Advanced Variables
While most cases only require setting the standard variables, some cases might require setting some or all advanced variables as well. The following variables should only be changed from their default value if a specific behavior is desired. An advanced variable set to the value of None will use the default case.

**mw** is the meridional wave number. In the standard case this value is set equal to zw.

**jmax** is the number of Gaussian latitudes. jmax = imax/2

**imax** is the number of longitude grid points. imax >= 3 * zw + 1. imax must be an even number.

**steps_per_day** is the number of time steps per day. It gives you the delta t in the time differencing scheme. The length of a day is 86400 seconds, so delta t = 86400/steps_per_day. Changing this number implies time step changes and should be implemented carefully. The values used in the standard case were determined expertimentally.

**custom_path** is the full path of the folder in which you wish to save your data when running the model, or the full path of the folder you wish to retrieve your data from when running the postprocess. If custom_path is set, expname is ignored. Note that this must be an existing folder. If you use a custom_path when running the model, you must use the same custom_path in postprocess to access the same data.

**custom_kmax** is used to safeguard against using unexpected values for the kmax. If custom_kmax is set, it will be used instead of kmax. By default the program only supports kmax with a value of either 11 or 26. Other values are implementable, but the user must modify subs1_utils.py routine bscst. If unclear email bkirtman@miami.edu for clarification.


## Project Structure
The project folders are structured as follows:


### Benchmarks
This folder contains html files from the preprocessor and the RunModel.beta that can be used to determine that the model has been installed and run correctly.
Download the html files that correspond to the resolution you are trying to run.
To verify the preprocessor compare the figure that the notebook produces with the figures in the corresponding html file in this directory.
To verify that the model is running correctly compare the second to last cell that provides some tensor values on days 1-6. These should be very similar to your first six days of your simulation.


### Environments
A folder containing the yml files for creating python environments.  
The linux, mac, and windows files can be used to create an environment on their corresponding operating system.
The agcm_environment.yml file is a generic yml file for creating a python environment on any system.


### HeldSuarez
The Held and Suarez (1994) dynamical core test of our model
Held, I. M., & Suarez, M. J. (1994). A Proposal for the Intercomparison of the Dynamical Cores of Atmospheric General Circulation Models. https://journals.ametsoc.org/view/journals/bams/75/10/1520-0477_1994_075_1825_apftio_2_0_co_2.xml


### MultiThreadModel
A folder containing the main model of this project.  
See Getting Started for instructions on running the preprocess and model.


### Postprocess
A folder containing the postprocess to be run after the model.  
See Getting Started for instructions on running the postprocess.


### TopoData
Folder contains topography data at 0.5 degrees that can be interpolated to desired resolution.
