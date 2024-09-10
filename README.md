Brief Instuctions on how to run the model:

1) Make sure you have the correct environment (see environments folder)

2) Download subs1_utlis.py, RunModel.Beta.ipynb (or RunModel.PrescribedMean.ipynb),Preprocess.ipynb from the MultiThread_Model folder. The RunModel.Beta.ipybb is the weakly prescribed mean version and the RunModel.PrescribedMean.ipynb is the strongly prescribed mean version of the model. See manuscript for detailed discussion of the two versions of the model.

3) Edit RunModel.Beta.ipynb & Preprocess.ipynb to choose your resolution, number of months to run, and make sure you folder paths are correctly set - that is, where to write the output. Documentation in the notebooks should help in doing this.

4) Run Preprocess.ipynb as a jupyter notebook. You can easily modify the topography, background state or the heating in this script, but we suggest running without modification first.

5) Run RunModel.Beta.ipynb (or RunModel.PrescribedMean.ipynb)

6) There are two post processing scripts in the postprocessing fold for vertical interpolation for sigma to pressure coordinates. The perferred post-posssing uses the metpy indicated in the filename. The raw model output is in the native sigma coordinate in the vertical and is on the Gaussian grid for the horizontal. 
