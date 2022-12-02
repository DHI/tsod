import streamlit as st
from tsod.active_learning import MEDIA_PATH


def general():
    st.markdown(
        """
Welcome to the Time Series Outlier Detection web app!  
This project is a human-in-the-loop system for detecting outliers in time series, 
allowing for the cleaning of noisy datasets, which is often a requirement before the dataset can be used further (e.g. for training ML-models).

### General notes
- This project implements a quite simple workflow. If you are unsure about what the use of a widget / field is, either hover your mouse above the little question mark next to it, or check the documentation. 
- This app is currently a simple web app without user identification or saving of intermediate results. That means that if you refresh the page, you will start from scratch again. However, the app does allow for the download and upload of annotations, models and datasets so you may use them again at a later time.
- It is possible to upload multiple datasets containing multiple series (columns) each. You can add annotations and train models on every individual series. Is is currently not possible to train a single model using features from multiple series (multivariate outlier prediction). However, this is one of the possible future improvements.
### Recommended workflow
    """
    )

    st.image(str(MEDIA_PATH / "workflow.png"), use_column_width=True)

    st.markdown(
        """
    There are several ways of training your outlier detection models. Which one of those works best depends very much on your use case, however here are a few general guidelines. For more details on each step, please find the designated page instructions in the other tabs.

1. Upload your data (under *Data Upload* in the sidebar in the *Outlier Annotation*-page). There are a number of formats supported, see the instructions on Outlier Annotation.
2. Add some annotations for one of your series. If you have previously annotated and saved that series, remember to upload your annotations file from disk (under *Save / load previous* in the sidebar in the *Outlier Annotation*-page). No need to add too many annotations in the first iteration, better to train a model quickly to gain insights into what the model has learned.
3. Head to the page *Model Training*, choose a modelling method (currently only Random Forest Classifier is implemented) and choose some parameters (most of the modelling choices are abstracted away on purpose). Click on *Train Outlier Model* on the bottom of the sidebar to train an initial model.
4. After a short amount of time you will see a brief training summary including train set metrics, (if defined: test set metrics) and feature importances. The next step is to head over to the *Model Prediction*-page to judge the quality of the model. 
5. By default, model predictions for the entire training series are generated when training a model. On the prediction page, you can use any model to generate predictions on any of your datasets/series. Once you have some predictions, you will see them visualized in the main window. As there can be many predicted outliers (especially for earlier models), the predictions are summarized in the outlier distribution bar plot. Each bar represents a time window containing an equal number of datapoints and the height of the bar shows you how many outliers each model predicts in that window. Click on a bar of interest.
6. You can now see the predicted outliers in a new plot underneath. Try to identify patterns of faulty prediction and generate some new annotations by correcting them directly in the graph. You can also add individual annotations, just like in step 2).
7. Alternatively, you can also generate further annotations by heading the the *Annotation Suggestion*-page. There you are prompted to give simple yes or no answer for selected points (based on model uncertainty).
8. After having added some further annotations, you can train another model iteration. For this, either head back to the *Model Training*-page. If you don't want to change any model parameters, you can also click on 'Retain most recent model with new data' (available on both the *Model Prediction* and the *Annotation suggestion*-page). This will train a new model using the same parameters as before, generate new predictions and bring you to the prediction page for comparison.
9. Repeat the circle of adding annotations, retraining and evaluating the results until you are satisfied.
10. To remove outliers from any of your datasets/series, head to the *Data Download*-page. There you can create new datasets by removing predicted outliers, as well as download any dataset you have uploaded/created.
    """
    )


def outlier_annotation():

    st.markdown(
        """
    The *Outlier Annotation*-page is designated to the manual adding of annotations to any series. As the "entrypoint" of the app, it also holds the functionality to upload datasets.  
The main window will always only contain an interactive plot window. In the sidebar, you'll find all widgets related to interacting with the annotation process. 
"""
    )
    st.markdown("***")
    c1, c2 = st.columns([3, 1])

    c1.markdown(
        """
### Uploading Data

The first field in the sidebar allows you to upload your datasets. For trying out the app, you can also click on 'Add generated data' to add a toy dataset with two random series.  

Currently, the following file formats are supported for uploading your data from disk: 
- CSV
- XLSX / XLS
- DFS0

If your dataset is split into multiple files, you select all files and they will be merged into a single dataset. However, the data needs to be consistent (may not contain multiple values for the same timestamp for the same series). A variety of different timestamp formats are supported.  
Optionally, you can give your dataset a name for easy identification, otherwise it will receive a handle based on the names of the uploaded files.  
To finish, click on 'Upload'. Once your files have been validated and merged, you will be able to select your dataset under *Data Selection* in the sidebar.

"""
    )
    c2.image(
        str(MEDIA_PATH / "data_upload.png"),
        use_column_width=True,
        # width=400,
        caption="The 'Data Upload' field.",
    )
    st.markdown("***")
    c1, c2 = st.columns([3, 1])
    c1.markdown(
        """
### Main plot window

The main plot window will display the selected series for the selected time interval. By default, the plot will contain a series displaying only your actual datapoints, as well as a series containing the connecting line. This is purely for convenience reasons, individual points are always selected by clicking on the datapoints.  
As soon as you selected at least one point, you will see your selection marked in purple (also a new entry will be added in the legend).  
Once you add annotations, they will be marked as a new series as well.  
To select multiple neighboring points at once, it is easier to use the 'Horizontally Select' or 'Box Select' - options by activating them in the top right corner of the plot window. 
    """
    )
    c2.image(
        str(MEDIA_PATH / "selection_options.png"),
        use_column_width=True,
        caption="You can change data selection modes in the top right corner of the main plot window.",
    )
    st.markdown("***")
    c1, c2 = st.columns([3, 1])

    c1.markdown(
        """
### Annotation Controls

**Actions**

This fields allows you to choose what to do with your selection. Your selection has two label options (Outlier or Normal) and can be assigned to either the train or the test set.  
'Clear Selection' removes your entire selection, but keeps all points annotated thus far. 'Clear All' resets your annotation state, removing selected as well as annotated points. 

**Time Range Selection**

This field offers control over the time range that is displayed in the main plot window.  
In order to assure that the app works with datasets on any time scale (nanoseconds to decades), when first loading in a new dataset, a time range will automatically be chosen so that the main plot contains the last 200 points of data.  
If your selected series spans a time range of more than a day, a calender widget will be available to select start & end date, as well as two time widgets for setting start & end time.  
You can also directly set the number of datapoints the plot should contain, using the number input on the bottom of the field.  
'Show All' will display the entire series (not recommended for large number of points).  
The 'Shift back' and 'Shift forward' - buttons are useful for stepping through your data in equal time steps. As any initial visualization starts at the end of the timestamp index, clicking on 'Shift back' will determine the current range that is being displayed and then update the plot backwards in time, keeping the range equal (the previous start timestamp will become the new end timestamp).  
**Recommended workflow for stepping through your dataset:**  
Select an appropriate time range that makes outliers easily visible for you => step through the dataset using the Shift buttons and add annotations.

**Save / load previous**

To continue annotations in a different session, you might want to download your current progress. Click the 'Download Annotations' - button to save your annotations for the current dataset to disk as binary data.  
Use the 'Upload Annotations' - uploader to add previously created annotations. This assumes that you already have the correct dataset loaded, the actual data is not saved together with the annotations. 
    """
    )
    c2.image(
        str(MEDIA_PATH / "time_range_selection.png"),
        use_column_width=True,
        caption="Select a time range that fits your data and lets you easily identify outliers. Then step through your dataset while keeping that range.",
    )


def model_training():
    st.markdown(
        """
    The *Model Training*-page is designated to choosing modelling methods & hyperparameters, training models and evaluating their performances. In the sidebar, you'll find all widgets related to the training process.  
    On top of the main window, you will see a short annotation summary. If you have already trained a model, you will also be able to see how many annotation points where added since the last model was trained.  
After training a new model, train metrics will show underneath (& test metrics if defined). Precision, Recall & F1 scores are shown separately for annotated outliers and normal points. If you have already trained a model, you will also be able to see how the metrics have changed compared to the previous model.
Underneath that, you'll find a plot & table showing some of the feature importances of the last trained model. If you have already trained a model, you will also be able to see how the importances have changed compared to the previous model. 
"""
    )
    st.markdown("***")
    st.markdown(
        """
    ### Training Controls

**Data Selection:**  
If you have annotated multiple datasets or one dataset containing multiple series, you can choose your training series here. By default, the last series you have added annotations for is selected.  

**Modelling Options:**  
Choose which modelling approach to use to predict outliers (more below).

**Feature Options:**  
Control some basic, model-specific feature parameters (more below):

To start training, click on 'Train Outlier Model' on the bottom of the sidebar.  
The checkbox 'Auto generate predictions for entire annotation series' (selected by default) simply means that after training, the newly trained model will be used to create predictions for the entire selected series. This is just a convenience option, as this would be the most common next step in the suggested workflow anyway. 

### Modelling Options

Right now, only one supervised learning method is implemented for outlier prediction. More are to follow in the future.  
For each annotated point (outlier or normal), a set of features is generated that is fed to the model to predict whether the point is an outlier or not.


**Random Forest Classifier**

[scikit-learn docs - RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
[scikit-learn docs - RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

The feature set is constructed by taking the x points before the annotated point, the y points after and normalizing them using the value of the annotated point. This way the model can only look at the relative changes of a series leading up to or after the point in question.  
If you are interested in building any kind of 'real time' outlier detection model, set the number of points after to 0 (default).  
The model using the feature set is a simple scikit-learn Random Forest Classifier. Behind the scenes, the random seeds are fixed for reproducible results and a small hyperparameter search is performed for cross validation.  
Training times depend on the number of annotated points, but should generally never be longer than a minute. 
    """
    )


def model_prediction():
    st.markdown(
        """
    The *Model Prediction*-page is designated to creating & visualizing the predictions of any trained or uploaded model on any series. In the sidebar, you'll find all widgets related the generation of predictions.  
    The main window serves the visualizations for each selected series, while the sidebar is used to generate new predictions.
"""
    )
    st.markdown("***")
    st.markdown(
        """
## Prediction Controls (sidebar)
    
It is recommended to generate predictions whenever training a new model (default). However, it is also possible to upload previously trained models and generate predictions for any uploaded series.  
This allows for several potentially interesting workflows, such as uploading a test set as a separate dataset or evaluating the capabilities of a model on a completely different dataset. If you have previously created a capable model for your use case, you can skip the annotation and training processes and jump straight to the model prediction page, upload your model and clean your datasets.  
Use the controls in the sidebar to select one or multiple models trained this session and/or upload models from disk. Then add individual series from any of your uploaded datasets for which you would like these models to generate predictions. 
***
    """
    )

    st.markdown(
        """
        ## Prediction visualization (main window)

Each series you have generated predictions for will appear underneath each other in main window, each with its own set of visualization options and graphs.
    """
    )
    st.image(
        str(MEDIA_PATH / "distribution_to_prediction_plot.png"),
        use_column_width=True,
        caption="The distribution plot (left) summarizes the predicted outlier distribution. Clicking on any of the bar groups will bring up the outlier plot (right) for that time range underneath.",
    )

    c1, c2 = st.columns([2, 1])

    c1.markdown(
        """
### Visualization Options


1. In the multiselect at the top you will be able to choose up to 4 models that have been used to generate predictions for this series. By default (when a new set of predictions is added), the two most current models will be selected here, as this comparison is usually the most relevant (comparing the predictions of the most recent model to those of the previous one).  
    Underneath, you will then see a small summary table, showing for each selected model:  
    - The model name
    - A 'Remove'-button to quickly remove that model from the selection
    - The parameters of the model
    - The number of predicted outliers for the series
    - The number of predicted normal points for the series
    - A color picker to choose which color that model should be shown in in the following graphs

2. Each bar in the following distribution plot represents the number of predicted outliers for a given model. In order to not display too many points at once, the series is split into multiple parts containing an equal number of datapoints. That number can be adjusted using this slider.

3. Set the height of both the distribution plot and the outlier plot (in pixel).

4. If there are many segments of the distribution plot containing no predicted outliers, activate this checkbox to only display bars where there are predicted outliers. 

5. If any of these checkboxes are set, annotated train or test outliers that where not predicted to be outliers by the model will be highlighted by a blinking symbol in the distribution plot. This is useful for identifying at a glance where predictions are deviating from annotations. 

    """
    )

    c2.image(
        str(MEDIA_PATH / "model_visualization.png"),
        use_column_width=True,
        caption="Customization options for visualizing model predictions.",
    )
    c1, c2 = st.columns([2, 1])
    c1.markdown(
        """
### Prediction Correction

You can use the outlier plots themselves to add further annotations, based on your model predictions. For this, you have several options to select points:

- Click on predictions markers or normal datapoints for individual selection
- Utilize the Box- or Horizontal selection modes to select multiple prediction markers at once (default). This way when drawing a box, you will only select points where at least one model made a prediction. This is useful to quickly correct multiple faulty predictions
- By unchecking the checkbox 'Area select: Only select predicted outliers' above the outlier plot, you can also select any datapoints within the range of a Box- or Horizontal select. 

Underneath the outlier plot, you will find another set of annotation buttons which you can use to label your selected points.
    """
    )

    c1, c2 = st.columns([2, 1])
    c1.markdown(
        """
    ### Retraining

For convenience, the 'Retrain most recent model with new data' - button exists in order to retrain the most recent model (using the same parameters) and generate new predictions using that model. This allows for quicker iterations.  

    """
    )
    c2.image(
        str(MEDIA_PATH / "retrain.png"),
        use_column_width=True,
        caption="The retrain button allows for quicker iterations, without having to switch pages.",
    )


def annotaion_suggestion():
    st.markdown(
        """
     The *Annotation Suggestion*-page is designated to presenting the user with interesting points to annotate, using a simple yes-no-dialogue.  
 For Random Forest Classifiers, the order of presented points is determined by the degree of disagreement of the individual decision trees (points with the most disagreement first).  
 In the sidebar, you can choose from any model trained in the current session. Points are always drawn from the series the model was trained on.  
 You can also set the number of neighboring points to display in the plot.
    """
    )

    st.image(
        str(MEDIA_PATH / "annotation_suggestion.png"),
        use_column_width=True,
        caption="In the annotation suggestion page, you are prompted for annotations of specific points.",
    )


def data_download():
    st.markdown(
        """
The *Data Download*-page is designated to the removal of outliers from your datasets and the download of your cleaned data.  
In the sidebar, choose a dataset. If the dataset has multiple series, you can then choose which of its series to clean. The final dataset will always contain all its original series, regardless of which series you choose.  
Next pick any of your trained or uploaded models to use for the cleaning.  
Currently, there are two methods available for handling the predicted outliers: Either deleting them from the series completely or performing linear interpolation.  
Click on 'Preview' to generate and review 3 random samples that show you effect of removing the outliers in the chosen way.  
After reviewing, you can then add your cleaned data as a new dataset, either to download right away or to clean further using another model. You could also start an new annotation process for the newly created dataset.  
Finally, at the bottom of the sidebar, you'll have the option to download any of your session dataset to disk, either as a .csv or a .xlsx file. 
        """
    )
    st.image(
        str(MEDIA_PATH / "data_download.png"),
        use_column_width=True,
        caption="Before saving your cleaned data, you may preview your chosen method of removing outliers.",
    )


INSTRUCTION_DICT = {
    "General": general,
    "Outlier Annotation": outlier_annotation,
    "Model Training": model_training,
    "Model Prediction": model_prediction,
    "Annotation Suggestion": annotaion_suggestion,
    "Data Download": data_download,
}
