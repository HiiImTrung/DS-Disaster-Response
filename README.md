## Disaster Response Pipeline Project

## Project Summary
This project is a part of the Udacity Data Scientist Nanodegree program. Its objective is to construct a Natural Language Processing (NLP) model capable of categorizing messages made during disaster situations. The model is trained on the set of data comprising authentic messages transmitted during past disaster occurrences. Subsequently, the model is deployed to classify upcoming messages during the unexpected events.

## File Structure
data/: Contains data files and scripts for data processing.
-    disaster_messages.csv: CSV file containing messages dispatched during disasters.
-    disaster_categories.csv: CSV file containing message categories.
-    process_data.py: Python script for cleaning data and storing in a database.

models/: Contains scripts and files for training and saving the classification model.
-    train_classifier.py: Python script for training the classifier and saving the model.
-    classifier.pkl: Saved classifier model.

app/: Contains files for the web application.
-    run.py: Python script to run the web app.
-    templates/: HTML templates for the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseTT.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponseTT.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`


Open a web browser and go to http://0.0.0.0:3000/ to view the web app.
