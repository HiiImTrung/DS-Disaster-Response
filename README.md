## Disaster Response Pipeline Project

## Project Summary
This project is a part of the Udacity Data Scientist Nanodegree program. Its objective is to construct a Natural Language Processing (NLP) model capable of categorizing messages dispatched during disaster situations. The model is trained on a dataset comprising authentic messages transmitted during past disaster occurrences. Subsequently, the model is deployed to classify incoming messages during similar events.

## File Structure
- data/: Contains data files and scripts for data processing.
-    disaster_messages.csv: CSV file containing messages dispatched during disasters.
-    disaster_categories.csv: CSV file containing message categories.
-    process_data.py: Python script for cleaning data and storing in a database.

- models/: Contains scripts and files for training and saving the classification model.
-    train_classifier.py: Python script for training the classifier and saving the model.
-    classifier.pkl: Saved classifier model.

- app/: Contains files for the web application.
-    run.py: Python script to run the web app.
-    templates/: HTML templates for the web app.

## Running Python Scripts and Web App
To run the ETL pipeline that cleans data and stores it in a database:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run the ML pipeline that trains the classifier and saves the model:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
To run the web app:

cd app
python run.py

Open a web browser and go to http://127.0 to view the web app.
