# train_classifier.py

import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Load data from the SQLite database.

    Args:
    database_filepath: str. Filepath to SQLite database.

    Returns:
    X: dataframe. Features dataset.
    Y: dataframe. Target dataset.
    category_names: list of str. List of target category names.
    """
    engine = create_engine(f'sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]  # Assuming target columns start from the 5th column
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and clean text.

    Args:
    text: str. Input text.

    Returns:
    tokens: list of str. List of tokens after processing.
    """
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return tokens

def build_model():
    """
    Build machine learning pipeline with a reduced search space for faster results.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Reducing the search space for faster testing
    parameters = {
        'clf__estimator__n_estimators': [50],  # Reduce n_estimators
        'clf__estimator__min_samples_split': [2]  # Keep one value for min_samples_split
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3, n_jobs=-1)  # Reduced to 2 folds
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print classification report.

    Args:
    model: model object. Trained model.
    X_test: dataframe. Test features.
    Y_test: dataframe. Test targets.
    category_names: list of str. List of target category names.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Category: {col}\n', classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the model to a pickle file.

    Args:
    model: model object. Trained model.
    model_filepath: str. Filepath to save the pickle file.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    """
    Main function to load data, build model, train and evaluate, and save the model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()