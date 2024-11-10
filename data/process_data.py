import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    
    Args:
    messages_filepath: str. Filepath for the messages CSV file.
    categories_filepath: str. Filepath for the categories CSV file.
    
    Returns:
    df: dataframe. Merged dataframe of messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on the 'id' column
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Clean the data by splitting categories, converting to binary, and removing duplicates.
    
    Args:
    df: dataframe. Dataframe containing merged messages and categories data.
    
    Returns:
    df: dataframe. Cleaned data.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Convert category values to integers and handle "related" column
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    
    # Drop rows where the "related" column has a value of 2
    categories = categories[categories['related'] != 2]
    
    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save the clean data to an SQLite database.
    
    Args:
    df: dataframe. Cleaned data.
    database_filename: str. Filepath for the SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    """
    Run the ETL pipeline: Load, clean, and save data.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()