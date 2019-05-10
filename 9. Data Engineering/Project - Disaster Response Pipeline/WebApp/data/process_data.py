import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories data and returns a merged dataframe.
    Args:
        messages_filepath(string) : path to messages.csv.
        categories_filepath(string) : path to categories.csv.
    Returns:
        pandas.DataFrame : Merged messages and categories dataframe.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Cleans and returns the input pandas.dataFrame.
    """
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    row = categories.loc[0,:]
    categories.columns = row.apply(lambda x: x.split('-')[0])
    for column in categories:
        categories[column] = pd.to_numeric(categories[column].str[-1:])
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(subset='id', inplace=True)
    df['related'].replace(2, 1, inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves the input dataframe in a sqlite database of the specified name with the table name 'Messages'.
    Args:
        df(pandas.DataFrame) : dataframe to be saved.
        database_filename(string) : database filename.
    Returns:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, if_exists="replace", index=False)


def main():
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