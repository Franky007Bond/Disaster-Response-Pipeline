import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        Loads two csv-files cotaining messages and associated categories
        and merges these together on common id
    
        Args:
            message_filepath       str                  path to message csv-file
            categories_filepath    str                  path to categories csv-file
        
        Returns:
            df                     pandasd.DataFrame    merged dataframe    
    """
    
    # load messages dataset
    try:
        messages = pd.read_csv(messages_filepath)
    except:
        sys.exit("Couldn't open file: {}".format(messages_filepath))
        
    # load categories dataset
    try:
        categories = pd.read_csv(categories_filepath)
    except:
        sys.exit("Couldn't open file: {}".format(categories_filepath))
        
    # merge datasets on id
    df = messages.merge(categories, on='id',how='inner')
    
    return df

    
def clean_data(df):
    """
        Following steps are applied to the dataframe containing 
        disaster messages and categories:
        - drop rows with nan-values for message or categories
        - encode categories string to 36 columns for the individual categories
        - correct categories values to zeros and ones only
        - drop duplicate datasets
     
        Args:
            df       pandas.DataFrame       merged dataframe containing disaster messages and categories
                                            output of load_data function 
     Returns:
            df      pandas.DataFrame        cleaned dataframe
                                            input for ML model 
    """
    
    # drop rows with nan-values for message or categories
    # as these are the arguments that are required for the machine learning model
    # this is done to make the algorithm more robust even if the data provided 
    # does not have any such issue
    df.dropna(subset=['message','categories'])
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[0:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    
    for column in categories:
        # extract actual value for each category
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # correct data values to zeros and ones only
    # provided data contained twos as well that are transformed to ones 
    for col in categories.columns:
        categories[col] = categories[col].apply(lambda x: 1 if (x==2) else x)
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
        Stores the dataframe to an SQL database
    
        Args:
            df                  pandas.DataFrame    dataframe to be saved
                                                    in our use case: cleaned dataframe /
                                                        output of clean_data-function
            database_filename   str                 filepath where to store datanase file                               
          
        Returns:
            None
    """
    
    try:
        engine = create_engine('sqlite:///{}'.format(database_filename))                     
        df.to_sql('DisasterResponses', engine, index=False, if_exists='replace')
    except:
        sys.exit("Couldn't write data to : {}".format(database_filename))                       
    

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