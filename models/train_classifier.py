import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
        Loads the Disaster Reponse messages and categories from an SQL databse
        created by an ETL-pipeline as defined in process_data.py  
        
        Args:
            database_filepath   str                 filepath to SQL databse
            
        Returns:
            X                   pandas.DataFrame    messages extracted from SQL database
            Y                   pandas.Datafrane    category labels from SQL database
            category_names      list of str         names of category labels    
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponses', con = engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original','genre'], axis=1)
    category_names = Y.columns.tolist()

    return  X, Y, category_names


def tokenize(text):
    """
        Transforms the text into a list of cleaned tokens
        Cleaning consists of: 
        - splitting text into tokens
        - removing stopwords
        - case normalizing 
        - lemmatizing
        
        Args:
            text            str             text to be tokenized
            
        Returns:
            clean_tokens    list of str     list of cleaned tokens
    """
    # extract words
    tokens = word_tokenize(text)
    # remove stopwords
    tokens = [t for t in tokens if not t in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for token in tokens:
        # normalize & lemmatize
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)    
    
    return clean_tokens


class ContainingNumberExtractor(BaseEstimator, TransformerMixin):
    """
        Class to apply the additional feature "contains cardinal number"
        in the ML pipeline
    """

    def containing_number(self, text): 
        """
            Function too apply additional Feature
            
            Args:
                text     str
            
            Returns
               boolean   True in case text contains cardinal number
        """
        pos_tags = nltk.pos_tag(tokenize(text))
        for word, tag in pos_tags:
            if tag == 'CD':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.containing_number)
        return pd.DataFrame(X_tagged)



def build_model():
    """
        Creates the ML model to classify disaster responses
        Model is created in form of a pipeline that combines
        CountVectorizer, TfidfTransofmer, the additional feature
        ContainingNumberExtractor() and SVC as classifier
        The Model uses GridSearch for Hyperparamter tuning
        
        Args:
            None
            
        Returns
            pipeline    Multiclassification model    built model
    """
    pipeline = Pipeline([
      ('features', FeatureUnion([
          ('text_pipeline', Pipeline([
              ('vect', CountVectorizer(tokenizer=tokenize)),
              ('tfidf', TfidfTransformer())                                     
          ])),
          ('containing_number', ContainingNumberExtractor())                                          
      ])),
      ('clf', MultiOutputClassifier(OneVsRestClassifier(SVC())))
    ])   
    
    params_svc = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__tfidf__norm': ['l1', 'l2'],
        'clf__estimator__estimator__kernel': ['linear', 'rbf'],
    }
    
    cv = GridSearchCV(pipeline, param_grid=params_svc, verbose=3, n_jobs=3)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluates the model according to provided testdata and generates
        a classification report incl. F1 score and calculates and outputs accuracy 
        
        Args:
            model:          Mulitclass model    trained classification model
            X_Test          pandas.DataFrame    Features of test data
            Y_test          pandas.DataFrame    Labels of test data            
            category_names  list of str         List of category names
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print('Acurracy: {:.1f}%'.format(np.mean(Y_test.values==Y_pred)*100))

def save_model(model, model_filepath):
    """
        Stores provided model in pickle format
        
        Args:
            model           Mulitclass model    trained model
            model_filepath  str                 filepath where to store model
        Returns:
            None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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