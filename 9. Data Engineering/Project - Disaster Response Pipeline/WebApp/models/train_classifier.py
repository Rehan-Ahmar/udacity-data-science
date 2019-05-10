import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    Loads data from the specified sqlite database.
    Args:
        database_filepath(string) : path to sqlite database.
    Returns:
        pandas.DataFrame : dataframe of input feature(message).
        pandas.DataFrame : dataframe of output labels(categories).
        list: list of category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    return X, Y, Y.columns.tolist()


def tokenize(text):
    """
    Cleans, tokenizes, removes stopwords and stems the input text.
    Args:
        text(string) : input text.
    Returns:
        list : list of cleaned and stemmed tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()   
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    tokens = [lemmatizer.lemmatize(tok, pos='v') for tok in tokens]
    #tokens = [PorterStemmer().stem(tok) for tok in tokens]
    return tokens

def build_model():
    """
    Returns a machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100), n_jobs=-1))
    ])
    return pipeline

def build_gridsearch_model():
    """
    Returns a GridSearchCV object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100]
        #'clf__estimator__n_estimators': [100, 200, 500],
        #'clf__estimator__max_depth': [None, 10, 20, 50],
        #'clf__estimator__min_samples_leaf': [1, 2, 4],
        #'clf__estimator__min_samples_split': [2, 5, 7]
    }
    return GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints the accuracy and classification report for the given model and test data.
    Args:
        model(skicit-learn model/pipeline) : model/pipeline to be evaluated.
        X_test(pandas.DataFrame) : test input variables.
        Y_test(pandas.DataFrame) : test output variables.
        category_names(list) : list of output category names.
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    print('Accuracy:\n{}'.format((y_pred == Y_test).mean()))
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves the input model as a pickle file at the specified filepath.
    Args:
        model(scikit-learn model) : model to be saved.
        model_filepath(string) : model filepath.
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        #model = build_gridsearch_model()
        
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