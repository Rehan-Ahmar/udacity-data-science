import json
import re
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
    return tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = df.drop(['id','message','original','genre'], axis=1)
    category_counts = categories.sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    
    words = pd.Series(' '.join(df['message'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)).lower().split())
    top_10_counts = words[~words.isin(stopwords.words("english"))].value_counts()[:10]
    top_10_words = list(top_10_counts.index)
    
    counts = categories.groupby(categories.sum(axis=1)).count()['related']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle' : 30
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=counts.index,
                    y=counts.values
                )
            ],

            'layout': {
                'title': 'Distribution of marked categories per message',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of marked categories"
                },
            }
        },
        {
            'data': [
                Bar(
                    x=top_10_words,
                    y=top_10_counts
                )
            ],

            'layout': {
                'title': 'Top 10 most frequent words in messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names[::-1],
                    z=categories.corr().values
                )
            ],

            'layout': {
                'title': 'Heatmap of categories',
                'xaxis': {
                    'tickangle' : 30
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()