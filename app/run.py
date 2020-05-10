import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponses', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # count messages per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # share of messages labelled by categories
    category_shares = (df.iloc[:, 4:].sum()/df.shape[0]).sort_values(ascending=False)
    category_names = list(category_shares.index)
    
    # count the number of categories the messages are allocated to
    message_category_counts = df.iloc[:, 4:].sum(axis=1)
    
    # build category co-occurence matrix
    # https://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas
    category_labels = df.iloc[:,4:].columns.tolist()
    data = df.iloc[:,4:]

    # Compute cooccurrence matrix 
    cooccurrence_matrix = np.dot(data.transpose(),data)

    # Compute cooccurrence matrix in percentage
    # FYI: http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    #      http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero/32106804#32106804
    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))

    
    # create visuals
   
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
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
                    y=category_shares,
                    x=category_names,
                )
            ],

            'layout': {
                'title': 'Share of messages labelled by categories',
                'xaxis': {
                    'title': "Categories",
                    'tickangle':45
                },
                'yaxis': {
                    'tickformat': "%"
                }
            }
        },
        {    
            'data': [
                Histogram(
                    x=message_category_counts,
                )
            ],

            'layout': {
                'title': 'Histogram on number of categories messages are allocated to',
                'xaxis': {
                    'title': "Categories",
                },
                'yaxis': {
                    'title': "counts",
                    'range': [0,7000]
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x = category_labels,
                    y = category_labels,
                    z = cooccurrence_matrix_percentage,
                    colorscale = 'Blues',
                    reversescale = True
                )
            ],

            'layout': {
                'title': 'Co-occurence of categories in messages',
                'xaxis': {
                    'tickfont' : {'size':8}
                },
                'yaxis': {
                    'tickfont' : {'size':8}
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