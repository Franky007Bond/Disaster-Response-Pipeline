

# Disaster Response Pipeline

## Table of Contents

- [Introduction & Motivation](#introduction)
- [Quick Start incl. Installation](#quick_start)
    - [Instructions](#instructions)
- [Structure of pipeline](#structure_pipeline)
    - [ETL-Pipeline](#etl)
    - [ML-Pipeline](#ml)
        + [Accuracy score vs. accuracy mean](#accuracy)
        + [Feature Engineering](#feature_engineering)
        + [Grid Search](#grid_search)
    - [Web App](#web_app)
- [Discussion & Outlook](#discussion)
    + [Dealing with imbalanced dataset](#imbalanced_dataset)
    + [Using more sophisticated language models](#language_models)
- [Structure of repository](#structure_rep)
- [Contributions](#contributions)

## Introduction & Motivation<a name="introduction"></a>
This project has been initiated as part of my [udacity](https://www.udacity.com/) Data Scienctist nanodegree program.

The use case comprises analyzing and classifying disaster messages provided by [Figure Eight](https://www.figure-eight.com/). Following a disaster, help organizations typically receive millions of communications either direct or via social media right at the time when they have the least capacity to filter and pull out the messages which are the most important ones.

As usually many different organizations are involved in the disaster response and take care of different parts of the problem, to forward the messages to the right contacts is crucial. This leads to the requirement to classify the messages and allocated them to specific categories.

## Quick Start incl. Installation<a name="quick_start"></a>
The pipeline is programmed in Python 3.  Following libraries are required to successfully run the app:

* pandas
* numpy
* sklearn
* sqlalchemy   
* pickle
* json
* plotly
* flask

To run the notebooks included in the repository you further need the libraries:

* matplotlib
* seaborn 

### Instructions<a name="instructions"></a>
After downloading / cloning the repository following steps are necessary to start the app:

>1. Run the following commands in the project's root directory to set up your database and model.

>       - To run the ETL pipeline that cleans the data and stores it in the database
>    
         ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 
       ```

>        - To run the ML pipeline that trains the classifier and saves it
>  
         ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
         ```

>2. Run the following command in the app's directory to run your web app.
>
    ```python run.py```

>3. Open the following web address in a browser:  http://0.0.0.0:3001/

## Structure of pipeline<a name="structure_pipeline"></a>
The following figure shows an overview of the different elements of the pipeline:

![Disaster Pipeline Structure](\assets\Disaster-Response-Pipeline-Structure.png)

### ETL-Pipeline<a name="etl"></a>
Within the ETL-pipeline, data preparation steps are executed, including dropping nan values (even if the original dataset does not contain any nan values, the pipeline can deal with it), encoding the different categories and dropping any duplicate datasets.
After the cleaning procedure the data is stored in an SQLite database.
To get an impression on the structure of the dataset, please have a look at the corresponding workbook ETL Pipeline Preparation.ipynb where some Exploratory Data Analysis is performed.

### ML-Pipeline<a name="ml"></a>
The Machine learning pipeline uses [term frequency–inverse document frequency (TFIDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to retrieve the information in the disaster messages.

To categorize the messages based on the TFDIF representation, several Machine learning models have been benchmarked (see notebook Desaster_Response_ML_Pipeline.ipynb):

* Random Forest Classifier
* AdaBoost
* Support Vector Machines
* Naïve Bayes

The following table compares the performance of the different models when the standard parameters are applied:

|Classifier     | Precision |  Recall  | F1-score | Acc. score | Acc.mean |
|---------------|-----------|----------|----------|------------|----------|
| Random Forest |  0.767281 | 0.520754 | 0.560696 |   0.251875 | 0.947623 |
| Ada Boost     |  0.728436 | 0.574396 | 0.621264 |   0.251240 | 0.947454 |
| SVM           |  0.779908 | 0.561096 | 0.604282 |   0.298156 | 0.950692 |
| Naive Bayes   |  0.539227 | 0.371198 | 0.375510 |   0.163001 | 0.935502 |

#### Accuracy score vs. accuracy mean<a name="accuracy"></a>
The difference between the two accuracy columns persists in different definition. The first column *accuracy score* show the results of the ```sklearn.accuracy_score``` method whereas the column *accuracy mean* shows the values for ```np.mean(y_predict == y_test)```, i.e. the former does a comparison on the complete categories vectors (all 36 values must comply), whereas the latter one does an element-wise assessment. You can find more details in [this post](https://datascience.stackexchange.com/questions/58961/sklearn-accuracy-scorey-test-y-predict-vs-np-meany-predict-y-test).

To further enhance the performance, two additional techniques have been implemented:

* Feature Engineering
* Hyperparameter Tuning via GridSearch

#### Feature Engineering<a name="feature_engineering"></a>
Following additional features have been tested:

* At least one sentence in the message is starting with a verb
* Message contains url
* Message contains a cardinal number
* Message length in characters
* Message lengths in words

The charts below show the spread of these features among the different categories. You can find the corresponding png-files in the assets subfolder.

| [![Feature Benchmark 1](\assets\Feature_Engineering_1.png)](\assets\Feature_Engineering_1.png) | [![Feature Benchmark 2](\assets\Feature_Engineering_2.png)](\assets\Feature_Engineering_2.png) |
|:---:|:---:|
| [![Feature Benchmark 3](\assets\Feature_Engineering_3.png)](\assets\Feature_Engineering_3.png)|    |

As the feature *Message contains a cardinal number* shows the widest spread, this one has been included in the pipeline.

#### Grid Search<a name="grid_search"></a>
Grid Search is a way to explore the optimal Hyperparameters It is a computationally very greedy approach as every single combination of the parameters is tested.
Due to the computational expense, the search space has been limited to 8 – 12 pairs of parameter values for the different models.
The table below shows the performance results for the best parameter sets for each model:

|Classifier     | Precision |  Recall  | F1-score | Acc. score | Acc.mean |
|---------------|-----------|----------|----------|------------|----------|
| Random Forest |  0.764857 | 0.496484 | 0.546002 |   0.264336 | 0.946521 |
| Ada Boost     |  0.720281 | 0.581750 | 0.629862 |   0.264590 | 0.947369 |
| SVM           |  0.762571 | 0.592719 | 0.637009 |   0.315321 | 0.951971 |
| Naive Bayes   |  0.381635 | 0.248403 | 0.222292 |   0.198220 | 0.927333 |

As you can see the accuracy score improved for all the models. As the Support Vector Machine model outperforms the other algorithms in all indicators, this is the one that finally gets implemented.

### Web App<a name="web_app"></a>
The Web App uses the Flask framework and the plotly library for graphics visualization.

The Web App illustrates charts on some general statistics of the disaster response dataset, e.g. the share of messages that are labelled with the different categories or the co-occurrence of the categories within the same message.

Furthermore, the text input field at the top of the page allows to classify  new disaster response messages.

![Web App screenshot](\assets\Web-App.png)

![Web App screenshot - categorized_message](\assets\Categorized_message.png)

## Discussion & Outlook<a name="discussion"></a>
As we can see from the tables above there is still room for improvement with regard to the classification performance.
Following are some thoughts on potential approaches:

### Dealing with imbalanced dataset<a name="imbalanced_dataset"></a>
The used dataset is imbalanced. As we find examples for some labels in more than 50% of all messages, for some other labels we hardly have any example. To increase the number of examples for these labels we could use different approaches:

* Manually add further annotated examples to the dataset, which could easily become  pretty time-consuming.
* Jason Wei proposes four [EDA (Easy Data Augmentation) techniques]( https://towardsdatascience.com/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610) for boosting performance on text classification tasks, i.e. Synonym Replacement, Random Insertion, Random Swap, Random Deletion.
* Further ideas are suggested by Edward Ma which include (Contextualized) Word Embeddings, Back Translation (translating the original text from the source language to a target language and back to the source language will most likely not result in the original text but in another example) or even Text Generation. For more details please see his [post on TowardsDataScience]( https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28)

### Using more sophisticated language models<a name="language_models"></a>
TFIDF is quite a simple approach to analyze text as it relies only on the frequency of the occurrence of the tokens, but not e.g. on the context.
Throughout the last years, tremendous progress has been achieved in the field of NLP, starting with word embedding models ([Word2Vec]( https://en.wikipedia.org/wiki/Word2vec), 2013 by Google  & [fastText]( https://en.wikipedia.org/wiki/FastText), 2015 by Facebook), and representations based on  aggregated global word-word co-occurrence statistics from a corpus ([GloVe]( https://nlp.stanford.edu/projects/glove/), 2014 by Stanford) to more recent Transformer language models ([BERT]( https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html), 2018 by Google & [GPT-2]( https://openai.com/blog/better-language-models/) by OpenAI)
Applying Transfer Learning we could re-use one of these pretrained models and only re-train the last layers on our classification task.

## Structure of Repository<a name="structure_rep"></a>

```
- app
| - template
| |- master.html    # main page of web app
| |- go.html        # classification result page of web app
|- run.py           # Flask file that runs app

- assets            # subfolder containing graphics & screenshots
|-

- data
|- disaster_categories.csv  # data to process - message categories 
|- disaster_messages.csv    # data to process - disaster response messages
|- process_data.py          # ETL-pipeline
|- DisasterResponse.db      # database to save cleaned data 

- models
|- train_classifier.py      # ML-pipeline
|- classifier.pkl           # saved model 

- Desaster_Response_ML_Pipeline.ipynb   # Feature Engineering & Grid Search benchmark
- ETL Pipeline Preparation.ipynb        # Exploratory Data Analysis
- README.md
```

## Contributions<a name="contributions"></a>
The analysis on the category co-occurrence has been inspired by Franck Dernoncourt’s reply on this  [stackoverflow post]( https://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas). 
