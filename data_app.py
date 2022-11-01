import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import datetime
from io import StringIO 
import pickle
model_1 = pickle.load(open('model_1_rf.sav', 'rb')) #load random forest for prediction of neutral/ non- neutral sentiment
model_2 = pickle.load(open('model_2_svm.sav', 'rb')) #load random forest for prediction of positive/ negative sentiment
tfid_vect = pickle.load(open('vectorizer.pickle', 'rb')) #load vectorizer
tfid_vect2 = pickle.load(open('vectorizer2.pickle', 'rb')) #load vectorizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def process_text(text): #preprocessing function that does stemming, lemmatizing, drop punctuations and removing stopwords
    if type(text) != str:
      text = str(text)
    text = text.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    stemmer = SnowballStemmer('english')
    
    
    lem_text = lemmatizer.lemmatize(text)
    stem_text = stemmer.stem(lem_text)
    
    drop_punc = [char for char in stem_text if char not in string.punctuation]
    drop_punc = ''.join(drop_punc)
    clean_text = [word for word in drop_punc.split() if word not in stopwords.words('english')]
    
    text = ' '.join(clean_text)
    
    return text

def clean_list_text(text):
  if type(text) != str:
    return text[0]
  else:
    return text

st.title('💻 LoL Sentiment Analyzer')


st.subheader('Previous Sentiments')

data_load_state = st.text('Loading data...')
data = pd.read_csv("main_data.csv")
data.dt = pd.to_datetime(data.dt)
data_load_state.text('Loading data...done!')
st.write(data.head(10))


st.subheader('Previous Trends')

data.rename(columns ={'sentiment': 'labelled_sentiment'}, inplace = True)
data['has_sentiment'] = data.labelled_sentiment.map({0: 0, -1: 1, 1:1})
sentiments =  data.loc[data.has_sentiment == 1][['dt','title','labelled_sentiment']]
senti_group = sentiments.groupby(['dt','labelled_sentiment'])['title'].count().reset_index()
senti_group = senti_group.rename(columns = {"title" : "count"})
senti_merge = senti_group.merge(sentiments.groupby(['dt'])['title'].count().reset_index() ,  how = 'left', on = 'dt')
senti_merge['percentage'] = senti_merge['count'] / senti_merge['title']
senti_percentage = senti_merge[['dt', 'percentage','labelled_sentiment']].pivot(columns = 'labelled_sentiment', index = 'dt')

senti_percentage.plot.bar(stacked = True, figsize=(10, 6), 
                          ylabel='percentage', xlabel= 'Date', title= 'LOL sentiment polarity')
plt.legend(title= 'Sentiments', bbox_to_anchor=(1.05, 1), loc='upper left')
dfmt = DateFormatter("%Y-%m-%d") # proper formatting Year-month-day
DateFormatter(dfmt)
plt.xticks(rotation= 90)
st.pyplot(plt)


# Sentiment analysis for user-uploaded csv file 

st.subheader('Add new data')

date_of_new_input = st.date_input("Specify the date of the new posts", min_value=datetime.date(2022, 8, 1))
st.write('Entered date is:', date_of_new_input)

uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    data_load_state = st.text('Loading new data...')
    # read new csv file
    input_df = pd.read_csv(uploaded_file)
    # add new date column
    input_df.insert(loc = 0, column="dt", value =pd.to_datetime(date_of_new_input)) 
    st.write(input_df.head(10))
    data_load_state.text('Loading new data...done!')
    #preprocessing 'title' row before feeding to model
    input_df['clean_title'] = input_df['title'].apply(lambda x : process_text(x)) 
    #perform scaling with mean and std dev
    input_X = tfid_vect.transform(input_df['clean_title']) 
    #make predictions with model 1 to determine whether sentiment is neutral or not
    y_pred = model_1.predict(input_X)
    new_df = input_df.copy()
    new_df['has_sentiment'] = y_pred

    #filter out neutral sentiments to prepare for model 2
    sentiment_df = new_df[new_df['has_sentiment']==1].copy()  
    input_new = tfid_vect2.transform(sentiment_df['clean_title'])
    y_pred_new = model_2.predict(input_new) 
    sentiment_df.loc[:,'sentiment'] = y_pred_new

    st.subheader("Sentiment Prediction for New Data")
    pie_labels = ["neutral (0)", "negative (-1)", "positive (1)"]
    pie_values = sentiment_df['sentiment'].value_counts()
    num_negative = pie_values[-1]
    num_positive = pie_values[1]
    num_neutral = new_df.value_counts('has_sentiment')[0]
    pie_sizes = [num_neutral, num_negative, num_positive]
    
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    
    st.pyplot(fig1)

    st.subheader("Identified Text with Sentiments")
    st.write(sentiment_df)

    st.subheader("New Sentiment Trends")
    sentiment_df.rename(columns ={'sentiment': 'labelled_sentiment'}, inplace = True)
    new_data_sentiments =  sentiment_df.loc[:,['dt','title','labelled_sentiment']]
    
    updated_sentiments = pd.concat([sentiments, new_data_sentiments], axis = 0)
    
    senti_group = updated_sentiments.groupby(['dt','labelled_sentiment'])['title'].count().reset_index()
    senti_group = senti_group.rename(columns = {"title" : "count"})
    senti_merge = senti_group.merge(updated_sentiments.groupby(['dt'])['title'].count().reset_index() ,  how = 'left', on = 'dt')
    senti_merge['percentage'] = senti_merge['count'] / senti_merge['title']
    senti_percentage = senti_merge[['dt', 'percentage','labelled_sentiment']].pivot(columns = 'labelled_sentiment', index = 'dt')

    senti_percentage.plot.bar(stacked = True, figsize=(10, 6), 
                              ylabel='percentage', xlabel= 'Date', title= 'LOL sentiment polarity')
    plt.legend(title= 'Sentiments', bbox_to_anchor=(1.05, 1), loc='upper left')
    dfmt = DateFormatter("%Y-%m-%d") # proper formatting Year-month-day
    DateFormatter(dfmt)
    plt.xticks(rotation= 90)
    st.pyplot(plt)




