#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:13:59 2020

@author: Sparsh
"""

# Importing libraries for our use

import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


## API Credentials

# Enter your own credentials
consumer_key = "enter_yours"
consumer_secret = "enter_yours"
access_token = "enter_yours"
access_token_secret = "enter_yours"

# Enter details here
query_value = "enter_your_query_here"
start_date = '2000-6-1'
end_date = '2020-12-1'
# 3T means 3 minutes, W for week, M for month
time_period = '3T'

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth) 


# Data collection from tweepy
Data = pd.DataFrame(columns=['Date','Keyword','Location', 'Tweet', 'Label'])
# Creating the API object while passing in auth information
api = tweepy.API(auth)

# The search term you want to find
# query_value = "enter_your_query_here"
query = query_value+" -filter:retweets"
# Language code (follows ISO 639-1 standards)
language = "en"

# Calling the user_timeline function with our parameters
results = api.search(q=query, lang=language, count=100)

# foreach through all tweets pulled
for tweet in results:
  if (tweet.user.location):
    Data = Data.append({'Date' : tweet.created_at, 'Keyword' : query_value,
                        'Location' : tweet.user.location, 'Tweet': tweet.text, 'Label' : 0}, ignore_index=True)
  else:
    Data = Data.append({'Date' : tweet.created_at, 'Keyword' : query_value,
                        'Location' : "Nan", 'Tweet': tweet.text, 'Label' : 0}, ignore_index=True)


# Entering the date filter for filtering tweets between dates
# Data.head(5)
# start_date = '2000-6-1'
# end_date = '2020-12-1'
Data = Data[(Data['Date'] >= start_date) & (Data['Date'] <= end_date)]

# Saving data to Twitts.csv
Twitts = Data.drop(['Label'], axis = 1)
Twitts.to_csv('Twitts.csv', index=False)

# Create a function to clean the tweets
def cleanTxt(text):
 text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
 text = re.sub('#', '', text) # Removing '#' hash tag
 text = re.sub('RT[\s]+', '', text) # Removing RT
 text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
 
 return text


# Clean the tweets
Data['Tweet'] = Data['Tweet'].apply(cleanTxt)
Data['Tweet'].replace("[^a-zA-Z]"," ",regex=True, inplace=True) 
# Show the cleaned tweets
# Data

# Sentiment analysis model part starts

# Loading our sentiment analysis model
from flair.models import TextClassifier
from flair.data import Sentence
classifier = TextClassifier.load('en-sentiment')

# Running for all rows
for i in range(len(Data)):
  try:
      sentence = Sentence(Data['Tweet'][i])
      # Predicting the sentiment
      classifier.predict(sentence)
      value = sentence.labels[0].to_dict()['value']
      # If sentiment is positive : we take it's value, else we multiply it by -1
      if value == 'POSITIVE':
        result = sentence.to_dict()['labels'][0]['confidence']
      else :
        result = -(sentence.to_dict()['labels'][0]['confidence'])
      # Storing the result, back in our dataframe in Label column
      Data['Label'][i] = result
      # print('Sentence above is: ', sentence.labels)
  except:
      Data['Label'][i] = 0

# Data.info()
# Changing datatype, so that we can use time-series analysis
Data['Date'] = pd.to_datetime(Data['Date'])
Data['Label'] = Data['Label'].astype(float)


# Resampling for getting data into periods
# 3T : 3 minutes, 30S : 30 seconds, M : Month, W : week
# Getting sentiment score by Keyword and Location
final_data = Data.groupby(['Keyword','Location']).resample(time_period, on='Date').mean().reset_index()
final_data.to_csv('Sentiment.csv', index=False)
print(final_data)

# # Getting sentiment score by only Keyword
# final_data = Data.groupby(['Keyword']).resample('3T', on='Date').mean().reset_index()
# final_data.to_csv('Sentiment.csv', index=False)
# print(final_data)

# # Getting sentiment score only by periods
# final_data = Data.resample('3T', on='Date').mean().reset_index()
# final_data.to_csv('Sentiment.csv', index=False)
# print(final_data)