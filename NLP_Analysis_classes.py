# In this notebook we will define classes to use in NLP analysis_notebook:

# Import libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
import re
import nltk
import string
import pickle

from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from gensim.summarization.textcleaner import split_sentences

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from matplotlib import pyplot
from textblob import TextBlob, Word, Blobber

# # to install textblob in your conda packages:
# # 1. go to anaconda prompt
# # 2. cd Anaconda3>Scripts>conda install -c conda-forge textblob

import numpy as np
import nltk
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')
from nltk import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# Define classes:
class sentiment_analysis:
    '''
    this class finds the polarity over time for any given boutique names, calculate the polarity average and displays
    the lineplot for clarification 
    '''
    def __init__(self, boutique_reviews):
        self.boutique_reviews = boutique_reviews
         
    def clean_review(self, boutique_list):
        """
        This function cleans a block of text by applying: text_cleaning and remove_stopwords.
        Input:text = the text to be cleaned.
        Output: the text stripped of punctuation and made lowercase, with no stopwords.
        """        
        subset_df = self.boutique_reviews[self.boutique_reviews['boutique_names'].isin(boutique_list)]
        #subset_df = boutique_reviews[self.boutique_reviews.boutique_names in boutique_list]
        for review in subset_df["reviews"]:
            # u'\xa0' represents a non-breaking space in the text block that needs to be removed.
            review = review.replace(u'\xa0', u' ')

            #remove multiple fullstops and make a single fullstop
            review = re.sub('\.+', '. ', review)

            #remove multiple spaces and make a single space.
            review = re.sub(' +', ' ', review)

            #remove all tokens that are not alphabetic
            review = re.sub(r'\d+', '', review)

            #normalization
            review =  review.lower()

            #Define punctuations according to nltk corpus.
            punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~.+'''

            #remove punctuations, traverse the given string and if any punctuation marks occur replace it with null 
            for i in review: 
                if i in punctuations: 
                    review = review.replace(i, "") 

            tokens = word_tokenize(review)
            #remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if not token in stop_words]
            #return the cleaned text in a sentence format.
            cleaned_review = ' '.join([''.join(token) for token in tokens])
            subset_df_clean = subset_df.replace(review, cleaned_review, inplace=False)

        return subset_df_clean
    
    
    def subset_df_sentiment(self, subset_df_clean):
        polarity = []
        subjectivity = []
        for review in subset_df_clean["reviews"]:
            polarity.append(TextBlob(review).sentiment.polarity)
            subjectivity.append(TextBlob(review).sentiment.subjectivity)

        subset_df_clean_sentiment = subset_df_clean
        subset_df_clean_sentiment['polarity'] = polarity
        subset_df_clean_sentiment['subjectivity'] = subjectivity
        return subset_df_clean_sentiment

    
    def df_polarity(self, subset_df_clean_sentiment):
        """
        Returns a Pandas DataFrame containing the date and polarity for each review.
        Columns = date, polarity
        """
        dates = subset_df_clean_sentiment["review_dates"]
        polarity = subset_df_clean_sentiment["polarity"]
        review = subset_df_clean_sentiment["reviews"]

        data = {"date":dates, "polarity":polarity, "review":review}
        df = pd.DataFrame(data).sort_values(by=['date'], ascending=True)#.set_index('date')
        #print(df)
        return df
    
    
    def df_polarity_average(self, df, window=720):
        """
        This function will provide a dataframe containing the sentiment polarity moving average 
        for the window chosen.
    
        Input: df = Pandas dataframe containing review polarity and date columns.
    
        Output: Returns a Pandas dataframe with date and polarity columns representing the moving average polarity every 30 days
        for the chosen window size.
        """
        current_date = df.date.min()
        end_date = df.date.max()

        print("current_date: ", current_date)
        print("end_date: ", end_date)
        window_start = current_date - pd.Timedelta(int(window/2), unit ='D')
        window_end = current_date + pd.Timedelta(int(window/2), unit ='D')

        print("window_start: ", window_start)
        print("window_end: ", window_end)    

        time_delta = pd.Timedelta(30, unit='D') #How often to calculate average
        print("time_delta: ", time_delta)    

        d = []
        while current_date < end_date:
            polarity_average = df.polarity[(df.date < window_end) & (df.date > window_start)].mean()
            d.append({'date':current_date,'polarity':polarity_average})
            window_start += time_delta
            window_end   += time_delta
            current_date += time_delta
            dd = pd.DataFrame(d)
        return dd
    
    
    def display_polarity(self, df, dd, boutique_list):
        plt.figure(figsize=(10,4))
        for boutique in boutique_list:
            sns.lineplot( x="date", y="polarity", data= df, color="blue",alpha=0.3,label=f"{boutique}(polarity)")
            sns.lineplot( x="date", y="polarity", data= dd, color="red",alpha=0.3,label=f"{boutique} (polarity_average)")
            plt.legend(prop={'size': 11})
        plt.grid(alpha=0.3)
        plt.show()
            
    def sentiment_analysis_summary(self, boutique_list):
        subset_df_clean = self.clean_review(boutique_list)
        subset_df_clean_sentiment = self.subset_df_sentiment(subset_df_clean)
        df = self.df_polarity(subset_df_clean_sentiment)
        dd = self.df_polarity_average(df)
        self.display_polarity(df, dd, boutique_list) 
        #return df
       
    def compare_polarity_vs_ratings(self, subset_df_clean):
        print(len(subset_df_clean), "reviews used")
        ratings = sorted(subset_df_clean.review_ratings.unique(),reverse=True)
        plt.figure(figsize=(20,8))
        for rating in ratings:
            sns.distplot(subset_df_clean.polarity[subset_df_clean.review_ratings==rating], kde=True, label=f"{rating} stars")
            plt.legend()
            plt.xlim((-0.75,1))
            plt.xlabel('Sentiment Polarity');
            title = 'Distribution of Review Sentiment by Star Rating'
            