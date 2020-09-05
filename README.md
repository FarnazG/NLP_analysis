# Web scraping and Natural language processing

This project works on data gathered from yelp website to assess San Francisco Clothing Boutiques through Natural language processing analysis.


# Project Breakdown

This project consists of 3 parts:


#### 1. Data gathering through web scraping: data for this project was gathered from yelp website and organized in two csv files

[Web_scraping_notebook](https://github.com/FarnazG/project006/blob/master/Web_Scraping_notebook.ipynb)

* sf_wclothing_boutiques_info.csv

a file with all san francisco womens clothing boutique names and information including contact_numbers, addresses, URL_pages, price_range and average_ratings

* sf_wclothing_boutiques_review.csv

a file with all san francisco womens clothing boutique names, number_reviews, reviews, each review dates and ratings.



#### 2. NLP analysis:[NLP_analysis_notebook](https://github.com/FarnazG/project006/blob/master/NLP_Analysis_notebook.ipynb)

This notebook analyzes reviews to extract required information and consists of:

* Applying Text_cleaning on reviews grouped by star_ratings

* Sentiment Analysis including Polarity and Subjectivity

![alt text](https://github.com/FarnazG/project006/blob/master/images/display_polarity_over_time_for_each_specific_boutique.png)

* Comparing review_polarities vs review_star_ratings to reflect which term is more informative

![alt text](https://github.com/FarnazG/project006/blob/master/images/compare_review_polarity_vs_review_star_ratings.png)

* TF_IDF and word_count analysis to reflect how important a word is to a review and how many times it has been repeated

![alt text](https://github.com/FarnazG/project006/blob/master/images/most_important_words_in_store_reviews.png)

* Word-Embeding analysis to gain further insights about boutique reviews

![alt text](https://github.com/FarnazG/project006/blob/master/images/word_embedding.png)



#### 3. Construct and fit a keras deep learning model to predict ratings using business reviews.

[NLP_deeplearning_model](https://github.com/FarnazG/project006/blob/master/NLP_Deeplearning_model.ipynb)
