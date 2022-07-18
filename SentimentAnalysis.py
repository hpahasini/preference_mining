import csv
import numpy as np
import json


class SentimentAnalysis:
    def __init__(self):
        self.np_dictionary = self.dicToArray()

    def get_words_in_tweet(self, tweet):
        words_in_tweet = []
        if len(tweet) != 0:
            words_in_tweet = tweet.split()
        return words_in_tweet

    def load_dictionary(self, file_path):
        dictionary_list = []
        with open(file_path, "r") as f:
            csv_data = csv.reader(f, delimiter=",")
            for row in csv_data:
                word_list = []
                word_list.append(row[0])  # Add context of word i.e adjective, noun, etc
                word_list.append(row[1])  # Add the score of word
                word_list.append(row[2])  # Add  words
                dictionary_list.append(
                    word_list
                )  # Add details of the word into the dictionary
        return dictionary_list

    def compute_sentiment_scores_of_word(self, word, dictionary_indices):
        rows = dictionary_indices[0]
        score_sum = 0
        for r in rows:
            score_sum += float(self.np_dictionary[r][1])
        avg_score_word = score_sum / (len(rows))
        return avg_score_word

    def sentiment_analysis(self, cleaned_tweets):
        count = 0
        total = 0
        positive_tweets = 0
        negative_tweets = 0
        neutral_tweets = 0
        sentiment_analysis_results = []
        # to send the results to the final output
        analysis_results = {"Sentiment_Type":None, "Sentiment_Score": None, "Prediction_Array": None}

        analysis_results_to_plot = []
        predictsPerKey = []
        for c_tweet in cleaned_tweets:
            count += 1
            #print(c_tweet, ":")
            analysis_result_tweet = {}
            words = self.get_words_in_tweet(c_tweet)
            if not words:  # If tweet is empty (exception handling)
                sentiment_analysis_results.append(analysis_result_tweet)
            else:
                analysis_result_tweet["cleaned_tweet"] = c_tweet
                analysis_result_tweet["tweet_words"] = words
                scores_sum = 0
                tweet_words_scores = []
                for word in words:
                    word_scores = {}
                    indices = np.where(self.np_dictionary == str(word))
                    dict_rows = indices[0]
                    if (
                        len(dict_rows) == 0
                    ):  # If the word is not present in the dictionary.
                        result = 0  # Consider it neutral
                        word_scores["word"] = str(word)
                        word_scores["score"] = str(result)
                        tweet_words_scores.append(word_scores)
                        continue
                    else:
                        sentiment_score = self.compute_sentiment_scores_of_word(
                            word, indices
                        )
                        word_scores["word"] = str(word)
                        word_scores["score"] = str(sentiment_score)
                        tweet_words_scores.append(word_scores)
                        scores_sum += float(
                            sentiment_score
                        )  # Sum up score of every word in tweet
                word_count = len(words)
                tweet_sentiment_score = scores_sum / word_count
                total += tweet_sentiment_score
                j = None
                if tweet_sentiment_score > 0:
                    tweet_sentiment = "positive"
                    #print("Positive")
                    positive_tweets += 1
                    j = 1
                elif tweet_sentiment_score < 0:
                    tweet_sentiment = "negative"
                    #print("Negative")
                    j = 2
                    negative_tweets += 1
                else:
                    tweet_sentiment = "neutral"
                    #print("Neutral")
                    j = 3
                    neutral_tweets += 1
                predictsPerKey.append(j)
                analysis_results["Sentiment_Type"] = tweet_sentiment
                analysis_results["Sentiment_Score"] = tweet_sentiment_score
                # print(tweet_sentiment_score)
                # print(tweet_sentiment)
        # print(predictsPerKey)
        precentages={"positive":None,"negative": None,"neutral": None}
        positive_precentage = round((positive_tweets/count)*100,2)
        negative_precentage = round((negative_tweets/count)*100,2)
        neutral_precentage = round((neutral_tweets/count)*100,2)
        precentages={"positive":positive_precentage,"negative": negative_precentage,"neutral": neutral_precentage}
        analysis_results["Prediction_Array"] = predictsPerKey
        return analysis_results,precentages

        # analysis_result_tweet["tweet_words_scores"] = tweet_words_scores
        # analysis_result_tweet["basic_avg_sentiment_score"] = str(tweet_sentiment_score)
        # analysis_result_tweet["basic_avg_sentiment"] = tweet_sentiment
        # sentiment_analysis_results.append(analysis_result_tweet)
        # analysis_results_to_plot.append(str(tweet_sentiment_score))

    def dicToArray(self):
        dictionary_list = self.load_dictionary(
            "sentiOut.csv"
        )  # Change file path as required
        # global np_dictionary
        np_dict = np.array(dictionary_list)
        return np_dict
