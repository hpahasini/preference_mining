from SentimentAnalysis import SentimentAnalysis
from ConfusionMatrix import ConfusionMatrix
import numpy as np
import os
import csv
import json
from nltk import word_tokenize
from textblob import TextBlob


def get_tweet_sentiment(tweet):
    # creating TextBlob object of the tweet
    analysis = TextBlob(tweet)
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 3
    else:
        return 2


def preferenceRanking(scoreList):
    SortedScore = list(sorted(scoreList.items(), key=lambda kv: kv[1], reverse=True))
    return SortedScore


def main():
    csvFilepath = "@wenushika.csv"
    fileName = os.path.basename(csvFilepath)
    username = os.path.splitext(fileName)[0]

    print("_____________________________________")
    print("User Name --- ::",username)

    # ass[0]igning sentimentalAnalysis class into sentiAnalyse
    sentiAnalyse = SentimentAnalysis()

    # calling dicToArray() function in SentimentalAnalysis class
    sentiAnalyse.dicToArray()
    # labeledTweets={}
    predictionArray = []
    precent_N_Keys ={}
    actualArray = []
    ScoreList = {}
    sentence_dict = json.loads(open("data2222.json", "r").read())
    for key in sentence_dict.keys():
        tweets = sentence_dict.get(key)
        analysis_results,precentages = sentiAnalyse.sentiment_analysis(tweets)
        ScoreList[key] = precentages["positive"]
        precent_N_Keys[key] = precentages
        predictions = analysis_results["Prediction_Array"]
        predictionArray = predictionArray + predictions
        tot_tweets = posit = negat = neut = 0
        pos = []
        for tweet in tweets:
            tot_tweets += 1
            j = get_tweet_sentiment(tweet)
            pos.append(j)
            if j == 1:
                posit += 1
            elif j == 3:
                negat += 1
            elif j == 2:
                neut += 1
        actualArray = actualArray + pos
    conf_matrix = ConfusionMatrix(actualArray, predictionArray)
    Ranking = preferenceRanking(ScoreList)
    for index in range(len(Ranking)):
        curr_Key = Ranking[index][0]
        positive_precentage = precent_N_Keys[curr_Key]["positive"]
        negative_precentage = precent_N_Keys[curr_Key]["negative"]
        neutral_precentage = precent_N_Keys[curr_Key]["neutral"]

        print("")
        print((index + 1)," --- ", curr_Key)
        print("Positive : ",positive_precentage)
        print("Negative : ",negative_precentage)
        print("Neutral  : ",neutral_precentage)
    print("")   
    conf_matrix.getAccurecyScore()
    conf_matrix.getClassificationReport()


# calling main
if __name__ == "__main__":
    main()

