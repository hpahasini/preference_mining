from SentimentAnalysis import SentimentAnalysis
from ConfusionMatrix import ConfusionMatrix
from plotCharts import PlotCharts
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
    csvFilepath = "@AmilDilshan_1X.csv"
    fileName = os.path.basename(csvFilepath)
    username = os.path.splitext(fileName)[0]

    print("_____________________________________")
    print("User Name --- ::", username)

    # ass[0]igning sentimentalAnalysis class into sentiAnalyse
    sentiAnalyse = SentimentAnalysis()

    # calling dicToArray() function in SentimentalAnalysis class
    sentiAnalyse.dicToArray()
    dict_precentages = {}
    predictionArray = []
    actualArray = []
    ScoreList = {}
    preferences = {}
    sentence_dict = json.loads(open("data2222.json", "r").read())
    for key in sentence_dict.keys():
        tweets = sentence_dict.get(key)
        # calling sentiment_analysis() function in SentimentalAnalysis class
        analysis_results, precentages = sentiAnalyse.sentiment_analysis(tweets)
        dict_precentages[key] = precentages
        predictions = analysis_results["Prediction_Array"]
        predictionArray = predictionArray + predictions
        tot_tweets = posit = negat = neut = 0
        pos = []
        for tweet in tweets:
            tot_tweets += 1
            # print(tweet)
            j = get_tweet_sentiment(tweet)
            pos.append(j)
            # print(pos)
            if j == 1:
                posit += 1
            elif j == 3:
                negat += 1
            elif j == 2:
                neut += 1
        actualArray = actualArray + pos
        ScoreList[key] = posit - negat
        preferences[key] = {"total": tot_tweets, "precentages": precentages}
    # toPlot = {"user": username, "preferences": preferences}
    # print(toPlot)

    conf_matrix = ConfusionMatrix(actualArray, predictionArray)
    Ranking = preferenceRanking(ScoreList)
    for index in range(len(Ranking)):
        print("")
        category = Ranking[index][0]
        print((index + 1), " --- ", category)
        print("Positive : ", dict_precentages[category]["positive"])
        print("Negative : ", dict_precentages[category]["negative"])
        print("Neutral  : ", dict_precentages[category]["neutral"])
    print("")
    conf_matrix.getAccurecyScore()
    conf_matrix.getClassificationReport()
    # plot = PlotCharts(toPlot)
    # plot.createCharts()


# calling main
if __name__ == "__main__":
    main()
