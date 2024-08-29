"""
15-110 Hw6 - Social Media Analytics Project
Name: Shandon Herft
AndrewID: sherft
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### WEEK 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]


'''
parseLabel(label)
#3 [Check6-1]
Parameters: str
Returns: dict mapping str to str
'''
def parseLabel(label):
    # Splitting the label to extract name, position, and state
    parts = label.split("(")
    namePosition = parts[0].strip().split(":")[1].strip()
    positionState = parts[1].strip().split("from")
    position = positionState[0].strip()
    state = positionState[1].strip(")").strip()
    
    # Capitalize the first letter of each word in the name and position
    name = ' '.join(part.capitalize() for part in namePosition.split())
    position = ' '.join(part.capitalize() for part in position.split())
    
    return {"name": name, "position": position, "state": state}




'''
getRegionFromState(stateDf, state)
#4 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    # Filtering the DataFrame to find the row corresponding to the provided state
    stateRow = stateDf[stateDf['state'] == state]
    
    # If a row is found, return the region value, else return None
    if not stateRow.empty:
        return stateRow.iloc[0]['region']
    else:
        return None



'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    hashtags = []
    i = 0
    while i < len(message):
        if message[i] == '#':
            start_index = i
            end_index = i + 1
            while end_index < len(message) and message[end_index] not in endChars:
                end_index += 1
            hashtag = message[start_index:end_index]
            hashtags.append(hashtag)
            i = end_index
        else:
            i += 1
    return hashtags

'''
findSentiment(classifier, message)
#6 [Check6-1]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    # Get sentiment score using the classifier
    sentimentScore = classifier.polarity_scores(message)["compound"]
    
    # Map sentiment score to sentiment descriptor
    if sentimentScore > 0.1:
        return "positive"
    elif sentimentScore < -0.1:
        return "negative"
    else:
        return "neutral"


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    # Step 1: Create six new empty lists
    names = []
    positions = []
    states = []
    regions = []
    hashtags = []
    sentiments = []

    # Step 2: Create sentiment classifier
    classifier = SentimentIntensityAnalyzer()

    # Step 3: Iterate over "label" column
    for label in data["label"]:
        # Step 3a: Parse label to get name, position, and state
        parsedLabel = parseLabel(label)
        # Correct the name
        correctedName = parsedLabel["name"].strip().replace("Mcconnell", "McConnell")
        names.append(correctedName)  # Strip leading and trailing spaces
        positions.append(parsedLabel["position"])
        states.append(parsedLabel["state"])

        # Step 3b: Get region from state
        region = getRegionFromState(stateDf, parsedLabel["state"])
        regions.append(region)

    # Step 4: Iterate over "text" column
    for text in data["text"]:
        # Step 4a: Find hashtags in text
        textHashtags = findHashtags(text)
        hashtags.append(textHashtags)

        # Step 4b: Find sentiment of text
        sentiment = findSentiment(classifier, text)
        sentiments.append(sentiment)

    # Step 5: Assign lists to DataFrame columns
    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions
    data["hashtags"] = hashtags
    data["sentiment"] = sentiments

    # Return None
    return None





### WEEK 2 ###


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    # If colName and dataToCount are both empty strings, count all data points by state
    if colName == "" and dataToCount == "":
        countDict = data["state"].value_counts().to_dict()
        return countDict
    
    # Filter DataFrame based on colName and dataToCount
    filteredData = data[data[colName] == dataToCount]
    
    # Generate dictionary mapping each state to count of occurrences
    countDict = filteredData["state"].value_counts().to_dict()
    
    return countDict



'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    regions = data['region'].unique()  # Get unique regions
    result = {}  # Initialize result dictionary
    
    for region in regions:
        regionData = data[data['region'] == region]  # Filter data for current region
        regionDict = {}  # Initialize dictionary for current region
        
        # Count occurrences of each value in colName for current region
        for value in regionData[colName].unique():
            count = regionData[colName].value_counts()[value]
            regionDict[value] = count
        
        result[region] = regionDict  # Assign inner dictionary to outer dictionary key
    
    return result


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    hashtagCounts = {}  # Initialize dictionary to store hashtag counts
    
    # Iterate over each row in the DataFrame
    for hashtagsList in data["hashtags"]:
        if isinstance(hashtagsList, list):  # Check if the value is a list
            # Iterate over each hashtag in the list
            for hashtag in hashtagsList:
                # Update the count of the current hashtag in the dictionary
                if hashtag in hashtagCounts:
                    hashtagCounts[hashtag] += 1
                else:
                    hashtagCounts[hashtag] = 1
    
    return hashtagCounts


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def getCount(item):
    return item[1]

def mostCommonHashtags(hashtags, count):
    # Sort hashtags by count in descending order
    sortedHashtags = sorted(hashtags.items(), key=getCount, reverse=True)
    
    # Initialize dictionary to store the most common hashtags
    mostCommon = {}
    
    # Iterate through sorted hashtags to find the top count hashtags
    for hashtagFreqPair in sortedHashtags[:count]:
        hashtag, freq = hashtagFreqPair
        mostCommon[hashtag] = freq
    
    return mostCommon



'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    totalScore = 0  # Initialize total sentiment score
    totalMessages = 0  # Initialize total number of messages containing the hashtag
    
    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        if isinstance(row["hashtags"], list) and hashtag in row["hashtags"]:
            sentiment = row["sentiment"]
            if sentiment == "positive":
                totalScore += 1
            elif sentiment == "negative":
                totalScore -= 1
            
            totalMessages += 1
    
    # Calculate the sentiment score
    if totalMessages == 0:
        return 0.0
    else:
        return totalScore / totalMessages


### WEEK 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    states = list(stateCounts.keys())
    counts = list(stateCounts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(states, counts)
    plt.title(title)
    plt.xlabel('States')
    plt.ylabel('Counts')
    plt.xticks(rotation=90)
    plt.show()



'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def get_rate(item):
    return item[1]

def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    # Calculate rates for each state feature count
    stateRates = {}
    for state, featureCount in stateFeatureCounts.items():
        stateRates[state] = featureCount / stateCounts[state]
    
    # Sort states by feature rate in descending order
    sortedStates = sorted(stateRates.items(), key=get_rate, reverse=True)
    
    # Get the top n states
    topNStates = dict(sortedStates[:n])
    
    # Graph the top n states
    graphStateCounts(topNStates, title)


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    # Initialize lists for feature names, region names, and region-feature values
    featureNames = []
    regionNames = list(regionDicts.keys())
    regionFeatureValues = []

    # Construct feature list and initialize region-feature value lists
    for regionData in regionDicts.values():
        for feature, _ in regionData.items():
            if feature not in featureNames:
                featureNames.append(feature)
    
    # Iterate over regions to fill region-feature value lists
    for regionData in regionDicts.values():
        regionValues = []
        for feature in featureNames:
            if feature in regionData:
                regionValues.append(regionData[feature])
            else:
                regionValues.append(0)
        regionFeatureValues.append(regionValues)

    # Plot side-by-side bar chart
    sideBySideBarPlots(featureNames, regionNames, regionFeatureValues, title)



'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    # Generate dictionary mapping hashtags to their counts
    hashtagCounts = getHashtagRates(data)
    
    # Get the top 50 most common hashtags
    topHashtags = mostCommonHashtags(hashtagCounts, 50)
    
    # Initialize lists for hashtags, frequencies, and sentiment scores
    hashtags = list(topHashtags.keys())
    frequencies = list(topHashtags.values())
    sentimentScores = []
    
    # Calculate sentiment score for each hashtag
    for hashtag in hashtags:
        sentiment = getHashtagSentiment(data, hashtag)
        sentimentScores.append(sentiment)
    
    # Plot scatter plot
    scatterPlot(frequencies, sentimentScores, hashtags, "Hashtag Sentiment by Frequency")



#### WEEK 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()