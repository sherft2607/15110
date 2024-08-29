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
    return


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    return


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    return


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    return


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    return


### WEEK 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


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
    """print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()"""

    ## Uncomment these for Week 3 ##
    """print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()"""