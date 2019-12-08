import csv
import pandas as pd

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from google.cloud import storage


def get_sentiment(lyrics):
    # Instantiates a client
    client = language.LanguageServiceClient()

    document = types.Document(
        content=lyrics,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the lyrics
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    print(sentiment)

    return sentiment.score

# Function to open a .csv file and add sentiment column
def run():
    threads = []
    with open('data/lyrics.csv','r') as csvinput:
        reader = pd.read_csv(csvinput, nrows=100)

        for index, row in reader.iterrows():
            row = row.copy()
            reader.loc[index, 'sentiment'] = get_sentiment(row['lyrics'])

        reader.to_csv(r'data/labeled_lyrics.csv', header=True)

run()

# May work for iterating through a range  of rows
    # reader = pd.read_csv(csvinput, skiprows=[10], nrows=3)

    # for index, row in reader.iterrows():
    #     row = row.copy()
    #     reader.loc[index, 'sentiment'] = get_sentiment(row['lyrics'])
    # reader.to_csv('my_csv.csv', mode='a', header=False)
