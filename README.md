# lyrics-classifier
Classify a song as happy or sad based on their lyrics

This project is composed of 3 different parts:
1. A script to label a dataset of songs with lyrics using Google Cloud's Sentiment Analysis Natural Language API
2. A Django API that trains and creates a model based on that labeled dataset and interacts with the front-facing webpage to make API calls to the musixmatch API being used to do the song search and lyrics search
3. A webpage with HTML, CSS and Javascript files to render search bars for song name and artist name and rendering mood results

Model Results: 
- Unfortunately, the model currently does not render the best results
- The accuracy has been quite low and this can be due to dataset being 1imited to 100 songs at the moment... this is due to issues with labeling all songs in original dataset and data entries with empty spaces that are currently not being dealth with