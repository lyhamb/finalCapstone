""" Capstone Project #2 - Sentiment analysis of Amazon reviews. """ 

# Import libraries.
import spacy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define functions needed for analysis.
def preprocess(text):
    """
    This function preprocesses the 'text' input by tokenising the words,
    lemmantising them, formatting them to lower case and also removing
    any stop words and punctuation.
    """
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_stop
                     and not token.is_punct])

def getSubjectivity(text):
    """
    This function uses TextBlob to create a subjectivity score for
    the 'text' input.
    """
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    """
    This function uses TextBlob to create a polarity score for
    the 'text' input.
    """
    return TextBlob(text).sentiment.polarity

def getSentiment(score):
    """
    This function takes the 'score' input and assigns a sentiment based
    on it's value.
    """
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

# Import the data files to be analysed.
data = pd.read_csv('amazon_product_reviews.csv')


# Create a new column that concatenates the 'reviews.text' and 'reviews.title' columns.
data["ReviewsCombined"] = data["reviews.text"]+data["reviews.title"]


# Remove unnecessary columns from the dataset and remove missing values.
clean_data = data[["reviews.rating", "ReviewsCombined"]].dropna()


# Rename the 'reviews.rating' column.
clean_data.rename(columns={"reviews.rating":"ReviewScore"}, inplace=True)


# Load the 'en_core_web_sm' spaCy model to enable natural language processing.
nlp = spacy.load("en_core_web_sm")


# Use the 'preprocess' function to clean the data and create a new column.
clean_data["ProcessedReview"] = clean_data["ReviewsCombined"].apply(preprocess)


# Use the 'getSubjectivity' function to apply scores to the 'ProcessedReview' data.
clean_data["Subjectivity"] = clean_data["ProcessedReview"].apply(getSubjectivity)


# Use the 'getPolarity' function to apply scores to the 'ProcessedReview' data.
clean_data["Polarity"] = clean_data["ProcessedReview"].apply(getPolarity)


# Use the 'getSentiment' function to apply a sentiment to the 'Polarity' scores.
clean_data["Sentiment"] = clean_data["Polarity"].apply(getSentiment)


# Create a graph of the sentiment distribution.
ax = clean_data["Sentiment"].value_counts().sort_index() \
    .plot(kind="bar",
          title="Sentiment Distribution",
          figsize=(10, 5))
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Reviews")
plt.show()

# Print total number of reviews for each sentiment.
print(clean_data.Sentiment.value_counts())


# Create a list of tokens for each sentiment classification.
pos_words = ' '.join([w for w in clean_data['ProcessedReview'][clean_data.Sentiment=='Positive']])
neu_words = ' '.join([w for w in clean_data['ProcessedReview'][clean_data.Sentiment=='Neutral']])
neg_words = ' '.join([w for w in clean_data['ProcessedReview'][clean_data.Sentiment=='Negative']])

# Create a wordcloud for each classifications using these tokens.
pos_wordCloud = WordCloud(width=1000, height=500, random_state=5, max_font_size=125,
                          background_color='white').generate(pos_words)
neu_wordCloud = WordCloud(width=1000, height=500, random_state=5, max_font_size=125,
                          background_color='white').generate(neu_words)
neg_wordCloud = WordCloud(width=1000, height=500, random_state=5, max_font_size=125,
                          background_color='white').generate(neg_words)

# Print wordcloud graphs
fig, ax = plt.subplots(3, figsize=(10, 15))

ax[0].imshow(pos_wordCloud, interpolation='bilinear')
ax[0].set_title('Positive Words')
ax[0].axis('off')

ax[1].imshow(neu_wordCloud, interpolation='bilinear')
ax[1].set_title('Neutral Words')
ax[1].axis('off')

ax[2].imshow(neg_wordCloud, interpolation='bilinear')
ax[2].set_title('Negative Words')
ax[2].axis('off')

plt.show()


# Seperate sentiment classifications for further analysis
positive_reviews = clean_data[clean_data["Sentiment"] == "Positive"]
neutral_reviews = clean_data[clean_data["Sentiment"] == "Neutral"]
negative_reviews = clean_data[clean_data["Sentiment"] == "Negative"]

# Create a graph showing the rating score distribution by sentiment
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

# Positive reviews
axs[0].hist(positive_reviews["ReviewScore"], bins=5, color='green', edgecolor='black')
axs[0].set_title("Positive Reviews")
axs[0].set_xlabel("Rating")
axs[0].set_ylabel("Number of Reviews")
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(1))

# Neutral reviews
axs[1].hist(neutral_reviews["ReviewScore"], bins=5, color='blue', edgecolor='black')
axs[1].set_title("Neutral Reviews")
axs[1].set_xlabel("Rating")
axs[1].set_ylabel("Number of Reviews")
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(1))

# Negative reviews
axs[2].hist(negative_reviews["ReviewScore"], bins=5, color='red', edgecolor='black')
axs[2].set_title("Negative Reviews")
axs[2].set_xlabel("Rating")
axs[2].set_ylabel("Number of Reviews")
axs[2].xaxis.set_major_locator(ticker.MultipleLocator(1))

plt.tight_layout()
plt.show()


# Create a scatter graph to show the relationship between polarity and subjectivity.
plt.figure(figsize=(10,10))
plt.scatter(clean_data["Polarity"], clean_data["Subjectivity"])
plt.title("Polarity vs Subjectivity")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.show()


# Compare similiarity of a sample of reviews
sample1 = 2000
sample2 = 5000
doc1 = nlp(clean_data['ReviewsCombined'][sample1])
doc2 = nlp(clean_data['ReviewsCombined'][sample2])
similarity_score = doc1.similarity(doc2)
print(f"The similarity score between review index {sample1} and {sample2} is {similarity_score}")
