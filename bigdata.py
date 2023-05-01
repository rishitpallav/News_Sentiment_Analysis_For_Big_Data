import string
from pyspark.sql.functions import regexp_replace, udf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName('bigdata').getOrCreate()

# Read the data from the csv file
df = spark.read.csv('articles1.csv', header=True, inferSchema=True)

# Drop the columns that are not required
df = df.drop('c0', 'publication', 'author', 'year', 'month', 'url')

# drop the rows that have null values in any column
df = df.dropna()

# remove stopwords from the text column using nltk stopwords
stop_words = set(stopwords.words('english'))


@udf(returnType=StringType())
def remove_stopwords_udf(text):
    word_tokens = word_tokenize(text)
    filtered_words = [
        word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# apply the udf to the text column
df = df.withColumn('content', remove_stopwords_udf(df['content']))

# remove punctuations from the text column
# create a udf to remove punctuations from the text column


def remove_punctuations(text):
    return regexp_replace(text, '[^\w\s]', '')


# Apply the function to the 'content' column
df = df.withColumn('content', remove_punctuations(df['content']))

# create a pyspark udf to convert the text column to lowercase


@udf(returnType=StringType())
def lower_case_udf(text):
    return text.lower()


# apply the udf to the text column
df = df.withColumn('content', lower_case_udf(df['content']))


# remove numbers from the text column
@udf(returnType=StringType())
def remove_numbers_udf(sentence):
    return ''.join([i for i in sentence if not i.isdigit()])
# apply the udf to the text column


df = df.withColumn('content', remove_numbers_udf(df['content']))

# remove extra spaces from the text column


@udf(returnType=StringType())
def remove_extra_spaces_udf(sentence):
    return ' '.join(sentence.split())


# apply the udf to the text column
df = df.withColumn('content', remove_extra_spaces_udf(df['content']))

# print the first 5 rows of the dataframe
df.show(5)

# import vader sentiment analyzer
analyser = SentimentIntensityAnalyzer()

# perform sentiment analysis on the text column


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score


# create a udf to apply the sentiment analysis function to the text column
sentiment_analyzer = udf(lambda x: sentiment_analyzer_scores(x))

# apply the udf to the text column
df = df.withColumn('sentiment', sentiment_analyzer(df['content']))

# print the first 5 rows of the dataframe
df.show(5)
