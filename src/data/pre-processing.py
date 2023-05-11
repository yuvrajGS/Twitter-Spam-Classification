import pandas as pd
import string
from nltk.stem import PorterStemmer
from collections import Counter
ps = PorterStemmer()

dfTweets = pd.read_csv('data/raw/tweets.csv', encoding='utf-8', low_memory=False)
dfTweetsSpam = pd.read_csv('data/raw/tweets_spam.csv', encoding='utf-8', low_memory=False)
dfTweetsUsers = pd.read_csv('data/raw/users.csv', encoding='utf-8', low_memory=False)
dfTweetsSpamUsers = pd.read_csv('data/raw/users_spam.csv', encoding='utf-8', low_memory=False)

#get the number of tweets each user in dfTweetsSpamUsers has in dfTweetsSpam
dfTweetsSpamUsers['numTweets'] = dfTweetsSpamUsers['id'].apply(lambda x: dfTweetsSpam[dfTweetsSpam['user_id'] == x].shape[0])
#get the number of tweets each user in dfTweetsUsers has in dfTweets
dfTweetsUsers['numTweets'] = dfTweetsUsers['id'].apply(lambda x: dfTweets[dfTweets['user_id'] == x].shape[0])
#remove users in dfTweetsUsers that have 0 numTweets
dfTweetsUsers = dfTweetsUsers[dfTweetsUsers['numTweets'] != 0]
#remove users in dfTweetsSpamUsers that have 0 numTweets
dfTweetsSpamUsers = dfTweetsSpamUsers[dfTweetsSpamUsers['numTweets'] != 0]

#remove tweets in dfTweetsSpam that do not have a user in dfTweetsSpamUsers
dfTweetsSpam = dfTweetsSpam[dfTweetsSpam['user_id'].isin(dfTweetsSpamUsers['id'])]
#remove tweets in dfTweets that do not have a user in dfTweetsUsers
dfTweets = dfTweets[dfTweets['user_id'].isin(dfTweetsUsers['id'])]

# get users in dfTweets with more than 1000 tweets
users_with_more_than_1000_tweets = dfTweetsUsers[dfTweetsUsers['numTweets'] > 1000]['id']

# limit the number of tweets to 1000 for each user in dfTweets that has more than 1000 tweets
for user_id in users_with_more_than_1000_tweets:
    user_tweets = dfTweets[dfTweets['user_id'] == user_id]
    if user_tweets.shape[0] > 1000:
        dfTweets.drop(user_tweets.sort_values(by='created_at', ascending=False).iloc[1000:].index, inplace=True)


#label spam tweets & users
dfTweetsSpam['label'] = 1
dfTweetsSpamUsers['label'] = 1
#label non-spam tweets & users
dfTweets['label'] = 0
dfTweetsUsers['label'] = 0

#combine the spam and non-spam tweets into one dataframe
dfTweets = pd.concat([dfTweets, dfTweetsSpam], ignore_index=True)
#combine the spam and non-spam users into one dataframe
dfTweetsUsers = pd.concat([dfTweetsUsers, dfTweetsSpamUsers], ignore_index=True)


#fix in_reply_to_status_id in dfTweets: If the value is not 0, replace it with 1 & rename column
dfTweets['is_reply'] = dfTweets['in_reply_to_status_id'].apply(lambda x: 1 if x != 0 else 0)
dfTweets = dfTweets.drop('in_reply_to_status_id', axis=1)
#fix place in dfTweets: If the value is NaN replace with 0, else 1
dfTweets['place'] = dfTweets['place'].apply(lambda x: 0 if pd.isna(x) else 1)

userFeatures=['label','id','name', 'screen_name','followers_count', 'friends_count',
              'favourites_count','listed_count','location','default_profile',
              'geo_enabled','verified','description']
tweetFeatures = ['label','user_id','text','is_reply','place', 'retweet_count', 'favorite_count', 'possibly_sensitive', 'num_hashtags', 'num_mentions', 'num_urls',]

#feature selection
dfTweetsUsers = dfTweetsUsers[userFeatures]
dfTweets = dfTweets[tweetFeatures]
#rename columns to capitalize first letter
dfTweetsUsers.columns = [x.capitalize() for x in dfTweetsUsers.columns]
dfTweets.columns = [x.capitalize() for x in dfTweets.columns]


# split text column into words
words = dfTweets['Text'].str.split(expand=True).stack()

# count word frequencies with Counter object
wordFreq = Counter(word.translate(str.maketrans('', '', string.punctuation)).strip("…").lower() 
                   for word in words 
                   if "http" not in word and "@" not in word and "#" not in word and not word.startswith("www"))

# filter out words that appear less than 100 times with dictionary comprehension
wordFreq = {word:freq for word, freq in wordFreq.items() if freq >= 100}


def standardize_word(word):
    #remove punctuation, make lowercase & stem
    return ps.stem(word.translate(str.maketrans('', '', string.punctuation)).strip("…").lower())

#count the number of times each word appears in the text of each tweet for each user
freq_df = pd.DataFrame(columns=['Id'] + list(wordFreq.keys()))

for index, row in dfTweetsUsers.iterrows():
    userTweets = dfTweets[dfTweets['User_id'] == row['Id']]
    words = userTweets['Text'].str.split(expand=True).stack()
    wordFreqTweet = Counter(standardize_word(word) for word in words if "http" not in word and "@" not in word and "#" not in word and not word.startswith("www"))
    freq_df.loc[index] = [row['Id']] + [wordFreqTweet[word] for word in wordFreq.keys()]

#merge
dfTweetsUsers = pd.merge(dfTweetsUsers, freq_df, on='Id')

#for each user, get the total number of words in all their tweets
dfTweetsUsers['totalWords'] = dfTweetsUsers[wordFreq.keys()].sum(axis=1)
#divide each word count by the total number of words in all the user's tweets
for word in wordFreq.keys():
    dfTweetsUsers[word] = dfTweetsUsers[word].div(dfTweetsUsers['totalWords'])

dfTweets.fillna(0, inplace=True)
#fill the Default_profile,Verified and Geo_enabled columns with 0 if NaN
dfTweetsUsers['Default_profile'].fillna(0, inplace=True)
dfTweetsUsers['Geo_enabled'].fillna(0, inplace=True)
dfTweetsUsers['Verified'].fillna(0, inplace=True)

dfTweetsUsers['Total_retweet_count'] = dfTweetsUsers['Id'].apply(lambda x: dfTweets[dfTweets['User_id'] == x]['Retweet_count'].sum())
dfTweetsUsers['Total_favorite_count'] = dfTweetsUsers['Id'].apply(lambda x: dfTweets[dfTweets['User_id'] == x]['Favorite_count'].sum())
dfTweetsUsers['Total_mentions'] = dfTweetsUsers['Id'].apply(lambda x: dfTweets[dfTweets['User_id'] == x]['Num_mentions'].sum())
dfTweetsUsers['Total_urls'] = dfTweetsUsers['Id'].apply(lambda x: dfTweets[dfTweets['User_id'] == x]['Num_urls'].sum())

dfTweetsUsers.to_csv('data/processed/users.csv', index=False)

print("Finished")