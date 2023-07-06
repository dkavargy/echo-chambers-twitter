import sys
import re
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import emoji

stopwords_qatar = ['Fifaworldcup', 'Fifaworldcup2022', 'Fifaworldcup22', 'FIFAWorldCup', 'WorldCup', 'WorldCup2022', 'Qatar2022', 'FIFAWorldCupQatar2022', 'WorldCupQatar2022', 'WorldCupQatar', 'Qatar']
stopwords_mahsa = ['MahsaAmini', 'IranRevolution', 'OpIran', 'IranRevolution2022', 'Mahsa_Amini']
stopwords_ukraine = ['Ukraine', 'Russia'] #, 'StopPutin', 'StopRussia', 'StopRussianAggression']

to_eliminate = ['#', 'FIFAWorldCup', 'WorldCup', 'Qatar2022', 'FIFAWorldCupQatar2022', \
                '\'s', '\'', '\"', '\\n', 'Cup', 'World', 'Qatar', 'QT', 'FIFA'\
                '\\u2003', '\\u202f', '\\U000e006e', '\\U000e0065' \
                '\\U000e0062', '\\U000e0067', '\\U000e007f', '\u3000',
                'U000e0062', 'U000e0065', 'xa0', 'u2003']

def flat_list_of_lists(lol):
    return [item for sublist in lol for item in sublist]

def generate_wordcloud(text, my_stopwords):
    stopwords = set(STOPWORDS).union(my_stopwords)
    wordcloud = WordCloud(stopwords=stopwords,
                          max_font_size=50,
                          max_words=100,
                          background_color="white").generate(text)
    return wordcloud

def get_emoji_free_text(text):
    return emoji.replace_emoji(text, replace='')

def clean_text(text):

    # remove emojis
    text = get_emoji_free_text(text)

    # remove urls
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"amp\S+", "", text)

    # remove non-utf8 characters
    text = bytes(text, 'utf-8').decode('utf-8','ignore')
    
    # remove words
    for word in to_eliminate:
        text = text.replace(word, '')

    # remove single characters
    text = ' '.join( [w for w in text.split() if len(w)>1] )

    return text

def get_hashtags(text):
    return re.findall(r'\b#\w+', text)


def main():
    if len(sys.argv) != 3:
        print ('Usage:', sys.argv[0], 'collection_name network_type(quoted/replied)')
        sys.exit()
    else:
        collection_name = sys.argv[1]
        network_type = sys.argv[2]
        nodes_file = 'networks/nodes-'+network_type+'_'+collection_name+'.csv'

        # read tweets per community
        df = pd.read_csv(nodes_file)
        df = df[df['community_id'] != -1]

        # select only the 10 largest communities
        df = df[df['community_id'] < 15]

        # get all tweets per community
        df_comm2tweets = defaultdict(lambda: "")
        for index, row in df.iterrows():
            c = row['community_id']
            tweets = row['tweets']
            df_comm2tweets[c] += tweets
        
        # all words wordclouds
        for c,all_tweets in df_comm2tweets.items():
            all_tweets = clean_text(all_tweets)
            if collection_name == 'qatar_data':
                my_stopwords = stopwords_qatar
            elif collection_name == 'mahsa_amini_data':
                my_stopwords = stopwords_mahsa
            elif collection_name == 'ukraine_war_data':
                my_stopwords = stopwords_ukraine
            wordcloud = generate_wordcloud(all_tweets, my_stopwords)
            plt.imshow(wordcloud)#, interpolation='bilinear')
            plt.axis('off')
            plt.savefig('communities/wordclouds_all_words/wordcloud-'+network_type+'_'+collection_name+'_comm'+str(c)+'.png')
        
        # hashtags wordclouds
        for c,all_tweets in df_comm2tweets.items():
            hashtags = get_hashtags(all_tweets)
            str_hashtags = ' '.join(hashtags).replace('#', '')
            if collection_name == 'qatar_data':
                my_stopwords = stopwords_qatar
            elif collection_name == 'mahsa_amini_data':
                my_stopwords = stopwords_mahsa
            elif collection_name == 'ukraine_war_data':
                my_stopwords = stopwords_ukraine
            wordcloud = generate_wordcloud(str_hashtags, my_stopwords)
            plt.imshow(wordcloud)#, interpolation='bilinear')
            plt.axis('off')
            plt.savefig('communities/wordclouds_hashtags/wordcloud-'+network_type+'_'+collection_name+'_comm'+str(c)+'.png')

if __name__ == "__main__":
    main()
        
