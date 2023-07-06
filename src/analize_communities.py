import re
import sys
import seaborn as sns
import emoji
from collections import Counter, defaultdict
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np

# https://spacy.io/universe/project/spacy-textblob
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

stopwords_qatar = ['Fifaworldcup', 'Fifaworldcup2022', 'Fifaworldcup22', 'FIFAWorldCup', 'WorldCup', 'WorldCup2022', 'Qatar2022', 'FIFAWorldCupQatar2022', 'WorldCupQatar2022', 'WorldCupQatar', 'Qatar']
stopwords_mahsa = ['MahsaAmini', 'IranRevolution', 'OpIran', 'IranRevolution2022', 'Mahsa_Amini']
stopwords_ukraine = ['Ukraine', 'Russia'] #, 'StopPutin', 'StopRussia', 'StopRussianAggression']
stopwords_baltimore = ['blacklivesmatter', 'freddiegray', 'baltimoreuprising', 'baltimore', 'ferguson', 'baltimoreriots', 'gt', 'lt', 'via']
stopwords_guns = ['gunsense', 'gunviolence', 'guncontrol']
stopwords_vaccination = ['covid19', 'covidvaccine', 'coronavirus', 'covid19vaccine', 'vaccination', 'covid', 'vaccine', 'vaccinated', 'people', 'vaccineswork', 'vaccines', 'vaccinessavelives', 'VaccinesWork', 'VaccinesSaveLives']


to_eliminate = ['#', 'FIFAWorldCup', 'WorldCup', 'Qatar2022', 'FIFAWorldCupQatar2022', \
                '\'s', '\'', '\"', '\\n', 'Cup', 'World', 'Qatar', 'QT', 'FIFA', 'RT', 'Quote'\
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


# set number of communities we want to take (based on the community sizes plots)
def get_num_top_comms(collection_name, network_type):
    if collection_name == 'qatar_data':
        if network_type == 'quoted':
            return 5
        elif network_type == 'replied':
            return 4
    elif collection_name == 'ukraine_war_data':
        if network_type == 'quoted':
            return 3
        elif network_type == 'replied':
            return 4
    elif collection_name == 'mahsa_amini_data':
        if network_type == 'quoted':
            return 4
        elif network_type == 'replied':
            return 5
    elif collection_name == 'baltimore_riots':
        if network_type == 'quoted':
            return 25
        elif network_type == 'replied':
            return 25
    elif collection_name == 'gun_control':
        if network_type == 'quoted':
            return 10
        elif network_type == 'replied':
            return 10
    elif collection_name == 'vaccination':
        if network_type == 'quoted':
            return 10
        elif network_type == 'replied':
            return 10
    elif collection_name == 'vacc_final':
        if network_type == 'quoted':
            return 10
        elif network_type == 'replied':
            return 25                

def plot_community_sizes(df_comms, out_file):
    size_distribution = Counter(df_comms['community_id'].values)
    plt.bar(size_distribution.keys(), size_distribution.values())
    plt.savefig(out_file)

def plot_hashtags_wordcloud_per_community(df_top_comms, my_stopwords, collection_name, network_type):
    com_indices = df_top_comms['community_id'].unique().tolist()
    for c in com_indices:
        tweets_list = df_top_comms[df_top_comms['community_id'] == c]['tweets'].values.tolist()
        tweets = ""
        for l in tweets_list:
            tweets += l

        hashtags = ' '.join(get_hashtags(tweets))
        wordcloud = generate_wordcloud(hashtags, my_stopwords)
        plt.imshow(wordcloud)#, interpolation='bilinear')
        plt.axis('off')
        # plt.show()
        plt.savefig('new_wordclouds/hashtags-'+network_type+'_'+collection_name+'_comm'+str(c)+'.png')


def plot_words_wordcloud_per_community(df_top_comms, my_stopwords, collection_name, network_type):
    com_indices = df_top_comms['community_id'].unique().tolist()
    for c in com_indices:
        tweets_list = df_top_comms[df_top_comms['community_id'] == c]['tweets'].values.tolist()
        tweets = ""
        for l in tweets_list:
            tweets += l

        tweets = clean_text(tweets)
        wordcloud = generate_wordcloud(tweets, my_stopwords)
        plt.imshow(wordcloud)#, interpolation='bilinear')
        plt.axis('off')
        # plt.show()
        plt.savefig('new_wordclouds/words-'+network_type+'_'+collection_name+'_comm'+str(c)+'.png')


def plot_assessments_wordcloud_per_community(assessments, out_file):
    out_file = out_file.replace('.png', '_')
    for c, words in assessments.items():
        words_str = ' '.join(words)
        wordcloud = generate_wordcloud(words_str, [])
        plt.imshow(wordcloud)#, interpolation='bilinear')
        plt.axis('off')
        out_file_c = out_file+str(c)+'.png'
        plt.savefig(out_file_c)

def find_sentiment(df_top_comms, nlp):
    com_indices = df_top_comms['community_id'].unique().tolist()
    polarity = defaultdict(lambda: [])
    subjectivity = defaultdict(lambda: [])
    assessments = defaultdict(lambda: [])
    for c in com_indices:
        tweets_list = df_top_comms[df_top_comms['community_id'] == c]['tweets'].values.tolist()
        for tweet in tweets_list:
            nlp.max_length = len(tweet) + 100
            doc = nlp(tweet)
            polarity[c].append(doc._.blob.polarity)
            subjectivity[c].append(doc._.subjectivity)

            assessments_tmp = flat_list_of_lists([x[0] for x in doc._.blob.sentiment_assessments.assessments])
            assessments[c].extend(assessments_tmp)

    return polarity, subjectivity, assessments


def plot_violin(d, out_file, name):
    maxsize = max([len(a) for a in d.values()])
    data_pad = {k:np.pad(v, pad_width=(0,maxsize-len(v),), mode='constant', constant_values=np.nan) for k,v in d.items()}
    df = pd.DataFrame(data_pad)
    fig, ax = plt.subplots()
    sns.violinplot(data=df)
    ax.set_xlabel('community')
    ax.set_ylabel(name)
    plt.savefig(out_file)


def main():
    if len(sys.argv) != 3:
        print ('Usage:', sys.argv[0], 'collection_name network_type(quoted/replied)')
        sys.exit()
    else:
        collection_name = sys.argv[1]
        network_type = sys.argv[2]
        nodes_file = 'networks/nodes-'+network_type+'_'+collection_name+'.csv'
        communities_file = 'communities_hierarchical/node2community-'+network_type+'_'+collection_name+'.csv'
        
        df_nodes = pd.read_csv(nodes_file)
        df_comms = pd.read_csv(communities_file)

        plot_size_file = 'communities_hierarchical/community_size_distribution-'+network_type+'_'+collection_name+'.png'
        # plot_community_sizes(df_comms, plot_size_file)

        # we keep only the top n largest communities of the network
        n_top_comms = get_num_top_comms(collection_name, network_type)
        df_top_comms = df_comms[df_comms['community_id'] < n_top_comms]
        df_top_comms = df_top_comms.merge(df_nodes[['user_id', 'tweets']], on='user_id', how='inner')

        # plot wordclouds
        if collection_name == 'qatar_data':
            my_stopwords = stopwords_qatar
        elif collection_name == 'mahsa_amini_data':
            my_stopwords = stopwords_mahsa
        elif collection_name == 'ukraine_war_data':
            my_stopwords = stopwords_ukraine
        elif collection_name == 'baltimore_riots':
            my_stopwords = stopwords_baltimore
        elif collection_name == 'gun_control':
            my_stopwords = stopwords_guns
        elif collection_name == 'vaccination' or collection_name == 'vacc_final':
            my_stopwords = stopwords_vaccination
        else:
            my_stopwords = []

        plot_hashtags_wordcloud_per_community(df_top_comms, my_stopwords, collection_name, network_type)
        plot_words_wordcloud_per_community(df_top_comms, my_stopwords, collection_name, network_type)

        ## sentiment analysis
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('spacytextblob')
        d_polarity, d_subjectivity, assessments = find_sentiment(df_top_comms, nlp)

        plot_wordcloud_file = 'sentiment/wordcloud/wordcloud-'+network_type+'_'+collection_name+'.png'
        plot_assessments_wordcloud_per_community(assessments, plot_wordcloud_file)

        sns.set_theme(style="darkgrid")

        plot_polarity_file = 'sentiment/polarity-'+network_type+'_'+collection_name+'.png'
        plot_violin(d_polarity, plot_polarity_file, 'polarity')

        plot_subjectivity_file = 'sentiment/subjectivity-'+network_type+'_'+collection_name+'.png'
        plot_violin(d_subjectivity, plot_subjectivity_file, 'subjectivity')



if __name__ == "__main__":
    main()