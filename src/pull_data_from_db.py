import sys
import json
import pandas as pd
from collections import defaultdict
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def connect_to_db():
    uri = "mongodb+srv://rodrigo:d7anhyfnDY76FIKe@atlascluster.ixhrypl.mongodb.net/?retryWrites=true&w=majority"
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        return client
    except Exception as e:
        print(e)
        sys.exit()

def main():
    if len(sys.argv) != 2:
        print ('Usage:', sys.argv[0], 'collection_name')
        sys.exit()
    else:
        collection_name = sys.argv[1]
        client = connect_to_db()
        d_user = defaultdict(lambda: defaultdict())
        user2quotedtweets = defaultdict(lambda: [])
        user2userlinks = defaultdict(lambda: [])
        with client:
            db = client[collection_name]
            
            if collection_name == 'ukraine_war_data':
                collection_name = 'ukraine_war_data '
            collection = db[collection_name]

            tweet2user = dict()
            for tweet in collection.find():
                user_id = tweet['user']['id']
                tweet_id = tweet['id_str']

                # tweet to user mapping
                tweet2user[tweet_id] = user_id

                # user info
                if user_id not in d_user.keys():
                    d_user[user_id]['username'] = tweet['user']['username']
                    d_user[user_id]['public_metrics'] = tweet['user']['public_metrics']

                # add tweet text to user
                if 'tweets' not in d_user[user_id].keys():
                    d_user[user_id]['tweets'] = [tweet['full_text']]
                else:
                    d_user[user_id]['tweets'].append(tweet['full_text'])

                # connect two users if quoted or replied to
                if 'referenced_tweets' in tweet.keys():
                    if tweet['referenced_tweets'][0]['type'] == 'quoted':
                        user2quotedtweets[user_id].append(tweet['referenced_tweets'][0]['id'])
                    elif tweet['referenced_tweets'][0]['type'] == 'replied_to':
                        replied_user_id = tweet['in_reply_to_user_id']
                        if user_id < replied_user_id:
                            user2userlinks[user_id].append(replied_user_id)
                        else:
                            user2userlinks[replied_user_id].append(user_id)
                    else:
                        print('Reference tweet unknown', tweet_id)

        # connect users from quoted tweets
        d_edges_quoted = defaultdict(lambda: defaultdict(lambda: 0))
        for user_id,quoted_list in user2quotedtweets.items():
            for tweet_id in quoted_list:
                try:
                    quoted_user = tweet2user[tweet_id]
                    if user_id < quoted_user:
                        d_edges_quoted[user_id][quoted_user] += 1
                    else:
                        d_edges_quoted[quoted_user][user_id] += 1
                    # print('quoted tweet recovered!')
                except:
                    continue

        # connect users from replied tweets
        d_edges_replied = defaultdict(lambda: defaultdict(lambda: 0))
        for user_id,replied_users in user2userlinks.items():
            if user_id in d_user.keys():
                for replied_user in replied_users:
                    if replied_user != user_id and replied_user in d_user.keys():
                        if user_id < replied_user:
                            d_edges_replied[user_id][replied_user] += 1
                        else:
                            d_edges_replied[replied_user][user_id] += 1
                        # print('quoted tweet recovered!') """



        # remove isolated nodes from quoted
        nodes_in_edges_quoted = set()
        for n1 in d_edges_quoted.keys():
            nodes_in_edges_quoted.add(n1)
            for n2 in d_edges_quoted[n1].keys():
                nodes_in_edges_quoted.add(n2)
        d_user_quoted = {k: v for k,v in d_user.items() if k in nodes_in_edges_quoted}

        # remove isolated nodes from replied
        nodes_in_edges_replied = set()
        for n1 in d_edges_replied.keys():
            nodes_in_edges_replied.add(n1)
            for n2 in d_edges_replied[n1].keys():
                nodes_in_edges_replied.add(n2)
        d_user_replied = {k: v for k,v in d_user.items() if k in nodes_in_edges_replied}

        # write quoted data to file
        collection_name = collection_name.replace(' ', '')
        edges_file = 'networks/edges-quoted_'+collection_name+'.csv'
        with open(edges_file, 'w') as out_edges:
            out_edges.write('source,target,weight\n')
            for n1 in d_edges_quoted.keys():
                for n2 in d_edges_quoted[n1].keys():
                    out_edges.write(n1+','+n2+','+str(d_edges_quoted[n1][n2])+'\n')
        out_edges.close()

        nodes_file = 'networks/nodes-quoted_'+collection_name+'.csv'
        df_nodes = pd.DataFrame.from_dict(d_user_quoted, orient='index').reset_index().rename(columns={'index': 'user_id'})
        df_nodes.to_csv(nodes_file, index=False)

        # write replied data to file
        edges_file = 'networks/edges-replied_'+collection_name+'.csv'
        with open(edges_file, 'w') as out_edges:
            out_edges.write('source,target,weight\n')
            for n1 in d_edges_replied.keys():
                for n2 in d_edges_replied[n1].keys():
                    out_edges.write(n1+','+n2+','+str(d_edges_replied[n1][n2])+'\n')
        out_edges.close()

        nodes_file = 'networks/nodes-replied_'+collection_name+'.csv'
        df_nodes = pd.DataFrame.from_dict(d_user_replied, orient='index').reset_index().rename(columns={'index': 'user_id'})
        df_nodes.to_csv(nodes_file, index=False)





if __name__ == "__main__":
    main()