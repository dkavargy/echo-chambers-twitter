import sys
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import igraph as ig
sns.set()
import random

# https://sharkcoder.com/data-visualization/mpl-bidirectional

pro_black_hashtags = ['PoliceBrutality', 'RacismInAmerica', 'BlackLivesMatter']
pro_blue_hashtags = ['BlueLivesMatter', 'UniteBlue', 'AllLivesMatter']

def is_tag_in_text(tags, text):
    for tag in tags:
        if tag.lower() in text.lower():
            return 1
    return 0

def get_probabilities(community2tweets):
    d_probs = defaultdict(lambda: defaultdict(lambda: []))
    for c,tweet_list in community2tweets.items():
        for user_tweets in tweet_list:
            ok_black = is_tag_in_text(pro_black_hashtags, user_tweets)
            ok_blue = is_tag_in_text(pro_blue_hashtags, user_tweets)
            d_probs[c]['black'].append(ok_black)
            d_probs[c]['blue'].append(ok_blue)
        d_probs[c]['black'] = np.mean(d_probs[c]['black'])
        d_probs[c]['blue'] = np.mean(d_probs[c]['blue'])
    return d_probs

def plot_black_blue(d_probs, out_file):
    data = []
    for c in d_probs.keys():
        problack = d_probs[c]['black']
        problue = d_probs[c]['blue']
        data.append([c, problack, problue])
    
    df = pd.DataFrame(data, columns=['community', 'pro-black', 'pro-blue'])
    # df = df.set_index('community')
    df = df.sort_values(by=['pro-blue'])
    ylabels = df['community'].values
    df = df.reset_index()

    font_color = '#525252'
    hfont = {'fontname':'Calibri'}
    facecolor = '#eaeaf2'
    color_red = '#000000'
    color_blue = '#0000FF'
    index = df.index
    column0 = df['pro-black']
    column1 = df['pro-blue']
    title0 = 'Pro-black'
    title1 = 'Pro-blue'

    fig, axes = plt.subplots(figsize=(10,5), facecolor=facecolor, ncols=2, sharey=True)
    fig.tight_layout()

    axes[0].barh(index, column0, align='center', color=color_red, zorder=10)
    axes[0].set_title(title0, fontsize=18, pad=15, color=color_red, **hfont)
    axes[1].barh(index, column1, align='center', color=color_blue, zorder=10)
    axes[1].set_title(title1, fontsize=18, pad=15, color=color_blue, **hfont) 

    # If you have positive numbers and want to invert the x-axis of the left plot
    axes[0].invert_xaxis() 

    axes[0].set(yticks=df.index, yticklabels=ylabels)
    axes[0].yaxis.tick_left()
    axes[0].tick_params(axis='y', colors='white') # tick color

    axes[1].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    for label in (axes[0].get_xticklabels() + axes[0].get_yticklabels()):
        label.set(fontsize=10, color=font_color, **hfont)
    for label in (axes[1].get_xticklabels() + axes[1].get_yticklabels()):
        label.set(fontsize=10, color=font_color, **hfont)

    # Add title and axis names
    plt.ylabel('Community ID', labelpad=320)
    
    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)
    plt.savefig(out_file)
    plt.clf()

def read_graph_from_edgelist(edges_file):
    df = pd.read_csv(edges_file)
    g = ig.Graph.TupleList(df.itertuples(index=False), directed=False, edge_attrs='weight')
    return g

def plot_stats(d_stats, network_type):
    for k in d_stats.keys():
        d_tmp = d_stats[k]

        d_tmp = dict(sorted(d_tmp.items(), key=lambda x:x[1]))

        names = list(d_tmp.keys())
        values = list(d_tmp.values())

        plt.bar(range(len(d_tmp)), values, tick_label=names)
        plt.title(k)
        plt.savefig('echo_chambers/stats-'+network_type+'_'+k+'_baltimore_riots.png')
        plt.clf()

def network_analysis(g, df_comms, network_type):

    d_stats = defaultdict(lambda: defaultdict(float))

    for i in range(25):
        df_tmp = df_comms[df_comms['community_id'] == i]
        users = df_tmp['user_id'].values

        # extract subgraphs
        users_nodes = [g.vs.find(name=x).index for x in users]
        g_c = g.induced_subgraph(users_nodes)

        d_stats['avg_path_length'][i] = np.mean(g_c.distances())
        # d_stats['avg_local_clustering'][i] = g_c.transitivity_avglocal_undirected()
        # d_stats['avg_local_clustering_weighted'][i] = g_c.transitivity_avglocal_undirected(weights=g_c.es['weight'])
        # d_stats['global_clustering'][i] = g_c.transitivity_undirected()
        # d_stats['density'][i] = g_c.density()
        # d_stats['avg_degree'][i] = np.mean(g_c.degree()) / np.max(g_c.degree())
        d_stats['avg_closeness'][i] = np.mean(g_c.closeness())
        # d_stats['avg_betweenness'][i] = np.mean(g_c.betweenness())
        # d_stats['avg_eigenvector'][i] = np.mean(g_c.eigenvector_centrality(directed=False))
        
    plot_stats(d_stats, network_type)

def get_influencers(g, nodes):
    degree = g.degree(nodes)
    d_degree = {nodes[x]: degree[x] for x in range(len(degree))}
    sorted_degrees = sorted(d_degree.items(), key=lambda x:x[1], reverse=True)
    top_15pct = int(len(sorted_degrees) * 0.15)
    sorted_degrees = sorted_degrees[:top_15pct]
    return [x for (x, y) in sorted_degrees]

def get_distance(g, nodes, influencers):
    distances = []
    for node in nodes:
        for inf in influencers:
            if node != inf:
                distance = g.shortest_paths(node, inf)[0][0]
                distances.append(distance)
    return np.mean(distances)

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

def plot_shortest_distance_to_influencers(g, nodes1, nodes2, influencers1, influencers2):
    distance_com1_influencers1 = get_distance(g, nodes1, influencers1)
    distance_com1_influencers2 = get_distance(g, nodes1, influencers2)
    distance_com2_influencers2 = get_distance(g, nodes2, influencers2)
    distance_com2_influencers1 = get_distance(g, nodes2, influencers1)

    data = {
        'distance to its own influencers' : [distance_com1_influencers1, distance_com2_influencers2],
        'distance to the opposite influencers' : [distance_com1_influencers2, distance_com2_influencers1]
    }

    fig, ax = plt.subplots()
    plt.xticks([0, 1], labels=['community1', 'community9'])
    bar_plot(ax, data, total_width=.8, single_width=.9)
    plt.savefig('echo_chambers/baltimore-quoted_controversy.png')
    plt.clf()

def random_walk_iter(g, u, own_influencers, other_influencers): 
    current = u
    while True:
        current = random.choice(g.successors(current))
        if current in own_influencers:
            return 1
        if current in other_influencers:
            return -1

def rwc(g, users, own_influencers, other_influencers):
    rwcs = []
    for u in users:
        res = random_walk_iter(g, u, own_influencers, other_influencers)
        rwcs.append(res)
    pxx = rwcs.count(1) / len(rwcs)
    pxy = rwcs.count(-1) / len(rwcs)
    return pxx, pxy


def compute_rawc(g, users_nodes1, users_nodes2, influencers1, influencers2):
    
    p11, p12 = rwc(g, users_nodes1, influencers1, influencers2)
    p22, p21 = rwc(g, users_nodes2, influencers2, influencers1)

    return p11*p22 - p12*p21

def plot_boxplot(data):
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    
    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True,
                    notch ='True', vert = 0)
    
    colors = ['#0000FF', '#FFFF00']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")
    
    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 2)
    
    # changing color and linewidth of
    # medians
    """ for median in bp['medians']:
        median.set(color ='red',
                linewidth = 3) """
    
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D',
                color ='#e7298a',
                alpha = 0.5)
        
    # x-axis labels
    ax.set_yticklabels(['RAWC\n (communities\n 1 and 9)', 'RAWC\n (random case)'])
    
    # Adding title
    plt.title("Random Walk Controversy (1000 repetitions)")
    
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim(-0.2, 1)
        
    # show plot
    plt.savefig('echo_chambers/baltimore-quoted_controversy_rawc.png')
    

def controversy_analysis(g, df_comms, com1, com2, network_type):
    df_com1 = df_comms[df_comms['community_id'] == com1]
    df_com2 = df_comms[df_comms['community_id'] == com2]
    users1 = df_com1['user_id'].values
    users2 = df_com2['user_id'].values

    users_nodes1 = [g.vs.find(name=x).index for x in users1]
    users_nodes2 = [g.vs.find(name=x).index for x in users2]

    influencers1 = get_influencers(g, users_nodes1)
    influencers2 = get_influencers(g, users_nodes2)

    plot_shortest_distance_to_influencers(g, users_nodes1, users_nodes2, influencers1, influencers2)

    rawc_scores = []
    for i in range(1000):    
        rawc_score = compute_rawc(g, users_nodes1, users_nodes2, influencers1, influencers2)
        rawc_scores.append(rawc_score)

    random_rawc_scores = []
    for i in range(1000):   
        random_users_nodes1 = random.sample(g.vs.indices, len(users_nodes1))
        random_users_nodes2 = random.sample(g.vs.indices, len(users_nodes2))
        random_influencers1 = random.sample(random_users_nodes1, len(influencers1))
        random_influencers2 = random.sample(random_users_nodes2, len(influencers2))
        random_rawc_score = compute_rawc(g, random_users_nodes1, random_users_nodes2, random_influencers1, random_influencers2)
        random_rawc_scores.append(random_rawc_score)

    data = [rawc_scores, random_rawc_scores]
    plot_boxplot(data)

    print("RAWC:", np.mean(rawc_scores), np.std(rawc_scores))
    print("RAWC_random:", np.mean(random_rawc_scores), np.std(random_rawc_scores))

def main():
    if len(sys.argv) != 2:
        print ('Usage:', sys.argv[0], 'network_type(quoted/replied)')
        sys.exit()
    else:
        collection_name = 'baltimore_riots'
        network_type = sys.argv[1]
        nodes_file = 'networks/nodes-'+network_type+'_'+collection_name+'.csv'
        edges_file = 'networks/edges-'+network_type+'_'+collection_name+'.csv'
        communities_file = 'communities_hierarchical/node2community-'+network_type+'_'+collection_name+'.csv'

        df_nodes = pd.read_csv(nodes_file)
        df_comms = pd.read_csv(communities_file)

        counter_communities = Counter(df_comms['community_id'].values)
        df_comms = df_comms[df_comms['community_id'] <= 25]

        community2tweets = defaultdict(lambda: [])
        for index,row in df_comms.iterrows():
            c = row['community_id']
            id_user = row['user_id']
            tweets_list = df_nodes[df_nodes['user_id'] == id_user]['tweets']
            community2tweets[c].extend(tweets_list)

        d_probs = get_probabilities(community2tweets)

        out_file = 'echo_chambers/plot-black-blue_'+network_type+'.png'
        plot_black_blue(d_probs, out_file)
        
        # network analysis
        ## read edgelist
        g = read_graph_from_edgelist(edges_file)

        ## preprocess the graph
        # keep only the largest connected component
        components = g.connected_components(mode='weak')
        g = components.giant()

        network_analysis(g, df_comms, network_type)

        com1 = 1
        com2 = 9
        controversy_analysis(g, df_comms, com1, com2, network_type)

if __name__ == "__main__":
    main()