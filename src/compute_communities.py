import sys
import matplotlib.pyplot as plt
import pandas as pd
import igraph as ig
import numpy as np
import seaborn as sns
from collections import defaultdict


def read_graph_from_edgelist(edges_file):
    df = pd.read_csv(edges_file)
    g = ig.Graph.TupleList(df.itertuples(index=False), directed=True)
    return g

def read_node_attributes(g, nodes_file):
    df = pd.read_csv(nodes_file)
    return df

def write_community_sizes(communities, out_name):
    sizes = sorted([len(c) for c in communities], reverse=True)
    o = open(out_name, 'w')
    o.write('community_id,size\n')
    for i in range(len(sizes)):
        o.write(str(i)+','+str(sizes[i])+'\n')
    o.close()


def plot_stats(collection_name, network_type, communities):
    out_name = 'communities/'+collection_name+'-'+network_type
    write_community_sizes(communities, out_name+'_sizes.csv')

def main():
    if len(sys.argv) != 3:
        print ('Usage:', sys.argv[0], 'collection_name network_type(quoted/replied)')
        sys.exit()
    else:
        collection_name = sys.argv[1]
        network_type = sys.argv[2]
        nodes_file = 'networks/nodes-'+network_type+'_'+collection_name+'.csv'
        edges_file = 'networks/edges-'+network_type+'_'+collection_name+'.csv'

        ## read edgelist
        g = read_graph_from_edgelist(edges_file)

        ## preprocess the graph
        # keep only the largest connected component
        components = g.connected_components(mode='weak')
        g = components.giant()

        ## community detection
        # calculate communities using infomap
        communities = g.community_infomap()
        communities_list = []
        for c in communities:
            comm = [g.vs[x]['name'] for x in c]
            communities_list.append(comm)
        communities_list.sort(key=len, reverse=True)
        plot_stats(collection_name, network_type, communities_list)

        ## add community id to each user
        # read node attributes
        df_nodes = read_node_attributes(g, nodes_file)
        
        # add community id for each node if we don't have communities yet
        if 'community_id' not in df_nodes:
            df_nodes['community_id'] = -1
            for i in range(len(communities_list)):
                for node_id in communities_list[i]:
                    df_nodes.loc[df_nodes['user_id'] == node_id, 'community_id'] = i
            df_nodes.to_csv(nodes_file, index=False)

        df_nodes[['user_id', 'community_id']].to_csv('communities/node2community-'+network_type+'_'+collection_name+'.csv', index=False)
        

if __name__ == "__main__":
    main()