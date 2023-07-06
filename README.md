# Master Degree Thesis
## Echo Chamber Detection
### Kavargyris Dimitrios - Christos

The following project was designed as part of his **thesis** for the Aristotle University of Thessaloniki in the Department of Informatics. We present an analysis for findingThe following project was designed as part of his thesis for the Aristotle University of Thessaloniki in the Department of Informatics. We present an analysis for finding echo chambers in widespread social media, using state-of-the-art datasets, which are available for use as created by the author for this purpose. in widespread social media, using state-of-the-art datasets, which are available for use as created by the author for this purpose.

#### Folder
In the src folder you can find:

* **analize_communities.py**: presenting the partition in smaller communities.
* **analyze_baltimore_riots.py**: presenting results of baltimore_riots community.
* **analyze_vacc_final.py**: presenting results of vacc_final community.
* **compute_communities.py**: calculated communites and colored them in the Cytoscape files.
* **compute_communities_hierarchical.py**: created communities using the multilevel Louvain algorithm.
* **pull_data_from_db.py**: parse data from mongodb cluster
* **wordcloud_communities**: added wordclouds of all words in tweets

In the folder of this repo you can find, 
```
communities/ : nodes and edges from all the networks
communities_hierachical/ : multilevel Louvain algorithm partitioning
```



