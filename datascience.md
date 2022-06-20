In order to get from a dataset in a format that is easy for humans to read and write to data visualizations and algorithmically-derived metrics, we first have to transform the dataset into a format usable by computers, in a process known as "data preprocessing". The first step involves restructuring the data to be "tidy," a term coined by statistician Hadley Wickham in the paper [Tidy data](https://vita.had.co.nz/papers/tidy-data.html), which says:
    "In tidy data:
        1. Each variable forms a column.
        2. Each observation forms a row.
        3. Each type of observational unit forms a table." (section 2.3, page 4, I don't know how to cite lol)
For our dataset, each song's tags moved from separate columns in one row to separate rows in one column, and any empty cells (or "null" values) were dropped. 

The next step uses the Python library scikit-learn's [`preprocessing.MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) module to turn the tidy dataset into a binary matrix indicating which tags are present in each song: each row is a song, each tag is a separate column, and the value of each cell is 1 for each tag that exists in each song and 0 otherwise.

At this point, each row represents one song; however, we want to examine and visualize how the tags interact with each other on a global level. We aggregate the data into a co-occurrence matrix by [computing the dot product of the binary matrix with itself](https://stackoverflow.com/a/63237262). In the resulting matrix the rows and columns are tags, and the value of each cell is the number of times the (row, column) pair of tags occur in the same song. 

The co-occurrence matrix has duplicated data: the value of each (row, column) pair is also reflected in the corresponding (column, row) pair. We deduplicate the data by keeping the matrix's upper triangle with the help of the Python library [NumPy's `triu`](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) module. The matrix's [main diagonal](https://mathworld.wolfram.com/Diagonal.html) reflects the number of times each tag is used; we store those values to set the size of each tag in the network graph and then set the value of the main diagonal to 0 using [NumPy's `fill_diagonal`](https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html) module.

Now that we have the number of times each pair of tags occur together, we can make a network graph showing how the tags relate to one another. The coordinates for each tag are calculated using [scikit-learn's implementation of the t-distributed Stochastic Neighbor Embedding (t-SNE) algorithm](https://scikit-learn.org/stable/modules/manifold.html#t-sne). t-SNE is a dimensionality reduction algorithm commonly used to create visualizations of high-dimensional data. "The goal is to take a set of points in a high-dimensional space and find a faithful representation of those points in a lower-dimensional space, typically the 2D plane. The algorithm is non-linear and adapts to the underlying data, performing different transformations on different regions" ([How to Use t-SNE effectively](https://distill.pub/2016/misread-tsne/)). 

We now have all the data we need to construct the network graph: the size of each tag (from the co-occurrence matrix's main diagonal), the size of the connection between each pair of tags (from the co-occurrence matrix's upper triangle), and the coordinates for each tag (from t-SNE). The last step is to deploy the steps outlined above in a web app for easy and rapid exploration. We used Plotly's [Dash](https://dash.plotly.com/) library to build the web app, and the [Dash Cytoscape](https://dash.plotly.com/cytoscape) library to build the actual graph. Dash allowed us to specify numerous parameters through dropdown menus, sliders, and other input components, giving us granular control over the resulting graph.



List of things to cite

- [scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn)
- [Tidy data](https://vita.had.co.nz/papers/tidy-data.html)
- [numpy.triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html)
- [TSNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne)
- [How to Use t-SNE effectively](https://distill.pub/2016/misread-tsne/)
- [Dash](https://dash.plotly.com/)
- [Dash Cytoscape](https://dash.plotly.com/cytoscape)
- [Pandas](https://pandas.pydata.org/docs/index.html)


PageRank

PageRank is an algorithm for measuring the relative importance of nodes in a network, made famous as the algorithm Google used to rank search results. From the viewpoint of a person randomly visiting websites by clicking on links, the score given to each node can be thought of as the probability of visiting that specific node. While this metaphor doesn't directly translate to our network of tags, using PageRank to calculate each tag's relative importance applies to any network of nodes and edges.

PageRank was designed for directed networks--meaning a links between nodes only go in one direction. Drawing on websites again, a blog linking to a New York Times article doesn't automatically make the Times link to the blog. The dataset used for this analysis is undirected; however, [NetworkX](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html), the library used to calculate PageRank scores, converts undirected networks to directed networks by replacing each undirected edge with two directed edges, one going in each direction.

[Original PageRank paper](http://ilpubs.stanford.edu:8090/422/)


Conclusion

