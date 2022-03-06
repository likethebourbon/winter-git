import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.kclique import k_clique_communities
from netwulf import visualize
import dash
from dash import html
import dash_cytoscape as cyto


mlb = MultiLabelBinarizer()


df = pd.read_csv("data.csv")
# df.columns = [col.replace(" ", "_").replace(r"/_", "").lower() for col in df.columns]

# df = df.dropna(how="all", axis=1)

# id_cols = ["game", "track_title", "franchise", "year", "platform", "company"]

# tag_columns = df.columns[7:]

# df["game_and_track"] = df["game"] + " -- " + df["track_title"]

# tags = df.set_index(["game_and_track"])[tag_columns]

# tags = tags.replace(to_replace=" ", value="_", regex=True)

# tags = tags.melt(ignore_index=False).dropna(subset=["value"]).reset_index()

# tags["value"] = tags["value"].astype(str)

# sizes = tags["value"].value_counts()
# sizes.to_csv("unique_tags.csv")


def make_edgelist(df, id_cols=None):
    df.columns = [
        col.replace(" ", "_").replace(r"/_", "").lower() for col in df.columns
    ]
    df = df.dropna(how="all", axis=1)

    if id_cols is None:
        id_cols = ["game", "track_title", "franchise", "year", "platform", "company"]

    tag_columns = [col for col in df.columns if col not in id_cols + ["number_of_tags"]]

    df["game_and_track"] = df["game"] + " -- " + df["track_title"]

    tags = df.set_index(["game_and_track"])[tag_columns]

    tags = tags.replace(to_replace=" ", value="_", regex=True)

    tags = tags.melt(ignore_index=False).dropna(subset=["value"]).reset_index()

    tags = tags[tags["value"] != "--"]

    tags = tags[["game_and_track", "value"]]

    tags.columns = ["source", "target"]

    tags = tags.astype({"target": str})

    return tags.reset_index(drop=True)


edgelist = make_edgelist(df)

G = nx.from_pandas_edgelist(edgelist, "source", "target")

pagerank = pd.DataFrame.from_dict(nx.pagerank(G), orient="index").reset_index()

pagerank.columns = ["node", "value"]

pagerank = pagerank.sort_values("value", ascending=False)


tags = edgelist.groupby("source").agg(
    {"target": list},
)

sizes = tags.sum(axis=0)

tags = pd.DataFrame(
    mlb.fit_transform(tags["target"]), columns=mlb.classes_, index=tags.index
)

coocc = tags.T.dot(tags)

np.fill_diagonal(coocc.values, 0)

song_nodes = [
    # {"data": {"id": item, "label": item}}
    {"data": {"id": item.replace(" ", "_"), "label": item, "type": "song"}}
    for item in tags.index.tolist()
]

tag_nodes = [
    # {"data": {"id": item, "label": item}}
    {"data": {"id": item.replace(" ", "_"), "label": item, "type": "tag"}}
    for item in tags.columns.tolist()
]

nodes = song_nodes + tag_nodes

edges = [
    {"data": {"source": source.replace(" ", "_"), "target": target.replace(" ", "_")}}
    for source, target in edgelist.itertuples(index=False)
]

elements = nodes + edges

app = dash.Dash(__name__)
application = app.server

stylesheet = [
    {
        "selector": "node",
        "style": {
            "width": f"mapData(size, 0, {coocc.max().max()}, 10, 200)",
            "height": f"mapData(size, 0, {coocc.max().max()}, 10, 200)",
            "content": "data(label)",
            "font-size": "75px",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": f"mapData(weight, 0, {coocc.max().max()}, 1, 50)",
            "height": f"mapData(weight, 0, {coocc.max().max()}, 1, 50)",
            "line-opacity": f"mapData(weight, 0, {coocc.max().max()}, .1, .8)",
            "opacity": f"mapData(weight, 0, {coocc.max().max()}, .1, .8)",
            "curve-style": "bezier",
        },
    },
]


app.layout = html.Div(
    [
        html.H1("test"),
        cyto.Cytoscape(
            id="cytoscape-layout-9",
            elements=elements,
            style={"width": "100%", "height": "800px"},
            layout={"name": "cose"},
            stylesheet=stylesheet,
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
