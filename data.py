import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import itertools
import networkx as nx
from netwulf import visualize

import dash
from dash import html, Output, Input, dcc
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go

mlb = MultiLabelBinarizer()
pca = PCA(n_components=2)
cyto.load_extra_layouts()
# tsne = TSNE(
#     n_components=2, perplexity=5, learning_rate="auto", random_state=0, init="pca"
# )

# df = pd.read_csv("data.csv")

df = pd.read_csv("data_with_metatags.csv")

# df.columns = [col.replace(" ", "_").replace(r"/_", "").lower() for col in df.columns]

# df = df.dropna(how="all", axis=1)

# id_cols = ["game", "track_title", "franchise", "year", "platform", "company"]

# tag_columns = df.columns[7:]

# df["game_and_track"] = df["game"] + " -- " + df["track_title"]

# tags = df.set_index(["game_and_track"])[tag_columns]

# tags = tags.replace(to_replace=" ", value="_", regex=True)

# tags = tags.melt(ignore_index=False).dropna(subset=["value"]).reset_index()

# tags = tags[tags["value"] != "--"]


def make_edgelist(df, id_cols=None):
    df.columns = [
        col.replace(" ", "_").replace(r"/_", "").lower() for col in df.columns
    ]
    df = df.dropna(how="all", axis=1)

    if id_cols is None:
        id_cols = ["game", "track_title", "franchise", "year", "platform", "company"]
        # id_cols = ["game", "track_title"]

    # df["year"] = df["year"] // 10 * 10

    tag_columns = [col for col in df.columns if col not in id_cols + ["number_of_tags"]]

    # df["game_and_track"] = f'{df["game"]} -- {df["track_title"]}'

    df["game_and_track"] = df["game"] + "--" + df["track_title"]

    tags = df.set_index(["game_and_track"])[tag_columns]

    tags = tags.replace(to_replace=" ", value="_", regex=True)

    tags = (
        tags.melt(ignore_index=False)
        .dropna(subset=["value"])
        .reset_index()
        .astype({"value": str})
    )

    from collections import Counter

    c = Counter(tags["value"].tolist())

    counts = pd.DataFrame(c.items())

    # counts = tags["value"].value_counts(sort=True)

    # tags["value"] = tags["value"].astype(str)

    counts.to_csv("counts.csv")

    # tags = tags[tags["value"].isin(counts[counts > 1].index)]

    tags = tags[tags["value"] != "--"]

    tags = tags[["game_and_track", "value"]]

    # tags.columns = ["source", "target"]

    # return tags.reset_index(drop=True)

    return tags


# tags = make_edgelist(df)

# tags["value"] = tags["value"].astype(str)

# tags["value"] = tags["value"].apply(lambda x: x.split(":")[1:] if ":" in x else x)

# tags = tags.explode("value", ignore_index=True)

# tags = tags.groupby("game_and_track").agg({"value": "unique"})

# tags = pd.DataFrame(
#     mlb.fit_transform(tags["value"]), columns=mlb.classes_, index=tags.index
# )

# coocc = tags.T.dot(tags)


# stylesheet = [
#     {
#         "selector": "node",
#         "style": {
#             "width": f"mapData(size, 0, {coocc.max().max()}, 10, 200)",
#             "height": f"mapData(size, 0, {coocc.max().max()}, 10, 200)",
#             "content": "data(label)",
#             "font-size": "75px",
#         },
#     },
#     {
#         "selector": "edge",
#         "style": {
#             "width": f"mapData(weight, 0, {coocc.max().max()}, 1, 50)",
#             "height": f"mapData(weight, 0, {coocc.max().max()}, 1, 50)",
#             "line-opacity": f"mapData(weight, 0, {coocc.max().max()}, .1, .8)",
#             "opacity": f"mapData(weight, 0, {coocc.max().max()}, .1, .8)",
#             "curve-style": "bezier",
#         },
#     },
# ]

# max_store = 122

# stylesheet = [
#     {
#         "selector": "node",
#         "style": {
#             "width": f"mapData(size, 0, {max_store}, 10, 200)",
#             "height": f"mapData(size, 0, {max_store}, 10, 200)",
#             "content": "data(label)",
#             "font-size": "75px",
#         },
#     },
#     {
#         "selector": "edge",
#         "style": {
#             "width": f"mapData(weight, 0, {max_store}, 1, 50)",
#             "height": f"mapData(weight, 0, {max_store}, 1, 50)",
#             "line-opacity": f"mapData(weight, 0, {max_store}, .1, .8)",
#             "opacity": f"mapData(weight, 0, {max_store}, .1, .8)",
#             "curve-style": "bezier",
#         },
#     },
# ]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Store(id="max-store", data=122),
                        html.P("Controls"),
                        dbc.Select(
                            id="christmas-dropdown",
                            options=[
                                {"label": "All songs", "value": "all"},
                                {"label": "Christmas songs only", "value": "christmas"},
                                {
                                    "label": "No Christmas songs",
                                    "value": "no_christmas",
                                },
                                {
                                    "label": "Nintendo",
                                    "value": "nintendo",
                                },
                                {
                                    "label": "No Nintendo",
                                    "value": "no_nintendo",
                                },
                            ],
                            value="all",
                        ),
                        html.Br(),
                        html.P("Distance multiplier:"),
                        dcc.Slider(
                            min=10,
                            max=110,
                            step=1,
                            value=50,
                            marks={
                                i: {"label": str(i)} for i in [10, 30, 50, 70, 90, 110]
                            },
                            id="zoom-slider",
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        html.P("t-SNE Perplexity:"),
                        dcc.Slider(
                            min=5,
                            max=105,
                            step=1,
                            value=5,
                            marks={
                                i: {"label": str(i)} for i in [5, 25, 45, 65, 85, 105]
                            },
                            id="perplexity-slider",
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        dbc.Checkbox(
                            id="highlight-edges", label="Highlight edges", value=False
                        ),
                        html.Br(),
                        dbc.Button("Export image", id="export-button"),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        cyto.Cytoscape(
                            id="network-graph",
                            layout={"name": "preset"},
                            # layout={
                            #     "name": "cose",
                            #     "idealEdgeLength": 200,
                            #     "nodeOverlap": 5,
                            #     "refresh": 20,
                            #     "fit": False,
                            #     "padding": 30,
                            #     "randomize": False,
                            #     "componentSpacing": 1000,
                            #     "nodeRepulsion": 400000,
                            #     "edgeElasticity": 100,
                            #     "nestingFactor": 5,
                            #     "gravity": 80,
                            #     "numIter": 1000,
                            #     "initialTemp": 200,
                            #     "coolingFactor": 0.95,
                            #     "minTemp": 1.0,
                            #     "animate": True,
                            # },
                            style={"width": "100%", "height": "800px"},
                            # stylesheet=stylesheet,
                            # elements=data,
                            # responsive=True,
                        )
                    ],
                    # width="auto",
                ),
            ]
        ),
        dbc.Row(dbc.Col(dcc.Graph(id="barchart"))),
        dbc.Row(dbc.Col(html.Div(id="stats-div"))),
    ]
)

# stylesheet = [{"selector": "node", "style": {"content": "data(label)"}}]


@app.callback(
    Output("network-graph", "stylesheet"),
    Input("network-graph", "mouseoverNodeData"),
    Input("max-store", "data"),
    Input("highlight-edges", "value"),
)
def color_children(edgeData, max_store, highlight_edges):
    # if max_store is None:
    # max_store = 122

    stylesheet = [
        {
            "selector": "node",
            "style": {
                "width": f"mapData(size, 0, {max_store}, 10, 200)",
                "height": f"mapData(size, 0, {max_store}, 10, 200)",
                "content": "data(label)",
                "font-size": "75px",
                "opacity": 0.75,
                "text-background-color": "#fff",
                "text-background-opacity": 1,
            },
        },
        {
            "selector": "edge",
            "style": {
                "line-color": "#194d19",
                "background-color": "#194d19",
                "width": f"mapData(weight, 0, {max_store}, 1, 50)",
                "height": f"mapData(weight, 0, {max_store}, 1, 50)",
                "line-opacity": f"mapData(weight, 0, {max_store}, .1, .25)",
                "opacity": f"mapData(weight, 0, {max_store}, .1, .25)",
                "curve-style": "bezier",
            },
        },
    ]
    if edgeData is None:
        return stylesheet

    if highlight_edges:

        # if "s" in edgeData["source"]:
        #     val = edgeData["source"].split("s")[0]
        # else:
        val = edgeData["label"]

        children_style = [
            {
                "selector": 'edge[source = "{}"]'.format(val),
                "style": {"line-color": "blue"},
            },
            {
                "selector": 'edge[target = "{}"]'.format(val),
                "style": {"line-color": "blue"},
            },
        ]

        return stylesheet + children_style
    return stylesheet


@app.callback(
    [
        Output("network-graph", "elements"),
        Output("max-store", "data"),
        Output("stats-div", "children"),
        Output("barchart", "figure"),
    ],
    [
        Input("christmas-dropdown", "value"),
        Input("zoom-slider", "value"),
        Input("perplexity-slider", "value"),
    ],
)
def make_graph_data(christmas, zoom, perplexity, df=df):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        random_state=0,
        init="pca",
    )

    if christmas == "nintendo":
        dff = df[df["company"] == "Nintendo"]
    elif christmas == "no_nintendo":
        dff = df[df["company"] != "Nintendo"]
    else:
        dff = df

    tags = make_edgelist(dff)

    tags.to_csv("tags.csv")

    # G = nx.from_pandas_edgelist(tags, "game_and_track", "value")

    def make_tags_df(tags):

        tags["value"] = tags["value"].astype(str)

        tags["value"] = tags["value"].apply(
            lambda x: x.split(":")[1:] if ":" in x else x
        )

        tags = tags.explode("value", ignore_index=True)

        tags = tags.groupby("game_and_track").agg({"value": "unique"})

        tags = pd.DataFrame(
            mlb.fit_transform(tags["value"]), columns=mlb.classes_, index=tags.index
        )

        return tags

    tags = make_tags_df(tags)

    tags.sum(axis=0).to_csv("counts_2.csv")

    tags = tags.drop(
        columns=[
            "percussion",
            "modal",
            "percussion_drum",
            "drums",
            "2",
            "3",
            "4",
            "6",
            # "9",
            "12",
        ]
    )

    tags.to_csv("tags_dropped.csv")

    xmas_coocc = (
        tags[tags["Christmas"] == 1]
        .drop(columns=["Christmas"])
        .T.dot(tags[tags["Christmas"] == 1].drop(columns=["Christmas"]))
    )
    no_xmas_coocc = (
        tags[tags["Christmas"] == 0]
        .drop(columns=["Christmas"])
        .T.dot(tags[tags["Christmas"] == 0].drop(columns=["Christmas"]))
    )

    # if christmas == "christmas":
    #     tags = tags[tags["Christmas"] == 1]
    #     tags = tags.drop(columns=["Christmas"])
    # elif christmas == "no_christmas":
    #     tags = tags[tags["Christmas"] == 0]

    nintendo_tags = make_tags_df(make_edgelist(df[df["company"] == "Nintendo"])).drop(
        columns=[
            "percussion",
            "modal",
            "percussion_drum",
            "drums",
            "2",
            "3",
            "4",
            # "6",
            "12",
        ]
    )
    no_nintendo_tags = make_tags_df(
        make_edgelist(df[df["company"] != "Nintendo"])
    ).drop(
        columns=[
            "percussion",
            "modal",
            "percussion_drum",
            "drums",
            "2",
            "3",
            "4",
            "6",
            "12",
        ]
    )

    nintendo_tags_coocc = nintendo_tags.T.dot(nintendo_tags)
    no_nintendo_tags_coocc = no_nintendo_tags.T.dot(no_nintendo_tags)

    coocc = tags.T.dot(tags)

    def make_pagerank_df(coocc):

        G = nx.from_pandas_adjacency(coocc)

        pagerank = pd.DataFrame.from_dict(nx.pagerank(G), orient="index").reset_index()

        pagerank.columns = ["node", "value"]

        pagerank = pagerank.sort_values("value", ascending=False)

        pagerank["value"] = pagerank["value"].round(decimals=5)

        return pagerank

    pagerank = make_pagerank_df(coocc)

    nintendo_pagerank = make_pagerank_df(nintendo_tags_coocc)
    nintendo_pagerank.columns = ["node", "nintendo_value"]
    no_nintendo_pagerank = make_pagerank_df(no_nintendo_tags_coocc)
    no_nintendo_pagerank.columns = ["node", "no_nintendo_value"]

    G_xmas = nx.from_pandas_adjacency(xmas_coocc)
    G_no_xmas = nx.from_pandas_adjacency(no_xmas_coocc)

    pagerank_xmas = pd.DataFrame.from_dict(
        nx.pagerank(G_xmas), orient="index"
    ).reset_index()

    pagerank_xmas.columns = ["node", "xmas_value"]

    pagerank_xmas = pagerank_xmas.sort_values("xmas_value", ascending=False)

    pagerank_no_xmas = pd.DataFrame.from_dict(
        nx.pagerank(G_no_xmas), orient="index"
    ).reset_index()

    pagerank_no_xmas.columns = ["node", "no_xmas_value"]

    pagerank_no_xmas = pagerank_no_xmas.sort_values("no_xmas_value", ascending=False)

    # pagerank_diff = pagerank_xmas.join(
    #     pagerank_no_xmas.set_index("node"), on="node", how="outer"
    # )
    pagerank_diff = nintendo_pagerank.join(
        no_nintendo_pagerank.set_index("node"), on="node", how="outer"
    )

    # pagerank_diff["diff"] = pagerank_diff["xmas_value"] - pagerank_diff["no_xmas_value"]
    pagerank_diff["diff"] = (
        pagerank_diff["nintendo_value"] - pagerank_diff["no_nintendo_value"]
    ).round(decimals=5)

    # pagerank_diff = pagerank_diff[pagerank_diff["diff"].abs() > 0.00]

    print(coocc.shape)

    coords = pd.DataFrame(
        tsne.fit_transform(coocc), index=tsne.feature_names_in_, columns=["x", "y"]
    )

    sizes = pd.DataFrame(index=coocc.index, data=np.diag(coocc), columns=["size"])

    sizes.to_csv("unique_tags.csv")

    node_data = pd.DataFrame(sizes).join(coords)

    np.fill_diagonal(coocc.values, 0)

    coocc = (
        coocc.where(np.triu(np.ones(coocc.shape)).astype(bool)).fillna(0).astype(int)
    )

    edge_list = (
        coocc.reset_index()
        .replace({0: np.nan})
        .melt(id_vars=["index"])
        .dropna()
        .reset_index(drop=True)
        .astype({"value": "int"})
    )

    edge_list.columns = ["source", "target", "weight"]

    nodes = [
        {
            "data": {
                "id": tag.Index,
                "label": tag.Index,
                "size": tag.size,
            },
            "position": {
                "x": tag.x * zoom,
                "y": tag.y * zoom,
            },
        }
        for tag in node_data.itertuples()
    ]

    edges_lists = [
        [
            {
                "data": {
                    "source": row[0],
                    "target": tag[0],
                    "label": tag[0],
                    "weight": tag[1],
                }
            }
            for tag in row[1].items()
            if tag[1] > 0
        ]
        for row in coocc.iterrows()
    ]

    edges = list(itertools.chain(*edges_lists))

    bar_df = pagerank_diff.copy()
    bar_df["no_nintendo_value"] *= -1

    bar_df["diff"] = (
        bar_df["diff"]
        .fillna(bar_df["nintendo_value"])
        .fillna(bar_df["no_nintendo_value"])
    )

    # bar_df = bar_df[bar_df["diff"].abs() > 0.007]

    fig = go.Figure()

    color_dict = {"nintendo_value": "#e4000f", "no_nintendo_value": "lightslategray"}
    for col in ["nintendo_value", "no_nintendo_value"]:
        fig.add_trace(
            go.Bar(
                x=bar_df[col].values,
                y=bar_df["node"],
                orientation="h",
                name=col.replace("_", " ").title(),
                customdata=bar_df[col],
                hovertemplate="%{y}: %{customdata}",
                marker_color=color_dict[col],
            )
        )

    fig.add_trace(
        go.Scatter(
            x=bar_df["diff"].values,
            y=bar_df["node"],
            mode="markers",
            name="Difference",
            marker_symbol="line-ns",
            marker_line_color="midnightblue",
            marker_color="lightskyblue",
            marker_line_width=2,
            marker_size=15,
        )
    )

    fig.update_layout(
        barmode="relative",
        height=600,
        width=600,
        yaxis_autorange="reversed",
        bargap=0.5,
        legend_orientation="v",
        legend_x=1,
        legend_y=0,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
    )

    return (
        nodes + edges,
        coocc.max().max(),
        dbc.Table.from_dataframe(
            pagerank_diff,
            # pagerank_diff.sort_values("diff", ascending=False),
            striped=True,
            bordered=True,
            hover=True,
        ),
        fig,
    )


@app.callback(
    Output("network-graph", "generateImage"), Input("export-button", "n_clicks")
)
def export_image(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        return {"type": "png", "action": "download"}
    raise dash.exceptions.PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)
