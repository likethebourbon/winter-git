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
import dash_daq as daq

import plotly.express as px
import plotly.graph_objects as go

from utils import make_edgelist, make_tags_df, filter_data, import_data

mlb = MultiLabelBinarizer()
pca = PCA(n_components=2)
cyto.load_extra_layouts()
# tsne = TSNE(
#     n_components=2, perplexity=5, learning_rate="auto", random_state=0, init="pca"
# )

# df = pd.read_csv("data_220308.csv")

df = import_data("data_220308.csv")

# Define values for dropdowns

platform_list = sorted(
    list(set(sum(df["platform"].dropna().str.split(":").tolist(), []))),
    key=str.casefold,
)

company_list = sorted(df["company"].dropna().unique(), key=str.casefold)

franchise_list = sorted(df["franchise"].dropna().unique(), key=str.casefold)

MIN_YEAR = int(df["year"].dropna().min())
MAX_YEAR = int(df["year"].dropna().max())


edgelist = make_edgelist(df)

tags_list = make_tags_df(make_edgelist(df), metatags=[], mlb=mlb).columns.tolist()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

application = app.server

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Store(id="max-store", data=122),
                        html.P("Controls"),
                        # dbc.Select(
                        #     id="christmas-dropdown",
                        #     options=[
                        #         {"label": "All songs", "value": "all"},
                        #         {"label": "Christmas songs only", "value": "christmas"},
                        #         {
                        #             "label": "No Christmas songs",
                        #             "value": "no_christmas",
                        #         },
                        #         {
                        #             "label": "Nintendo",
                        #             "value": "nintendo",
                        #         },
                        #         {
                        #             "label": "No Nintendo",
                        #             "value": "no_nintendo",
                        #         },
                        #         {
                        #             "label": "80s",
                        #             "value": "80s",
                        #         },
                        #         {
                        #             "label": "3/4 time",
                        #             "value": "3",
                        #         },
                        #     ],
                        #     value="all",
                        # ),
                        dcc.Dropdown(
                            id="platform-dropdown",
                            options=[{"label": p, "value": p} for p in platform_list],
                            multi=True,
                            placeholder="Select platform",
                        ),
                        dcc.Dropdown(
                            id="company-dropdown",
                            options=[{"label": p, "value": p} for p in company_list],
                            multi=True,
                            placeholder="Select publisher",
                        ),
                        dcc.Dropdown(
                            id="franchise-dropdown",
                            options=[{"label": p, "value": p} for p in franchise_list],
                            multi=True,
                            placeholder="Select franchise",
                        ),
                        html.Br(),
                        html.P("Game years"),
                        dcc.RangeSlider(
                            min=MIN_YEAR,
                            max=MAX_YEAR,
                            step=1,
                            value=[MIN_YEAR, MAX_YEAR],
                            id="year-slider",
                            marks={i: str(i) for i in [1990, 2000, 2010, 2020]},
                            tooltip={"placement": "top"},
                        ),
                        html.Br(),
                        daq.ColorPicker(
                            id="edge-color",
                            label="Edge Color",
                            value={"hex": "#421f89"},
                        ),
                        html.Br(),
                        html.P("Metatags to use"),
                        dcc.Dropdown(
                            options=[
                                {"label": c, "value": c}
                                for c in sorted(
                                    set(
                                        col.split(":")[1]
                                        for col in edgelist.loc[
                                            edgelist["value"].str.contains(":"), "value"
                                        ].unique()
                                    )
                                )
                                if c not in ["majorish", "minorish"]
                            ],
                            value=[
                                c
                                for c in sorted(
                                    list(
                                        set(
                                            col.split(":")[1]
                                            for col in edgelist.loc[
                                                edgelist["value"].str.contains(":"),
                                                "value",
                                            ].unique()
                                        )
                                    )
                                )
                                if c not in ["majorish", "minorish"]
                            ],
                            multi=True,
                            id="metatags-dropdown",
                        ),
                        html.Br(),
                        html.P("Required tags"),
                        dcc.Dropdown(
                            id="required-tags",
                            options=[{"label": t, "value": t} for t in tags_list],
                            multi=True,
                        ),
                        dbc.Checkbox(
                            id="remove-required-tags-checkbox",
                            label="Remove required tags from graph",
                            value=True,
                        ),
                        html.Br(),
                        html.P("Tags to exclude"),
                        dcc.Dropdown(
                            id="excluded-tags",
                            options=[{"label": t, "value": t} for t in tags_list],
                            multi=True,
                        ),
                        html.Br(),
                        html.P("Distance multiplier"),
                        dcc.Slider(
                            min=10,
                            max=210,
                            step=1,
                            value=50,
                            marks={
                                i: {"label": str(i)}
                                for i in [10, 50, 90, 130, 170, 210]
                            },
                            id="zoom-slider",
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        html.Br(),
                        html.P("t-SNE Perplexity"),
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
                        # dbc.Checkbox(
                        #     id="use-metatags", label="Use Metatags", value=False
                        # ),
                        # html.Br(),
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
    Input("edge-color", "value"),
)
def color_children(edgeData, max_store, highlight_edges, edge_color):

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
                "line-color": edge_color.get("hex", "#421f89"),
                "background-color": edge_color.get("hex", "#421f89"),
                "width": f"mapData(weight, 0, {max_store}, 1, 50)",
                "height": f"mapData(weight, 0, {max_store}, 1, 50)",
                "line-opacity": f"mapData(weight, 0, {max_store}, .1, .5)",
                "opacity": f"mapData(weight, 0, {max_store}, .1, .5)",
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
        # Output("stats-div", "children"),
        # Output("barchart", "figure"),
    ],
    [
        Input("platform-dropdown", "value"),
        Input("company-dropdown", "value"),
        Input("franchise-dropdown", "value"),
        Input("year-slider", "value"),
        Input("zoom-slider", "value"),
        Input("perplexity-slider", "value"),
        Input("metatags-dropdown", "value"),
        Input("required-tags", "value"),
        Input("remove-required-tags-checkbox", "value"),
        Input("excluded-tags", "value"),
    ],
)
def make_graph_data(
    platform,
    company,
    franchise,
    year,
    zoom,
    perplexity,
    metatags,
    required_tags,
    remove_required_tags,
    excluded_tags,
    df=df,
):

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        random_state=0,
        init="pca",
    )

    df = filter_data(df, platform, company, franchise, year)

    tags = make_edgelist(df)

    tags = make_tags_df(tags, metatags, mlb)

    if required_tags is not None and required_tags:
        for tag in required_tags:
            tags = tags[tags[tag] == 1]
        if remove_required_tags:
            tags = tags.drop(columns=required_tags)

    if excluded_tags is not None and excluded_tags:
        for tag in excluded_tags:
            tags = tags[tags[tag] == 0]

    print(tags)

    print(tags.sum(axis=0).sort_values())

    coocc = tags.T.dot(tags)

    # def make_pagerank_df(coocc):

    #     G = nx.from_pandas_adjacency(coocc)

    #     pagerank = pd.DataFrame.from_dict(nx.pagerank(G), orient="index").reset_index()

    #     pagerank.columns = ["node", "value"]

    #     pagerank = pagerank.sort_values("value", ascending=False)

    #     pagerank["value"] = pagerank["value"].round(decimals=5)

    #     return pagerank

    # pagerank = make_pagerank_df(coocc)

    coords = pd.DataFrame(
        tsne.fit_transform(coocc), index=tsne.feature_names_in_, columns=["x", "y"]
    )

    sizes = pd.DataFrame(index=coocc.index, data=np.diag(coocc), columns=["size"])

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

    return (
        nodes + edges,
        coocc.max().max(),
    )


# @app.callback(
#     [
#         Output("network-graph", "elements"),
#         Output("max-store", "data"),
#         Output("stats-div", "children"),
#         Output("barchart", "figure"),
#     ],
#     [
#         Input("christmas-dropdown", "value"),
#         Input("zoom-slider", "value"),
#         Input("perplexity-slider", "value"),
#         Input("metatags-dropdown", "value"),
#     ],
# )
# def make_graph_data(christmas, zoom, perplexity, metatags_dropdown, df=df):
#     tsne = TSNE(
#         n_components=2,
#         perplexity=perplexity,
#         learning_rate="auto",
#         random_state=0,
#         init="pca",
#     )

#     if christmas == "nintendo":
#         dff = df[df["company"] == "Nintendo"]
#     elif christmas == "no_nintendo":
#         dff = df[df["company"] != "Nintendo"]
#     else:
#         dff = df

#     tags = make_edgelist(dff)

#     tags.to_csv("tags.csv")

#     # G = nx.from_pandas_edgelist(tags, "game_and_track", "value")

#     tags = make_tags_df(tags, metatags_dropdown, mlb)

#     # tags.sum(axis=0).to_csv("tag_counts.csv")

#     # tags = tags.drop(
#     #     columns=[
#     #         # "percussion",
#     #         # "modal",
#     #         # "percussion_drum",
#     #         # "drums",
#     #         # "2",
#     #         # "3",
#     #         # "4",
#     #         # "6",
#     #         # "9",
#     #         # "12",
#     #     ]
#     # )

#     # tags.to_csv("tags_dropped.csv")

#     xmas_coocc = (
#         tags[tags["Christmas"] == 1]
#         .drop(columns=["Christmas"])
#         .T.dot(tags[tags["Christmas"] == 1].drop(columns=["Christmas"]))
#     )
#     no_xmas_coocc = (
#         tags[tags["Christmas"] == 0]
#         .drop(columns=["Christmas"])
#         .T.dot(tags[tags["Christmas"] == 0].drop(columns=["Christmas"]))
#     )

#     if christmas == "christmas":
#         tags = tags[tags["Christmas"] == 1]
#         tags = tags.drop(columns=["Christmas"])
#     elif christmas == "no_christmas":
#         tags = tags[tags["Christmas"] == 0]
#     elif christmas == "80s":
#         tags = tags[tags["80s"] == 1]
#         tags = tags.drop(columns=["80s"])
#     elif christmas == "3":
#         tags = tags[tags["3"] == 1]
#         tags = tags.drop(columns=["3"])
#     else:
#         tags = tags

#     nintendo_tags = make_tags_df(make_edgelist(df[df["company"] == "Nintendo"]), metatags_dropdown, mlb)
#     nintendo_tags = nintendo_tags[
#         [
#             col
#             for col in nintendo_tags.columns
#             if col
#             not in [
#                 "other_minorish",
#                 "dorian",
#                 "aeolian",
#             ]
#         ]
#     ]

#     # .drop(
#     #     columns=[
#     #         "other_minorish",
#     #         # "percussion",
#     #         # "modal",
#     #         # "percussion_drum",
#     #         # "drums",
#     #         # "2",
#     #         # "3",
#     #         # "4",
#     #         # "6",
#     #         # "12",
#     #     ]
#     # )
#     no_nintendo_tags = make_tags_df(make_edgelist(df[df["company"] != "Nintendo"]), metatags_dropdown, mlb)
#     no_nintendo_tags = no_nintendo_tags[
#         [
#             col
#             for col in no_nintendo_tags.columns
#             if col
#             not in [
#                 "other_minorish",
#                 "dorian",
#                 "aeolian",
#             ]
#         ]
#     ]
#     # .drop(
#     #     columns=[
#     #         "other_minorish",
#     #         # "percussion",
#     #         # "modal",
#     #         # "percussion_drum",
#     #         # "drums",
#     #         # "2",
#     #         # "3",
#     #         # "4",
#     #         # "6",
#     #         # "12",
#     #     ]
#     # )

#     nintendo_tags_coocc = nintendo_tags.T.dot(nintendo_tags)
#     no_nintendo_tags_coocc = no_nintendo_tags.T.dot(no_nintendo_tags)

#     coocc = tags.T.dot(tags)

#     def make_pagerank_df(coocc):

#         G = nx.from_pandas_adjacency(coocc)

#         pagerank = pd.DataFrame.from_dict(nx.pagerank(G), orient="index").reset_index()

#         pagerank.columns = ["node", "value"]

#         pagerank = pagerank.sort_values("value", ascending=False)

#         pagerank["value"] = pagerank["value"].round(decimals=5)

#         return pagerank

#     pagerank = make_pagerank_df(coocc)

#     nintendo_pagerank = make_pagerank_df(nintendo_tags_coocc)
#     nintendo_pagerank.columns = ["node", "nintendo_value"]
#     no_nintendo_pagerank = make_pagerank_df(no_nintendo_tags_coocc)
#     no_nintendo_pagerank.columns = ["node", "no_nintendo_value"]

#     G_xmas = nx.from_pandas_adjacency(xmas_coocc)
#     G_no_xmas = nx.from_pandas_adjacency(no_xmas_coocc)

#     pagerank_xmas = pd.DataFrame.from_dict(
#         nx.pagerank(G_xmas), orient="index"
#     ).reset_index()

#     pagerank_xmas.columns = ["node", "xmas_value"]

#     pagerank_xmas = pagerank_xmas.sort_values("xmas_value", ascending=False)

#     pagerank_no_xmas = pd.DataFrame.from_dict(
#         nx.pagerank(G_no_xmas), orient="index"
#     ).reset_index()

#     pagerank_no_xmas.columns = ["node", "no_xmas_value"]

#     pagerank_no_xmas = pagerank_no_xmas.sort_values("no_xmas_value", ascending=False)

#     # pagerank_diff = pagerank_xmas.join(
#     #     pagerank_no_xmas.set_index("node"), on="node", how="outer"
#     # )
#     pagerank_diff = nintendo_pagerank.join(
#         no_nintendo_pagerank.set_index("node"), on="node", how="outer"
#     )

#     # pagerank_diff["diff"] = pagerank_diff["xmas_value"] - pagerank_diff["no_xmas_value"]
#     pagerank_diff["diff"] = (
#         pagerank_diff["nintendo_value"] - pagerank_diff["no_nintendo_value"]
#     ).round(decimals=5)

#     # pagerank_diff = pagerank_diff[pagerank_diff["diff"].abs() > 0.00]

#     print(coocc.shape)

#     coords = pd.DataFrame(
#         tsne.fit_transform(coocc), index=tsne.feature_names_in_, columns=["x", "y"]
#     )

#     sizes = pd.DataFrame(index=coocc.index, data=np.diag(coocc), columns=["size"])

#     sizes.to_csv("unique_tags.csv")

#     node_data = pd.DataFrame(sizes).join(coords)

#     np.fill_diagonal(coocc.values, 0)

#     coocc = (
#         coocc.where(np.triu(np.ones(coocc.shape)).astype(bool)).fillna(0).astype(int)
#     )

#     edge_list = (
#         coocc.reset_index()
#         .replace({0: np.nan})
#         .melt(id_vars=["index"])
#         .dropna()
#         .reset_index(drop=True)
#         .astype({"value": "int"})
#     )

#     edge_list.columns = ["source", "target", "weight"]

#     nodes = [
#         {
#             "data": {
#                 "id": tag.Index,
#                 "label": tag.Index,
#                 "size": tag.size,
#             },
#             "position": {
#                 "x": tag.x * zoom,
#                 "y": tag.y * zoom,
#             },
#         }
#         for tag in node_data.itertuples()
#     ]

#     edges_lists = [
#         [
#             {
#                 "data": {
#                     "source": row[0],
#                     "target": tag[0],
#                     "label": tag[0],
#                     "weight": tag[1],
#                 }
#             }
#             for tag in row[1].items()
#             if tag[1] > 0
#         ]
#         for row in coocc.iterrows()
#     ]

#     edges = list(itertools.chain(*edges_lists))

#     bar_df = pagerank_diff.copy()
#     bar_df["no_nintendo_value"] *= -1

#     bar_df["diff"] = (
#         bar_df["diff"]
#         .fillna(bar_df["nintendo_value"])
#         .fillna(bar_df["no_nintendo_value"])
#     )

#     bar_df = bar_df[bar_df["diff"].abs() > 0.007]

#     fig = go.Figure()

#     color_dict = {"nintendo_value": "#e4000f", "no_nintendo_value": "lightslategray"}
#     for col in ["nintendo_value", "no_nintendo_value"]:
#         fig.add_trace(
#             go.Bar(
#                 x=bar_df[col].values,
#                 y=bar_df["node"],
#                 orientation="h",
#                 name=col.replace("_", " ").title(),
#                 customdata=bar_df[col],
#                 hovertemplate="%{y}: %{customdata}",
#                 marker_color=color_dict[col],
#             )
#         )

#     fig.add_trace(
#         go.Scatter(
#             x=bar_df["diff"].values,
#             y=bar_df["node"],
#             mode="markers",
#             name="Difference",
#             marker_symbol="line-ns",
#             marker_line_color="midnightblue",
#             marker_color="lightskyblue",
#             marker_line_width=2,
#             marker_size=15,
#         )
#     )

#     fig.update_layout(
#         barmode="relative",
#         height=800,
#         width=600,
#         yaxis_autorange="reversed",
#         bargap=0.5,
#         legend_orientation="v",
#         legend_x=1,
#         legend_y=0,
#         plot_bgcolor="rgba(0, 0, 0, 0)",
#         paper_bgcolor="rgba(0, 0, 0, 0)",
#         xaxis_showgrid=False,
#         yaxis_showgrid=False,
#     )

#     return (
#         nodes + edges,
#         coocc.max().max(),
#         dbc.Table.from_dataframe(
#             pagerank_diff,
#             # pagerank_diff.sort_values("diff", ascending=False),
#             striped=True,
#             bordered=True,
#             hover=True,
#         ),
#         fig,
#     )


@app.callback(
    Output("network-graph", "generateImage"), Input("export-button", "n_clicks")
)
def export_image(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        return {"type": "png", "action": "download"}
    raise dash.exceptions.PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)
