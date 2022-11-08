import itertools
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import dash_daq as daq
import numpy as np
import pandas as pd
from dash import Input, Output, dcc, html
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer

from utils import filter_data, import_data, make_edgelist, make_tags_df

mlb = MultiLabelBinarizer()

cwd = Path(__file__).parent

RAW_DATA_FILE = cwd / "data" / "raw" / "winter-data_220308.csv"
DEFAULT_PURPLE = "#421f89"
df = import_data(RAW_DATA_FILE)

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

metatags = sorted(
    list(
        set(
            col.split(":")[1]
            for col in edgelist.loc[
                edgelist["value"].str.contains(":"), "value"
            ].unique()
        )
    )
)

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
                            value={"hex": DEFAULT_PURPLE},
                        ),
                        html.Br(),
                        html.P("Metatags to use"),
                        dcc.Dropdown(
                            options=[
                                {"label": c, "value": c}
                                for c in metatags
                                if c not in ["majorish", "minorish"]
                            ],
                            value=[
                                c for c in metatags if c not in ["majorish", "minorish"]
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
                        dbc.Button("Export image", id="export-button"),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        dbc.Alert(
                            "Only one track matches the current filters. Expand filter parameters to view the network of tags.",
                            id="network-alert",
                            is_open=False,
                        ),
                        cyto.Cytoscape(
                            id="network-graph",
                            layout={"name": "preset"},
                            style={"width": "100%", "height": "800px"},
                        ),
                    ],
                ),
            ]
        ),
    ]
)


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
                "line-color": edge_color.get("hex", DEFAULT_PURPLE),
                "background-color": edge_color.get("hex", DEFAULT_PURPLE),
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

        val = edgeData["label"]

        children_style = [
            {"selector": f'edge[source = "{val}"]', "style": {"line-color": "blue"}},
            {"selector": f'edge[target = "{val}"]', "style": {"line-color": "blue"}},
        ]

        return stylesheet + children_style
    return stylesheet


@app.callback(
    [
        Output("network-graph", "elements"),
        Output("network-alert", "is_open"),
        Output("max-store", "data"),
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

    if len(df) < 2:
        return {}, True, 0

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

    coocc = tags.T.dot(tags)

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
        False,
        coocc.max().max(),
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
