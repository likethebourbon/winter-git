from collections import Counter
import pandas as pd
import numpy as np
from typing import Optional


def import_data(filename):
    df = pd.read_csv(filename)
    df.columns = [
        col.replace(" ", "_").replace(r"/_", "").lower() for col in df.columns
    ]
    df = df.dropna(how="all", axis=1)
    return df


def make_edgelist(df, id_cols=None):
    if id_cols is None:
        id_cols = [
            "game",
            "track_title",
            "franchise",
            "year",
            "platform",
            "company",
            "game_and_track",
        ]
        # id_cols = ["game", "track_title"]

    # df["year"] = df["year"] // 10 * 10

    tag_columns = [col for col in df.columns if col not in id_cols + ["number_of_tags"]]

    df["game_and_track"] = df["game"] + "--" + df["track_title"]

    tags = df.set_index("game_and_track")[tag_columns]

    tags = tags.replace(to_replace=" ", value="_", regex=True)

    tags = (
        tags.melt(ignore_index=False)
        .dropna(subset=["value"])
        .reset_index()
        .astype({"value": str})
    )

    # c = Counter(tags["value"].tolist())

    # counts = pd.DataFrame(c.items())

    # counts = tags["value"].value_counts(sort=True)

    # tags["value"] = tags["value"].astype(str)

    # counts.to_csv("counts.csv")

    # tags = tags[tags["value"].isin(counts[counts > 1].index)]

    tags = tags[tags["value"] != "--"]

    tags = tags[["game_and_track", "value"]]

    # tags.columns = ["source", "target"]

    # return tags.reset_index(drop=True)

    return tags


def make_tags_df(tags, metatags, mlb, threshold=2, export_other=False):

    tags["value"] = tags["value"].astype(str)

    tags["metatag"] = (
        tags["value"].apply(lambda x: x.split(":")[1] if ":" in x else np.nan).squeeze()
    )

    tags["tag"] = tags["value"].apply(lambda x: x.split(":")[0] if ":" in x else x)

    counts = tags.value_counts(subset=["metatag", "tag"]).reset_index()

    other = (
        counts[counts[0] < 10]
        .groupby("metatag")
        .agg(other=(0, "sum"), to_drop=("tag", "unique"))
    )

    other = other.reset_index()

    other["tag"] = "other_" + other["metatag"]

    other["metatag"] = other["tag"] + ":" + other["metatag"]

    other = other.set_index("metatag")

    if export_other:
        other.to_csv("tags_grouped_as_other.csv")

    replace_dict = {v: k for k, v in other[["to_drop"]].explode("to_drop").itertuples()}

    tags["value"] = tags.apply(
        lambda x: x["value"].replace(
            x["value"], replace_dict.get(x["tag"], x["value"])
        ),
        axis=1,
    )

    # tags = tags[
    #     ~tags["tag"].isin([col for l in other["to_drop"].tolist() for col in l])
    # ]

    to_export = tags[["value"]].copy()

    to_export["value"] = to_export["value"].apply(
        lambda x: ":".join(x.split(":")[::-1]) if ":" in x else x
    )

    pd.DataFrame(to_export["value"].unique()).to_csv("unique_tags.csv")
    # to_export["value"].value_counts().to_csv("counts_with_metatags.csv")

    tags["value"] = tags["value"].apply(
        lambda x: x.split(":")[
            1 if x.split(":")[1] in metatags + ["majorish", "minorish"] else 0
        ]
        if ":" in x
        else x
    )
    # tags = tags.explode("value", ignore_index=True)

    tag_counts = tags["value"].value_counts()

    tags_to_drop = tag_counts[tag_counts < threshold + 1].index.tolist()

    tags = tags[~tags["value"].isin(tags_to_drop)]

    tags = tags.groupby("game_and_track").agg({"value": "unique"})

    tags = pd.DataFrame(
        mlb.fit_transform(tags["value"]), columns=mlb.classes_, index=tags.index
    )

    return tags


def filter_data(
    df,
    platform: Optional[list] = None,
    company: Optional[list] = None,
    franchise: Optional[list] = None,
    year_range: Optional[list] = None,
):
    if platform is not None:
        df = df[df["platform"].fillna("none").str.contains("|".join(platform))]

    if company is not None:
        df = df[df["company"].fillna("none").str.contains("|".join(company))]

    if franchise is not None:
        df = df[df["franchise"].fillna("none").str.contains("|".join(franchise))]

    if year_range is not None:
        df = df[df["year"].between(year_range[0], year_range[1])]

    return df
