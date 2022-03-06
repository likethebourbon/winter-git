import pandas as pd

df = pd.read_csv("data_220306.csv")


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


tags = make_edgelist(df)

metatags = pd.DataFrame(tags[tags["value"].str.contains(":")]["value"])

metatags["inner_tag"] = metatags["value"].apply(lambda x: x.split(":")[0])

metatags["metatags"] = metatags["value"].apply(lambda x: x.split(":")[1:])

metatags = metatags.explode("metatags", ignore_index=True)

metatags["to_count"] = metatags["metatags"] + ":" + metatags["inner_tag"]

pd.DataFrame(metatags["to_count"].value_counts()).reset_index().sort_values(
    by="index"
).to_csv("metatags_breakdown.csv")
