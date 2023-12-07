import polars as pl


def run():
    df = pl.read_csv("data.csv")

    df = df.with_columns(
        pl.col("Tanggal").str.to_date(format="%d-%m-%Y").alias("Tanggal")
    )

    df = df.filter(pl.col("ff_avg") > 0)

    aggr = df.group_by(
        pl.col("Tanggal").dt.strftime("%Y-%m"),
        maintain_order=True
    ).agg(
        pl.col("ff_x").max().alias("vmax"),
        pl.col("ff_avg").mean().alias("vavg")
    )

    aggr.write_excel("final.xlsx", worksheet="Statistik")
