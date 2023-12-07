import polars as pl


if __name__ == "__main__":
  df = pl.read_csv("data.csv")

  df = df.with_columns(
    pl.col("Tanggal").str.to_date(format="%d-%m-%Y").alias("Tanggal")
  )

  aggr = df.group_by(
    pl.col("Tanggal").dt.strftime("%Y-%m"),
    maintain_order=True
  ).agg(
    pl.col("ff_x").max().alias("vmax"),
    pl.col("ff_avg").mean().alias("vavg")
  )

  aggr.write_excel("final.xlsx", worksheet="Statistik")