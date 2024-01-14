import glob
import numpy as np
import polars as pl


class FrekHelper:
    def __init__(self, fname, skip_rows=19, sep=","):
        self.fname = fname
        self.df = pl.read_csv(
            self.fname,
            separator=sep,
            has_header=False,
            skip_rows=skip_rows,
            truncate_ragged_lines=True
        )
        self.df.columns = ["t", "depan", "belakang"]
        self.df = self.df.select(
            pl.all().str.replace(",", ".").cast(pl.Float32))


class CollectionHelper:
    def __init__(self, path, frekmin, frekmaks, displacement=False, bentang=0, skala=0):
        self.path = path
        self.skala = skala
        self.bentang = bentang
        self.displacement = displacement

        queries = []
        for file in glob.glob(self.path):
            kec = file.replace(".txt", "").split("_")[-1]

            q = pl.scan_csv(
                file,
                skip_rows=15,
                separator="\t",
                has_header=False,
                new_columns=["f", "belakang", "depan"]
            )

            q = q.select(pl.all().str.replace(",", "."))
            q = q.with_columns(pl.lit(kec).alias("v"))
            q = q.select(pl.all().cast(pl.Float32))
            q = q.filter((pl.col("f") >= frekmin) & (pl.col("f") <= frekmaks))

            if self.displacement:
                q = q.with_columns(
                    ((pl.col("depan") + pl.col("belakang")) / 2).alias("heaving"),
                    ((pl.col("depan") - pl.col("belakang")) / 2).abs().alias("torsion")
                )

                q = q.with_columns(
                    (1000 * pl.col("heaving") /
                     (4 * (np.pi ** 2) * pl.col("f") ** 2)).alias("dispmodelheaving"),
                    (pl.col("torsion") / (4 * (np.pi ** 2) *
                     pl.col("f") ** 2)).alias("dispmodeltorsion")
                )

                q = q.with_columns(
                    (np.rad2deg(
                        np.arcsin(2 * pl.col("dispmodeltorsion") / self.bentang))).alias("theta")
                )

                q = q.with_columns(
                    (self.skala * pl.col("dispmodelheaving")
                     ).alias("dispaktualheaving")
                )

            queries.append(q)

        self.collections = pl.collect_all(queries)
