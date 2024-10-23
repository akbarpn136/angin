import glob

import numpy as np
import polars as pl
from scipy.linalg import hankel


class FrekHelper:
    def __init__(self, fname, skip_rows=19, sep=",", B=0.12):
        self.fname = fname
        self.B = B
        self.df = pl.read_csv(
            self.fname,
            separator=sep,
            has_header=False,
            skip_rows=skip_rows,
            truncate_ragged_lines=True,
        )
        self.df.columns = ["t", "depan", "belakang"]
        self.df = self.df.drop_nulls()

        # B merupakan jarak antar sensor akselerometer
        # Satuan B dalam meter, jadi perlu dikonversi ke mm
        varb = (pl.col("depan") - pl.col("belakang")).abs() / (self.B * 1000)

        self.df = self.df.with_columns(
            ((pl.col("depan") + pl.col("belakang")) / 2).alias("heaving"),
            (
                pl.when(pl.col("depan") > pl.col("belakang"))
                .then(-np.arcsin(varb))
                .otherwise(np.arcsin(varb))
            ).alias("torsion"),
        )

    def calc_lsce(self, tl, tr, y="heaving"):
        c = 1 / 30
        Moda = 32 * 2
        x = self.df.filter((pl.col("t") >= tl) & (pl.col("t") <= tr))
        x = x.with_columns(
            pl.col("t").diff().alias("dt"),
            (pl.col("t") - pl.col("t").min()).alias("twork"),
            (pl.col("t").min() - tl).alias("dtwork"),
            pl.col("t").count().alias("lh"),
            pl.col(y).abs().max().alias("M"),
        )

        lh = x.select(pl.col("lh")).item(1, 0)
        x = x.with_columns((pl.col(y) / pl.col("M")).alias("h"))

        dt = x.select(pl.col("dt")).item(1, 0)
        M = x.select(pl.col("M")).item(1, 0)
        h = x.select(pl.col(y)).to_numpy()
        twork = x.select(pl.col("twork")).to_numpy()
        xlsce = x.select(pl.col(y)).slice(0, lh - Moda).to_numpy()
        ylsce = x.select(pl.col(y)).slice(lh - Moda - 1, -1).to_numpy()
        Ho = x.select(pl.col(y) * -1).slice(Moda, lh).to_numpy()
        H = hankel(xlsce, ylsce)

        # Perform the least square to get the multiplier b
        b = np.linalg.lstsq(H, Ho, rcond=None)
        b = np.append(b[0], [[1]], axis=0)
        b = np.flipud(b)

        # Get the poles using natural logarithm function
        tmp = np.roots(b.flatten())
        S = np.log(tmp) / dt

        # Get the characteristic frequencies and dampings ( decay rate, damping factor), and the amplitude
        freqf = np.imag(S) / 2 / np.pi
        dampf = np.real(S)
        tmp = np.exp(np.dot(twork, S.reshape(1, -1)))
        A = np.linalg.lstsq(tmp, h * M, rcond=None)[0]

        # Recalculate the result
        Ylsce = np.real(np.dot(tmp, A))

        # Fit the amplitudes
        A = A * 2

        # Filtering negative frequencies
        tmp = np.where(freqf >= 0)[0]
        A = A[tmp]
        freqf = freqf[tmp]
        dampf = dampf[tmp]
        S = S[tmp]

        # Damping ratio (%)
        dampr = dampf / np.abs(S) * 100

        # Picking first selected characteristics
        tmp = np.where(np.abs(A) / np.max(np.abs(A)) >= c)[0]
        A = A[tmp]
        freqf = freqf[tmp]
        dampf = dampf[tmp]
        dampr = dampr[tmp]

        # Sort from smallest to largest frequency
        tmp = np.argsort(freqf)
        A = A[tmp]
        freqf = freqf[tmp]
        dampf = dampf[tmp]
        dampr = dampr[tmp]

        return freqf, dampf, dampr


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
                new_columns=["f", "depan", "belakang"],
            )

            q = q.select(pl.all().str.replace(",", "."))
            q = q.with_columns(pl.lit(kec).alias("v"))
            q = q.select(pl.all().cast(pl.Float32))
            q = q.filter((pl.col("f") >= frekmin) & (pl.col("f") <= frekmaks))

            if self.displacement:
                q = q.with_columns(
                    ((pl.col("depan") + pl.col("belakang")) / 2).alias("heaving"),
                    ((pl.col("depan") - pl.col("belakang")) / 2).abs().alias("torsion"),
                )

                q = q.with_columns(
                    (
                        1000 * pl.col("heaving") / (4 * (np.pi**2) * pl.col("f") ** 2)
                    ).alias("dispmodelheaving"),
                    (pl.col("torsion") / (4 * (np.pi**2) * pl.col("f") ** 2)).alias(
                        "dispmodeltorsion"
                    ),
                )

                q = q.with_columns(
                    (
                        np.rad2deg(
                            np.arcsin(2 * pl.col("dispmodeltorsion") / self.bentang)
                        )
                    ).alias("theta")
                )

                q = q.with_columns(
                    (self.skala * pl.col("dispmodelheaving")).alias("dispaktualheaving")
                )

            queries.append(q)

        self.collections = pl.collect_all(queries)
