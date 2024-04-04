import math
import typer
import numpy as np
import polars as pl
from typing_extensions import Annotated

from getaran.helper import FrekHelper


def _itd(df, idxl, idxh, idxt):
    # delta time
    dt = df.select(pl.col("t").diff().take(1)).item()

    # frequency sampling used
    fs = 1 / dt

    # number of data
    N = df.select(pl.col("t").count()).item()

    t = df.select(pl.col("t").slice(idxt, idxl + 1)).to_numpy().flatten()
    yh = df.select(pl.col("heaving").slice(idxh, idxl + 1)).to_numpy().flatten()
    yt = df.select(pl.col("torsion").slice(idxt, idxl + 1)).to_numpy().flatten()

    # first time shift coefficient
    N1 = math.ceil(1 / (32 * dt))

    # second time shift
    N2 = N1 + 1

    # set heaving motion
    set1_h = yh[: yh.size - N1 - N2 - 1]
    set2_h_N1 = yh[N1 : yh.size - N2 - 1]
    set2_h_N2 = yh[N2 : yh.size - N1 - 1]
    set3_h_N1N2 = yh[N1 + N2 : -1]

    # set torsional motion
    set1_t = yt[: yt.size - N1 - N2 - 1]
    set2_t_N1 = yt[N1 : yt.size - N2 - 1]
    set2_t_N2 = yt[N2 : yt.size - N1 - 1]
    set3_t_N1N2 = yt[N1 + N2 : -1]

    # system matrices
    phi1 = np.vstack((set1_h, set1_t, set2_h_N2, set2_t_N2))
    phi2 = np.vstack((set2_h_N1, set2_t_N1, set3_h_N1N2, set3_t_N1N2))

    # positif cycle
    # A_n = np.dot(phi2, phi2.T) / np.dot(phi1, phi2.T)

    print(phi1)


def mitd(
    fname: Annotated[
        str, typer.Argument(help="Lokasi beserta nama file")
    ] = "./contoh/getaran.csv",
    idxh: Annotated[
        int, typer.Option(help="Batas index minimum heaving. Contoh --idxh 0")
    ] = 0,
    idxt: Annotated[
        int, typer.Option(help="Batas index minimum torsion. Contoh --idxt 0")
    ] = 0,
    idxl: Annotated[
        int, typer.Option(help="Seberapa banyak baris data yang digunakan.")
    ] = 10,
    hh: Annotated[
        str, typer.Option(help="Kolom getaran yang dipilih dalam file fname.")
    ] = "h",
):
    hlp = FrekHelper(fname=fname, sep="\t")
    df = hlp.df
    df = df.with_columns(
        ((pl.col("depan") + pl.col("belakang")) / 2).alias("heaving"),
        ((pl.col("depan") - pl.col("belakang")) / 2).abs().alias("torsion"),
    )

    _itd(df=df, idxl=idxl, idxh=idxh, idxt=idxt)
