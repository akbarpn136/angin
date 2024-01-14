import numpy as np
import polars as pl
from typer import Option, Argument
from matplotlib import pyplot as plt
from typing_extensions import Annotated
from numpy.polynomial import polynomial as poly


def fmz(
    fname: Annotated[str, Argument(
        help="File flutter margin.")] = "./contoh/sukamahi.csv",
    sudut: Annotated[
        int, Option(help="Tampilkan data waterfall untuk sudut tertentu.")
    ] = 0,
):
    rho = 1.2
    df = pl.read_csv(fname)
    df = df.filter(pl.col("sudut") == sudut)
    df = df.with_columns(
        (0.5 * rho * pl.col("v") ** 2).alias("q"),
        (2 * np.pi * pl.col("f1")).alias("omg1"),
        (2 * np.pi * pl.col("f2")).alias("omg2"),
    )

    fs = df.select(pl.col("omg1", "omg2")).row(0)
    fs = ((fs[1] ** 2 - fs[0] ** 2) / 2) ** 2

    o1 = pl.col("omg1")
    o2 = pl.col("omg2")
    bet1 = pl.col("dampf1")
    bet2 = pl.col("dampf2")

    df = df.with_columns(
        (((o2 ** 2 - o1 ** 2) / 2 + (bet2 ** 2 - bet1 ** 2) / 2) ** 2).alias("F1"),
        (4 * bet1 * bet2 * ((o2 ** 2 + o1 ** 2) / 2 +
         2 * ((bet2 + bet1) / 2) ** 2)).alias("F2"),
        (((bet2 - bet1) / (bet2 + bet1)) *
         ((o2 ** 2 - o1 ** 2) / 2 + 2 * (((bet2 + bet1) / 2) ** 2) ** 2)).alias("F3")
    )

    df = df.with_columns(
        ((pl.col("F1") + pl.col("F2") + pl.col("F3")) / fs).alias("F")
    )

    df = df.select(pl.col("sudut", "v", "q", "F"))
    vv = df.select(pl.col("v")).to_numpy().flatten()
    qq = df.select(pl.col("q")).to_numpy().flatten()
    ff = df.select(pl.col("F")).to_numpy().flatten()

    # Curve Fit
    coef = poly.polyfit(qq, ff, 2)
    pp = poly.Polynomial(coef)
    tt = np.linspace(0, vv.max(), 100)

    # Calculate critical velocity
    qcrit = poly.polyroots(coef)
    vcrit = np.sqrt(2 * qcrit / rho)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.annotate(r"$V_{crit} =" + f"{vcrit[0]}$ m/s",
                 xy=(10, 10), xycoords="figure points")
    plt.scatter(vv, ff, marker="x", color="black", label="raw")
    plt.plot(tt, pp(tt), color="green", linewidth=1, label="fit")
    plt.title(f"Flutter Margin $(\\alpha = {sudut}^\circ)$")
    plt.xlabel(r"V $(m/s)$")
    plt.ylabel("F")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
