import numpy as np
import polars as pl

from matplotlib import pyplot as plt
from numpy.polynomial import polynomial as poly
from typer import Argument, Option
from typing_extensions import Annotated


def fmz(
    fname: Annotated[str, Argument(help="File flutter margin.")] = "./contoh/fm.csv",
    sudut: Annotated[
        int, Option(help="Tampilkan data waterfall untuk sudut tertentu.")
    ] = 0,
    show_index: Annotated[
        bool, Option(help="Tampilkan data plot index saja?.")
    ] = False,
):
    import scienceplots

    plt.style.use(["science", "high-vis"])

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
        (((o2**2 - o1**2) / 2 + (bet2**2 - bet1**2) / 2) ** 2).alias("F1"),
        (
            4 * bet1 * bet2 * ((o2**2 + o1**2) / 2 + 2 * ((bet2 + bet1) / 2) ** 2)
        ).alias("F2"),
        (
            ((bet2 - bet1) / (bet2 + bet1))
            * ((o2**2 - o1**2) / 2 + 2 * (((bet2 + bet1) / 2) ** 2) ** 2)
        ).alias("F3"),
    )

    df = df.with_columns(((pl.col("F1") + pl.col("F2") + pl.col("F3")) / fs).alias("F"))

    df = df.select(pl.col("q", "F"))
    qq = df.select(pl.col("q")).to_numpy().flatten()
    ff = df.select(pl.col("F")).to_numpy().flatten()

    # Curve Fit
    np.seterr("warn")
    coef = poly.polyfit(qq, ff, 1)
    pp = poly.Polynomial(coef)
    tt = np.linspace(0, qq.max(), qq.size)

    # Calculate critical velocity
    qcrit = poly.polyroots(coef)

    try:
        vcrit = np.round(np.sqrt(2 * qcrit[0] / rho), 2)
    except Warning as e:
        print(f"WARNING: {e}")
        vcrit = "N/A"
    except Exception as e:
        print(f"EXCEPTION: {e}")
        vcrit = "N/A"

    # Hitung R Squared
    yhat = pp(tt)
    ybar = ff.mean()
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((ff - ybar) ** 2)
    rsquared = np.round(ssreg / sstot, 2)
    print(f"RSquared = {rsquared}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.annotate(
        r"$V_{crit} ="
        + f"{vcrit}$ m/s, "
        + r"$q_{crit} ="
        + f"{np.round(qcrit[0], 2)}$ Pa",
        xy=(40, 10),
        xycoords="figure points",
    )
    if show_index:
        for idx, _ in enumerate(qq):
            plt.annotate(idx + 1, xy=(qq[idx], ff[idx]))

    plt.scatter(qq, ff, marker="x", color="black", label="raw")
    plt.plot(tt, yhat, color="green", linewidth=1, label="fit")
    plt.title(f"Flutter Margin $(\\alpha = {sudut}^\\circ)$")
    plt.xlabel(r"q $(Pa)$", fontsize=12)
    plt.ylabel(r"$F_z$", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
