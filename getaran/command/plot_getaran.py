import typer
import polars as pl
from matplotlib import pyplot as plt
from typing_extensions import Annotated

from getaran.helper import FrekHelper


def plotgtr(
    tl: Annotated[float, typer.Option(
        help="Batas waktu minimum. Contoh --tl 0.6159")],
    tr: Annotated[float, typer.Option(
        help="Batas waktu maksimum. Contoh --tr 2.2047")],
    fname: Annotated[str, typer.Argument(
        help="Lokasi beserta nama file")] = "./contoh/getaran.csv",
    t: Annotated[str, typer.Option(
        help="Kolom waktu yang dipilih dalam file fname.")] = "t",
    hh: Annotated[str, typer.Option(
        help="Kolom getaran yang dipilih dalam file fname.")] = "h",
):
    hlp = FrekHelper(fname=fname, sep="\t")
    df = hlp.df

    xpenuh = df.select(pl.col(t)).to_numpy()
    ypenuh = df.select(pl.col(hh)).to_numpy()

    filter = df.filter((pl.col(t) >= tl) & (pl.col(t) <= tr))
    xseleksi = filter.select(pl.col(t)).to_numpy()
    yseleksi = filter.select(pl.col(hh)).to_numpy()

    plt.plot(xpenuh, ypenuh, label="penuh", linewidth=1)
    plt.plot(xseleksi, yseleksi, color="r", label="seleksi", linewidth=1)
    plt.title("Seleksi getaran")
    plt.xlabel(t)
    plt.ylabel(hh)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
