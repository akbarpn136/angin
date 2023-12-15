import typer
import polars as pl
import pyvista as pv
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
    hlp = FrekHelper(fname=fname)
    df = hlp.df

    xpenuh = df.select(pl.col(t)).to_numpy()
    ypenuh = df.select(pl.col(hh)).to_numpy()

    filter = df.filter((pl.col(t) >= tl) & (pl.col(t) <= tr))
    xseleksi = filter.select(pl.col(t)).to_numpy()
    yseleksi = filter.select(pl.col(hh)).to_numpy()

    chart = pv.Chart2D(x_label=t, y_label=hh)
    chart.line(xpenuh, ypenuh, style="-", color="blue", label="penuh")
    chart.line(xseleksi, yseleksi, style="-", color="red", label="seleksi")
    chart.show()
