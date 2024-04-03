import typer
import numpy as np
import polars as pl
from scipy.linalg import hankel
from prettytable import PrettyTable
from typing_extensions import Annotated

from getaran.helper import FrekHelper


def mitd(
    tl: Annotated[float, typer.Option(help="Batas waktu minimum. Contoh --tl 0.6159")],
    tr: Annotated[float, typer.Option(help="Batas waktu maksimum. Contoh --tr 2.2047")],
    fname: Annotated[
        str, typer.Argument(help="Lokasi beserta nama file")
    ] = "./contoh/getaran.csv",
    t: Annotated[
        str, typer.Option(help="Kolom waktu yang dipilih dalam file fname.")
    ] = "t",
    hh: Annotated[
        str, typer.Option(help="Kolom getaran yang dipilih dalam file fname.")
    ] = "h",
):
    hlp = FrekHelper(fname=fname, sep="\t")
    df = hlp.df

    print(df.head())
