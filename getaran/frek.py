import typer
from typer import Typer
from typing_extensions import Annotated

from getaran.helper import FrekHelper


app = Typer(
    help="Sub aplikasi untuk perhitungan frekuensi dan damping."
)


@app.command(
    help="Perhitungan nilai frekuensi, damping factor dan damping ratio."
)
def lsce(
    tl: Annotated[float, typer.Option(
        help="Batas waktu minimum. Contoh --tl 0.6159")],
    tr: Annotated[float, typer.Option(
        help="Batas waktu maksimum. Contoh --tr 2.2047")],
    fname: Annotated[str, typer.Argument(
        help="Lokasi beserta nama file")] = "getaran.csv",
    t: Annotated[str, typer.Option(
        help="Kolom waktu yang dipilih dalam file fname.")] = "t",
    hh: Annotated[str, typer.Option(
        help="Kolom getaran yang dipilih dalam file fname.")] = "h",
):
    frek = FrekHelper(fname=fname)

    frek.calc_lsce(t=t, tl=tl, tr=tr, hh=hh)
