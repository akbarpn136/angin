from typer import Typer

from statistik import angin
from getaran import frek


app = Typer(
    help="Jembatan CLI digunakan untuk keperluan pengujian jembatan dalam terowongan angin.",
    pretty_exceptions_show_locals=False
)

app.add_typer(
    angin.app,
    name="angin",
    short_help="Sub aplikasi untuk kelola statistik angin.",
)

app.add_typer(
    frek.app,
    name="frek",
    short_help="Sub aplikasi untuk perhitungan frekuensi dan damping.",
)
