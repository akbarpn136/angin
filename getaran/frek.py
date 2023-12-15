from typer import Typer
from getaran.command.hitung_lsce import lsce
from getaran.command.plot_getaran import plotgtr


app = Typer(
    help="Sub aplikasi untuk perhitungan frekuensi dan damping."
)


app.command(
    help="Perhitungan nilai frekuensi, damping factor dan damping ratio."
)(lsce)


app.command(
    help="Menampilkan plot getaran beserta seleksi berdasarkan waktu."
)(plotgtr)
