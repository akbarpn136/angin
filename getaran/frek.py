from typer import Typer

from getaran.command.hitung_fm import fmz
from getaran.command.hitung_lsce import lsce
from getaran.command.hitung_mitd import mitd
from getaran.command.plot_getaran import plotgtr
from getaran.command.olah_riv import waterfall, displacement


app = Typer(help="Sub aplikasi untuk perhitungan frekuensi dan damping.")

app.command(help="Perhitungan nilai frekuensi, damping factor dan damping ratio.")(lsce)
app.command(help="Perhitungan nilai frekuensi, damping dan stifness.")(mitd)
app.command(help="Menampilkan plot getaran beserta seleksi berdasarkan waktu.")(plotgtr)
app.command(help="Olah data RIV dalam bentuk waterfall.")(waterfall)
app.command(help="Menghitung displacement dari data RIV.")(displacement)
app.command(help="Menghitung flutter margin dengan zimmerman.")(fmz)
