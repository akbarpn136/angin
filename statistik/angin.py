import typer
import polars as pl
from typer import Typer
from typing_extensions import Annotated


app = Typer(
    help="Sub aplikasi untuk kelola statistik angin."
)


@app.command(
    help="Perhitungan nilai maksimum dan rata-rata angin per tahun bulan."
)
def maxavg(
    fname: Annotated[str, typer.Argument(
        help="Lokasi beserta nama file")] = "./contoh/data.csv",
    simpan: Annotated[str, typer.Option(
        help="Simpan dengan nama file.")] = "final.xlsx",
    tanggal: Annotated[str, typer.Option(
        help="Kolom waktu yang dipilih dalam file fname.")] = "Tanggal",
    avg: Annotated[str, typer.Option(
        help="Kolom angin rata-rata yang dipilih dalam file fname.")] = "ff_avg",
    max: Annotated[str, typer.Option(
        help="Kolom angin maksimum yang dipilih dalam file fname.")] = "ff_x",
):
    df = pl.read_csv(fname)

    df = df.with_columns(
        pl.col(tanggal).str.to_date(format="%d-%m-%Y").alias("Tanggal")
    )

    df = df.filter(pl.col(avg) > 0)

    aggr = df.group_by(
        pl.col(tanggal).dt.strftime("%Y-%m"),
        maintain_order=True
    ).agg(
        pl.col(max).max().alias("vmax"),
        pl.col(avg).mean().alias("vavg")
    )

    aggr.write_excel(simpan, worksheet="Statistik")
