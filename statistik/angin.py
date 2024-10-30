import polars as pl
import typer
from matplotlib import pyplot as plt
from typer import Typer
from typing_extensions import Annotated


app = Typer(help="Sub aplikasi untuk kelola statistik angin.")
import scienceplots

plt.style.use(["science"])


@app.command(help="Perhitungan nilai maksimum dan rata-rata angin per tahun bulan.")
def maxavg(
    fname: Annotated[
        str, typer.Argument(help="Lokasi beserta nama file")
    ] = "./contoh/data.csv",
    simpan: Annotated[
        str, typer.Option(help="Simpan dengan nama file.")
    ] = "final.xlsx",
    tanggal: Annotated[
        str, typer.Option(help="Kolom waktu yang dipilih dalam file fname.")
    ] = "Tanggal",
    avg: Annotated[
        str, typer.Option(help="Kolom angin rata-rata yang dipilih dalam file fname.")
    ] = "ff_avg",
    max: Annotated[
        str, typer.Option(help="Kolom angin maksimum yang dipilih dalam file fname.")
    ] = "ff_x",
):
    df = pl.read_csv(fname)

    df = df.with_columns(
        pl.col(tanggal).str.to_date(format="%d-%m-%Y").alias("Tanggal")
    )

    df = df.filter(pl.col(avg) > 0)

    aggr = df.group_by(pl.col(tanggal).dt.strftime("%Y-%m"), maintain_order=True).agg(
        pl.col(max).max().alias("vmax"), pl.col(avg).mean().alias("vavg")
    )

    aggr.write_excel(simpan, worksheet="Statistik")


@app.command(help="Plot Atmospheric Boundary Layer (ABL).")
def plotabl(
    fname: Annotated[str, typer.Argument(help="Lokasi beserta nama file")],
):
    import numpy as np

    fs = 15
    pos = np.arange(0, 70)
    data = np.random.rand(100, 70)

    # df = pl.read_csv(fname)

    # print(df.head())

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))

    axs[0].grid()
    axs[0].set_title(r"Distribusi Kecepatan")
    axs[0].set_xlabel(r"Kecepatan (m/s)", fontsize=fs)
    axs[0].set_ylabel(r"Ketinggian (m)", fontsize=fs)
    plot = axs[0].violinplot(
        data,
        pos,
        points=80,
        vert=False,
        widths=0.7,
        showmeans=True,
        showextrema=True,
        showmedians=False,
    )

    plot["cmins"].set_color("darkgray")
    plot["cmaxes"].set_color("darkgray")
    plot["cbars"].set_color("darkgray")
    plot["cmeans"].set_linestyle("--")
    for body in plot["bodies"]:
        body.set_facecolor("lightgray")
        body.set_alpha(0.5)

    axs[1].grid()
    axs[1].set_title(r"Grafik ABL")
    axs[1].set_xlabel(r"Kecepatan rata-rata (m/s)", fontsize=fs)
    axs[1].set_ylabel(r"Ketinggian (m)", fontsize=fs)

    axs[1].scatter(data[0, :], pos, 30, marker="x", label="Rata-rata")
    # axs[1].plot(ured_h, wh_pl, linewidth=1.3, label="fit")
    axs[1].legend()

    fig.tight_layout()
    plt.show()
