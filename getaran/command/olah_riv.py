import glob
import polars as pl
from typer import Option, Argument
from matplotlib import pyplot as plt
from typing_extensions import Annotated

from getaran.helper import CollectionHelper


def waterfall(
    sudut: Annotated[int, Argument(
        help="Tampilkan data waterfall untuk sudut tertentu.")] = 0,
    frekmin: Annotated[float, Option(
        help="Batas minimum frekuensi (Hz).")] = 1,
    frekmaks: Annotated[float, Option(
        help="Batas maksimum frekuensi (Hz).")] = 30,
    zmax: Annotated[float, Option(
        help="Tampilkan nilai plot maksimum sumbu Z.")] = 0.5,
):
    helper = CollectionHelper(
        f"contoh/sukamahi/*_{sudut}_*",
        frekmin=frekmin,
        frekmaks=frekmaks,
    )

    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    for df in helper.collections:
        frek = df.select(pl.col("f")).to_numpy()
        depan = df.select(pl.col("depan")).to_numpy()
        belakang = df.select(pl.col("belakang")).to_numpy()
        v = df.select(pl.col("v")).to_numpy()

        ax1.plot(frek, v, depan)
        ax2.plot(frek, v, belakang)

    ax1.set(
        title=f"Akselerometer Depan (Sudut ${sudut}^\circ$)",
        zlim=(0, zmax),
        xlabel="Frekuensi (Hz)",
        ylabel="Kecepatan (m/s)",
        zlabel=r"Amplitudo ($m/s^2$)"
    )

    ax2.set(
        title=f"Akselerometer Belakang (Sudut ${sudut}^\circ$)",
        zlim=(0, zmax),
        xlabel="Frekuensi (Hz)",
        ylabel="Kecepatan (m/s)",
        zlabel=r"Amplitudo ($m/s^2$)"
    )

    plt.tight_layout()
    plt.show()


def displacement(
    sudut: Annotated[int, Argument(
        help="Tampilkan data waterfall untuk sudut tertentu.")] = 0,
    frekmin: Annotated[float, Option(
        help="Batas minimum frekuensi (Hz).")] = 1,
    frekmaks: Annotated[float, Option(
        help="Batas maksimum frekuensi (Hz).")] = 30,
    skala: Annotated[float, Option(
        help="Besaran konversi model ke aktual.")] = 1000,
    bentang: Annotated[float, Option(
        help="Lebar longitudinal dek.")] = 0.6788,
    aktual: Annotated[bool, Option(
        help="Tampilkan data aktual?")] = True,
):
    helper = CollectionHelper(
        f"contoh/sukamahi/*_{sudut}_*",
        frekmin=frekmin,
        frekmaks=frekmaks,
        displacement=True,
        bentang=bentang,
        skala=skala,
    )

    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for df in helper.collections:
        if aktual:
            disp = df.select(pl.col("dispaktualheaving")).to_numpy()
        else:
            disp = df.select(pl.col("dispmodelheaving")).to_numpy()

        theta = df.select(pl.col("theta")).to_numpy()
        v = df.select(pl.col("v")).to_numpy()

        ax1.plot(v[0], disp)
        ax2.plot(v[0], theta)

    ax1.set(
        title=f"Displacement Dek (Sudut ${sudut}^\circ$)",
        xlabel="Kecepatan (m/s)",
        ylabel="Displacement (m)",
    )

    ax2.set(
        title=f"Simpangan Dek (Sudut ${sudut}^\circ$)",
        xlabel="Kecepatan (m/s)",
        ylabel=r"$\theta$ ($^\circ$)",
    )

    plt.tight_layout()
    plt.grid(True)
    plt.show()
