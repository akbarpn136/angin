import glob
import polars as pl
from typer import Option, Argument
from matplotlib import pyplot as plt
from typing_extensions import Annotated


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
    queries = []
    for file in glob.glob(f"contoh/sukamahi/*_{sudut}_*"):
        kec = file.replace(".txt", "").split("_")[-1]

        q = pl.scan_csv(
            file,
            skip_rows=15,
            separator="\t",
            has_header=False,
            new_columns=["f", "belakang", "depan"]
        )

        q = q.select(pl.all().str.replace(",", "."))
        q = q.with_columns(pl.lit(kec).alias("v"))
        q = q.select(pl.all().cast(pl.Float32))
        q = q.filter((pl.col("f") >= frekmin) & (pl.col("f") <= frekmaks))

        queries.append(q)

    collections = pl.collect_all(queries)

    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    for df in collections:
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
        zlabel=r"Akselerometer ($m/s^2$)"
    )

    ax2.set(
        title=f"Akselerometer Belakang (Sudut ${sudut}^\circ$)",
        zlim=(0, zmax),
        xlabel="Frekuensi (Hz)",
        ylabel="Kecepatan (m/s)",
        zlabel=r"Akselerometer ($m/s^2$)"
    )

    plt.tight_layout()
    plt.show()


def displacement():
    print("Displacement dari data RIV.")
