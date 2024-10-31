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
    alpha: Annotated[float, typer.Option(help="Konstanta profil lingkungan")] = 0.14,
    tol: Annotated[float, typer.Option(help="Nilai toleransi (dalam decimal)")] = 0.1,
    href: Annotated[
        float, typer.Option(help="Tinggi referensi pitot tube (satuan dalam cm)")
    ] = 200,
):
    import numpy as np

    fs = 15
    v = fname.split("/")[-1].replace("v_", "").replace(".csv", "")
    v = float(v)
    pos = np.array(
        [
            0,
            0.3,
            0.5,
            0.7,
            0.9,
            1.1,
            1.5,
            1.9,
            2.3,
            2.6,
            3,
            4.1,
            5.1,
            6.1,
            7.1,
            8.1,
            9.1,
            10.1,
            11.1,
            12.1,
            13.1,
            14.1,
            15.1,
            16.1,
            17.1,
            18.1,
            19,
            20,
            25.1,
            27.5,
            29.5,
            31.5,
            33.5,
            35.5,
            37.5,
            39.5,
            41.5,
            43.5,
            45.5,
            47.5,
            49.5,
            51.5,
            53.5,
            55.5,
            57.5,
            59.5,
            61.5,
            64,
            65.5,
            67.5,
            69.5,
            71.5,
            73.5,
            75.5,
            77.5,
            78,
            79,
            80.5,
            82,
            83.5,
            85,
            86.5,
            88,
            89.5,
            91,
            92.5,
            94,
            95.5,
            97,
            98.5,
        ]
    )

    scanner_1 = [f"1-P-{i}" for i in range(101, 165)]
    scanner_2 = [f"1-P-{i}" for i in range(201, 207)]

    selected_col = scanner_1 + scanner_2

    df = pl.read_csv(fname)
    df.columns = [i.strip() for i in df.columns]
    df = df.select(pl.col(selected_col))
    df = df.select(pl.all().str.strip_chars())
    df = df.select(pl.all().str.replace("- ", "-"))
    df = df.select(pl.all().str.to_decimal())
    df = df.select(pl.all() * 6894.76)  # PSI to Pa
    df = df.select(
        (pl.all() * 2 / 1.22).abs().sqrt()
    )  # Kalkulasi kecepatan ==> P_dyn = 0.5 * rho * v * v

    data = df.to_numpy()

    v_rata = df.select(pl.all().mean())
    v_rata = v_rata.transpose()
    v_rata.columns = ["v_h"]
    v_rata = v_rata.with_columns(pl.Series(name="h", values=pos))
    v_rata = v_rata.with_columns((v * (pl.col("h") / href) ** alpha).alias("v_teori"))
    v_rata = v_rata.with_columns(
        (pl.col("v_teori") + pl.col("v_teori") * tol).alias("v_teori_plus"),
        (pl.col("v_teori") - pl.col("v_teori") * tol).alias("v_teori_minus"),
    )

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 9))

    axs.grid()
    axs.set_title(r"Distribusi Kecepatan")
    axs.set_xlabel(r"Kecepatan (m/s)", fontsize=fs)
    axs.set_ylabel(r"Ketinggian (cm)", fontsize=fs)
    plot = axs.violinplot(
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

    fig.tight_layout()

    fig1, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 9))

    axs1.grid()
    axs1.set_title(r"Grafik ABL")
    axs1.set_xlabel(r"Kecepatan rata-rata (m/s)", fontsize=fs)
    axs1.set_ylabel(r"Ketinggian (cm)", fontsize=fs)

    axs1.scatter(
        v_rata["v_h"].to_numpy(),
        v_rata["h"].to_numpy(),
        30,
        marker="x",
        label="Rata-rata",
    )
    axs1.plot(
        v_rata["v_teori"].to_numpy(),
        v_rata["h"].to_numpy(),
        linewidth=1.3,
        label="Kalkulasi",
    )
    axs1.plot(
        v_rata["v_teori_minus"].to_numpy(),
        v_rata["h"].to_numpy(),
        linewidth=1.3,
        label=f"Toleransi -{tol * 100}\\%",
    )
    axs1.plot(
        v_rata["v_teori_plus"].to_numpy(),
        v_rata["h"].to_numpy(),
        linewidth=1.3,
        label=f"Toleransi +{tol * 100}\\%",
    )

    axs1.legend()

    fig1.tight_layout()
    plt.show()
