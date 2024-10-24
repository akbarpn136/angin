import numpy as np
import polars as pl
import typer
from prettytable import PrettyTable
from scipy.linalg import hankel
from typing_extensions import Annotated

from getaran.helper import FrekHelper


def lsce(
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
    c = 1 / 30
    Moda = 32 * 2
    hlp = FrekHelper(fname=fname, sep="\t")
    df = hlp.df
    x = df.filter((pl.col(t) >= tl) & (pl.col(t) <= tr))
    x = x.with_columns(
        pl.col(t).diff().alias("dt"),
        (pl.col(t) - pl.col(t).min()).alias("twork"),
        (pl.col(t).min() - tl).alias("dtwork"),
        pl.col(t).count().alias("lh"),
        pl.col(hh).abs().max().alias("M"),
    )

    lh = x.select(pl.col("lh")).item(1, 0)
    x = x.with_columns((pl.col(hh) / pl.col("M")).alias("h"))

    dt = x.select(pl.col("dt")).item(1, 0)
    M = x.select(pl.col("M")).item(1, 0)
    h = x.select(pl.col(hh)).to_numpy()
    twork = x.select(pl.col("twork")).to_numpy()
    xlsce = x.select(pl.col(hh)).slice(0, lh - Moda).to_numpy()
    ylsce = x.select(pl.col(hh)).slice(lh - Moda - 1, -1).to_numpy()
    Ho = x.select(pl.col(hh) * -1).slice(Moda, lh).to_numpy()
    H = hankel(xlsce, ylsce)

    # Perform the least square to get the multiplier b
    b = np.linalg.lstsq(H, Ho, rcond=None)
    b = np.append(b[0], [[1]], axis=0)
    b = np.flipud(b)

    # Get the poles using natural logarithm function
    tmp = np.roots(b.flatten())
    S = np.log(tmp) / dt

    # Get the characteristic frequencies and dampings ( decay rate, damping factor), and the amplitude
    freqf = np.imag(S) / 2 / np.pi
    dampf = np.real(S)
    tmp = np.exp(np.dot(twork, S.reshape(1, -1)))
    A = np.linalg.lstsq(tmp, h * M, rcond=None)[0]

    # Recalculate the result
    Ylsce = np.real(np.dot(tmp, A))

    # Fit the amplitudes
    A = A * 2

    # Filtering negative frequencies
    tmp = np.where(freqf >= 0)[0]
    A = A[tmp]
    freqf = freqf[tmp]
    dampf = dampf[tmp]
    S = S[tmp]

    # Damping ratio (%)
    dampr = dampf / np.abs(S) * 100

    # Picking first selected characteristics
    tmp = np.where(np.abs(A) / np.max(np.abs(A)) >= c)[0]
    A = A[tmp]
    freqf = freqf[tmp]
    dampf = dampf[tmp]
    dampr = dampr[tmp]

    # Sort from smallest to largest frequency
    tmp = np.argsort(freqf)
    A = A[tmp]
    freqf = freqf[tmp]
    dampf = dampf[tmp]
    dampr = dampr[tmp]

    table = PrettyTable()
    table.field_names = ["freqf", "dampf", "dampr"]

    for i in range(A.size):
        table.add_row([freqf[i], dampf[i], dampr[i]])

    print(table)


def scanlsce(
    fname: Annotated[
        str, typer.Argument(help="Lokasi beserta nama file")
    ] = "./contoh/getaran.csv",
    rentang: Annotated[
        int,
        typer.Option(help="Rentang baris data yang digunakan. Contoh --rentang 100"),
    ] = 100,
    sudut: Annotated[
        float,
        typer.Option(help="Sudut serang dek jembatan. Contoh --sudut 0.0"),
    ] = 0.0,
):
    hlp = FrekHelper(fname=fname, sep="\t")
    df = hlp.df

    v = fname.split("/")[-1].replace("v_", "").replace(".txt", "")
    koleksi = []
    for frame in df.iter_slices(n_rows=rentang):
        tl = frame.item(0, 0)
        tr = frame.item(-1, 0)

        try:
            f_h, dampf_h, dampr_h = hlp.calc_lsce(tl, tr, "heaving")
            f_t, dampf_t, dampr_t = hlp.calc_lsce(tl, tr, "torsion")

            obj = dict(
                f1=f_h,
                f2=f_t,
                dampf1=dampf_h,
                dampr1=dampr_h,
                dampf2=dampf_t,
                dampr2=dampr_t,
            )

            ddf = pl.DataFrame(obj)

            koleksi.append(ddf)

        except Exception as _:
            continue

    frames = pl.concat(koleksi)
    frames = frames.with_columns(pl.lit(sudut).alias("sudut"), pl.lit(v).alias("v"))
    frames = frames.select(pl.col("sudut", "v", "f1", "f2", "dampf1", "dampf2"))

    frames.write_csv(f"FM_{sudut}_{v}.csv")
