import math
import typer
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from typing_extensions import Annotated
from scipy.linalg import lstsq, eig, pinv

from getaran.helper import FrekHelper


def _itd(df, fd, idxl, idxh, idxt):
    # delta time
    dt = df.select(pl.col("t").diff().take(1)).item()

    t = df.select(pl.col("t").slice(idxt, idxl + 1)).to_numpy().flatten()
    yh = df.select(pl.col("heaving").slice(idxh, idxl + 1)).to_numpy().flatten()
    yt = df.select(pl.col("torsion").slice(idxt, idxl + 1)).to_numpy().flatten()

    # first time shift coefficient
    N1 = math.ceil(1 / (4 * dt * fd))

    # second time shift
    N2 = N1 + 1

    # set heaving motion
    set1_h = yh[: yh.size - N1 - N2 - 1]
    set2_h_N1 = yh[N1 : yh.size - N2 - 1]
    set2_h_N2 = yh[N2 : yh.size - N1 - 1]
    set3_h_N1N2 = yh[N1 + N2 : -1]

    # set torsional motion
    set1_t = yt[: yt.size - N1 - N2 - 1]
    set2_t_N1 = yt[N1 : yt.size - N2 - 1]
    set2_t_N2 = yt[N2 : yt.size - N1 - 1]
    set3_t_N1N2 = yt[N1 + N2 : -1]

    # system matrices
    phi1 = np.vstack((set1_h, set1_t, set2_h_N2, set2_t_N2)).conj()
    phi2 = np.vstack((set2_h_N1, set2_t_N1, set3_h_N1N2, set3_t_N1N2)).conj()

    # positive cycle
    A_n1 = np.dot(phi2, phi2.T)
    A_n2 = np.dot(phi1, phi2.T)
    A_n = np.dot(A_n1, pinv(A_n2))

    # negative cycle
    A_p1 = np.dot(phi2, phi1.T)
    A_p2 = np.dot(phi1, phi1.T)
    A_p = np.dot(A_p1, pinv(A_p2))

    # average of positive and negative cycle
    A_a = (A_n + A_p) / 2

    eigval, eigvec = eig(A_a)

    # parameter identification
    lmd1 = np.log(eigval[0]) / (N1 * dt)
    lmd2 = np.log(eigval[1]) / (N1 * dt)
    lmd3 = np.log(eigval[2]) / (N1 * dt)
    lmd4 = np.log(eigval[3]) / (N1 * dt)

    D = np.array(
        [
            [eigvec[0, 0], eigvec[0, 1], eigvec[0, 2], eigvec[0, 3]],
            [eigvec[1, 0], eigvec[1, 1], eigvec[1, 2], eigvec[1, 3]],
            [
                lmd1 * eigvec[0, 0],
                lmd2 * eigvec[0, 1],
                lmd3 * eigvec[0, 2],
                lmd4 * eigvec[0, 3],
            ],
            [
                lmd1 * eigvec[1, 0],
                lmd2 * eigvec[1, 1],
                lmd3 * eigvec[1, 2],
                lmd4 * eigvec[1, 3],
            ],
        ]
    )

    vh0 = (yh[1] - yh[0]) / dt
    vt0 = (yt[1] - yt[0]) / dt

    ic = np.array([yh[0], yt[0], vh0, vt0]).reshape((-1, 1))
    # C = np.dot(inv(D), ic)
    C = lstsq(D, ic)[0]

    a = C[0, 0]
    b = C[1, 0]
    c = C[2, 0]
    d = C[3, 0]

    P11 = eigvec[0, 0]
    P12 = eigvec[0, 1]
    P13 = eigvec[0, 2]
    P14 = eigvec[0, 3]
    P21 = eigvec[1, 0]
    P22 = eigvec[1, 1]
    P23 = eigvec[1, 2]
    P24 = eigvec[1, 3]

    P = np.array(
        [
            [a * P11, b * P12, c * P13, d * P14],
            [a * P21, b * P22, c * P23, d * P24],
        ]
    )

    # P = np.real(P) + (1 * np.imag(P) * 1j)
    e = np.array(
        [
            np.exp(lmd1 * t),
            np.exp(lmd2 * t),
            np.exp(lmd3 * t),
            np.exp(lmd4 * t),
        ]
    )

    x = np.dot(P, e).T

    return t, yh, yt, x


def mitd(
    fname: Annotated[
        str, typer.Argument(help="Lokasi beserta nama file")
    ] = "./contoh/getaran.csv",
    idxh: Annotated[
        int, typer.Option(help="Batas index minimum heaving. Contoh --idxh 0")
    ] = 0,
    idxt: Annotated[
        int, typer.Option(help="Batas index minimum torsion. Contoh --idxt 0")
    ] = 0,
    idxl: Annotated[
        int, typer.Option(help="Seberapa banyak baris data yang digunakan.")
    ] = 10,
    fd: Annotated[
        float,
        typer.Option(
            help="Pilih nilai paling maksimum antara frekuensi heaving or torsion, satuan dalam Hz"
        ),
    ] = 10,
):
    hlp = FrekHelper(fname=fname, sep="\t", skip_rows=15)
    df = hlp.df
    df = df.with_columns(
        ((pl.col("depan") + pl.col("belakang")) / 2).alias("heaving"),
        ((pl.col("depan") - pl.col("belakang")) / 2).abs().alias("torsion"),
    )

    t, yh, yt, x = _itd(df=df, fd=fd, idxl=idxl, idxh=idxh, idxt=idxt)

    plt.figure(figsize=(9, 5))
    plt.subplot(121)
    plt.plot(t, yh, label="pengujian")
    plt.scatter(t, np.real(x[:, 0]), marker="x", color="black", label="ITD")
    plt.xlabel("time (s)")
    plt.ylabel("h [m]")
    plt.legend()
    plt.title("Heaving")

    plt.subplot(122)
    plt.plot(t, yt, label="pengujian")
    plt.scatter(t, np.real(x[:, 1]), marker="x", color="black", label="ITD")
    plt.xlabel("time (s)")
    plt.ylabel("$\\alpha$ (rad)")
    plt.legend()
    plt.title("Torsion")

    plt.show()
