import numpy as np
import polars as pl
import pyvista as pv
from scipy.linalg import hankel
from prettytable import PrettyTable


class FrekHelper:
    def __init__(self, fname):
        self.fname = fname
        self.df = pl.read_csv(self.fname)

    def calc_lsce(self, t, tl, tr, hh):
        c = 1/30
        Moda = 32 * 2
        x = self.df.filter((pl.col(t) >= tl) & (pl.col(t) <= tr))
        x = x.with_columns(
            pl.col(t).diff().alias("dt"),
            (pl.col(t) - pl.col(t).min()).alias("twork"),
            (pl.col(t).min() - tl).alias("dtwork"),
            pl.col(t).count().alias("lh"),
            pl.col(hh).abs().max().alias("M"),
        )

        lh = x.select(pl.col("lh")).item(1, 0)
        x = x.with_columns(
            (pl.col(hh) / pl.col("M")).alias("h")
        )

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
            table.add_row([
                freqf[i],
                dampf[i],
                dampr[i]
            ])

        print(table)

    def plot(self, t, tl, tr, hh):
        xpenuh = self.df.select(pl.col(t)).to_numpy()
        ypenuh = self.df.select(pl.col(hh)).to_numpy()

        filter = self.df.filter((pl.col(t) >= tl) & (pl.col(t) <= tr))
        xseleksi = filter.select(pl.col(t)).to_numpy()
        yseleksi = filter.select(pl.col(hh)).to_numpy()

        chart = pv.Chart2D(x_label=t, y_label=hh)
        chart.line(xpenuh, ypenuh, style="-", color="blue", label="penuh")
        chart.line(xseleksi, yseleksi, style="-", color="red", label="seleksi")
        chart.show()
