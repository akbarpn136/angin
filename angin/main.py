import polars as pl


def run():
    df = pl.read_csv("data.csv")

    df = df.with_columns(
        pl.col("Tanggal").str.to_date(format="%d-%m-%Y").alias("Tanggal")
    )

    df = df.filter(pl.col("ff_avg") > 0)

    aggr = df.group_by(
        pl.col("Tanggal").dt.strftime("%Y-%m"),
        maintain_order=True
    ).agg(
        pl.col("ff_x").max().alias("vmax"),
        pl.col("ff_avg").mean().alias("vavg")
    )

    aggr.write_excel("final.xlsx", worksheet="Statistik")


def lsce():
    import numpy as np
    from scipy.linalg import hankel
    from prettytable import PrettyTable

    tl = 0.6159
    tr = 2.2047
    c = 1/30
    Moda = 32 * 2

    x = pl.read_csv("getaran.csv")
    x = x.filter((pl.col("t") >= tl) & (pl.col("t") <= tr))
    x = x.with_columns(
        pl.col("t").diff().alias("dt"),
        (pl.col("t") - pl.col("t").min()).alias("twork"),
        (pl.col("t").min() - tl).alias("dtwork"),
        pl.col("t").count().alias("lh"),
        pl.col("h").abs().max().alias("M"),
    )

    lh = x.select(pl.col("lh")).item(1, 0)
    x = x.with_columns(
        (pl.col("h") / pl.col("M")).alias("h")
    )

    dt = x.select(pl.col("dt")).item(1, 0)
    M = x.select(pl.col("M")).item(1, 0)
    h = x.select(pl.col("h")).to_numpy()
    twork = x.select(pl.col("twork")).to_numpy()
    xlsce = x.select(pl.col("h")).slice(0, lh - Moda).to_numpy()
    ylsce = x.select(pl.col("h")).slice(lh - Moda - 1, -1).to_numpy()
    Ho = x.select(pl.col("h") * -1).slice(Moda, lh).to_numpy()
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
