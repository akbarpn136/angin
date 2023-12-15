import polars as pl


class FrekHelper:
    def __init__(self, fname):
        self.fname = fname
        self.df = pl.read_csv(self.fname)
