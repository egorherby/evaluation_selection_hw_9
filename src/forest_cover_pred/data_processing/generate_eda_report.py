import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv("data/raw/train.csv")
rep = ProfileReport(df, title="Forest Covertype Dataset Profiling Report")
rep.to_file("reports/report.html")
