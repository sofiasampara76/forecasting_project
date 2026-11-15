from pathlib import Path

import pandas as pd

file_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "136-indeksi-spozhivchikh-tsin.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
periods = sorted(df['Період'].dropna().astype(str).unique())

print(f"Всього унікальних періодів: {len(periods)}")
print(f"Початок даних: {periods[0]}")
print(f"Кінець даних:   {periods[-1]}")