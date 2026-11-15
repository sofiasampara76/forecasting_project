from pathlib import Path

import pandas as pd
import openpyxl

file_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "136-indeksi-spozhivchikh-tsin.xlsx"

print("="*60)
print("1. ШВИДКИЙ ПЕРЕДПЕРЕГЛЯД СТРУКТУРИ (Читаємо лише 5 рядків)")
print("="*60)

# Читаємо лише перші 5 рядків — це займає лічені мегабайти RAM
df_sample = pd.read_excel(file_path, engine='openpyxl', nrows=5)

print("\nЗнайдені колонки:")
for idx, col in enumerate(df_sample.columns):
    print(f"  [{idx}] {col}")

print("\nПерші 2 рядки даних:")
print(df_sample.head(2))


print("\n" + "="*60)
print("2. ПОТОКОВИЙ АНАЛІЗ ВСЬОГО ФАЙЛУ (Режим read_only = безризиковий для RAM)")
print("="*60)

# Завантажуємо книгу в режимі read_only (пам'ять майже не витрачається)
wb = openpyxl.load_workbook(file_path, read_only=True)
sheet = wb.active

total_rows = 0
header = None

# Сет-структури для збору унікальних значень перших 3 колонок (для оцінки розмірності)
col_samples = {}

for row_idx, row in enumerate(sheet.iter_rows(values_only=True)):
    if row_idx == 0:
        header = row
        for i, col_name in enumerate(header):
            col_samples[i] = set()
        continue

    total_rows += 1

    # Фіксуємо унікальні значення для перших колонок (з обмеженням розширюваності)
    for i, val in enumerate(row):
        if val is not None and len(col_samples[i]) < 100:  # беремо до 100 прикладів на колонку
            col_samples[i].add(str(val))

wb.close()

print(f"Загальна кількість рядків у файлі: {total_rows:,}")
print("\nКороткий зріз унікальних значень по колонках:")

for i, col_name in enumerate(header):
    sample_list = list(col_samples[i])[:3] # перші 3 приклади
    print(f" • Колонка [{i}] '{col_name}': {len(col_samples[i])}+ унікальних значень (напр: {sample_list})")

print("="*60)