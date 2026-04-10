"""
Проверка CSV с историей: читает файл из data/ и показывает структуру.
Положите файл в data/ и при необходимости задайте имя через переменную DATA_FILE ниже
или аргументом: python src/check_data.py xauusd_m5.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Корень проекта: .../gold-ml
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Имя файла по умолчанию (замените на своё или передайте аргументом)
DEFAULT_NAME = "xauusd_m5.csv"

# Возможные имена колонки времени (первая найденная будет использована как индекс времени)
TIME_CANDIDATES = ("time", "Time", "datetime", "DateTime", "date", "timestamp")


def find_csv(name: str | None) -> Path:
    if name:
        p = DATA_DIR / name
        if p.is_file():
            return p
        raise FileNotFoundError(f"Нет файла: {p}")

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Папка data не найдена: {DATA_DIR}\nСоздайте: mkdir -p {DATA_DIR}"
        )

    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"В {DATA_DIR} нет .csv файлов. Положите экспорт MT5 или передайте имя: "
            f"python src/check_data.py {DEFAULT_NAME}"
        )
    if len(csvs) > 1:
        print("Найдено несколько CSV, берём первый по имени:", csvs[0].name, file=sys.stderr)
    return csvs[0]


def pick_time_column(df: pd.DataFrame) -> str | None:
    for c in TIME_CANDIDATES:
        if c in df.columns:
            return c
    return None


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    path = find_csv(arg)

    print("Файл:", path)
    df = pd.read_csv(path, low_memory=False)

    tcol = pick_time_column(df)
    if tcol:
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        if df[tcol].isna().all():
            print("Предупреждение: не удалось распарсить время в колонке", tcol)

    print("\n--- head ---")
    print(df.head())
    print("\n--- dtypes ---")
    print(df.dtypes)
    print("\n--- shape ---", df.shape)


if __name__ == "__main__":
    main()
