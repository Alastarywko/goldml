"""
Агрегирует тиковый CSV в OHLCV-свечи и сохраняет в data/.

Запуск:
    python src/ticks_to_ohlcv.py             # по умолчанию M5, читает XAUUSD_TickData.csv
    python src/ticks_to_ohlcv.py --tf M1     # таймфрейм M1
    python src/ticks_to_ohlcv.py --tf H1     # таймфрейм H1

Таймфреймы: M1, M5, M15, M30, H1, H4, D1
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TF_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

CHUNK = 5_000_000  # строк за раз (регулируйте если мало/много RAM)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Тики → OHLCV")
    p.add_argument("--src", default="XAUUSD_TickData.csv", help="Имя файла тиков в data/")
    p.add_argument("--tf", default="M5", choices=TF_MAP.keys(), help="Таймфрейм")
    return p.parse_args()


def aggregate_chunk(chunk: pd.DataFrame, freq: str) -> pd.DataFrame:
    mid = (chunk["Ask"] + chunk["Bid"]) / 2
    chunk = chunk.assign(Mid=mid)
    chunk = chunk.set_index("Time")
    ohlcv = chunk["Mid"].resample(freq).ohlc()
    vol = chunk["Volume"].resample(freq).sum()
    ohlcv["volume"] = vol
    return ohlcv.dropna(subset=["open"])


def main() -> None:
    args = parse_args()
    freq = TF_MAP[args.tf]
    src = DATA_DIR / args.src

    if not src.is_file():
        raise FileNotFoundError(f"Нет файла: {src}")

    out_name = src.stem + f"_{args.tf}.parquet"
    out_path = DATA_DIR / out_name

    print(f"Читаем: {src}")
    print(f"Таймфрейм: {args.tf} ({freq}), кусками по {CHUNK:,} строк")
    print(f"Результат: {out_path}")

    reader = pd.read_csv(
        src,
        dtype={"Time": str},
        chunksize=CHUNK,
    )

    parts: list[pd.DataFrame] = []
    for i, chunk in enumerate(reader):
        chunk["Time"] = pd.to_datetime(chunk["Time"], format="%Y.%m.%d %H:%M:%S")
        chunk = chunk.sort_values("Time")
        parts.append(aggregate_chunk(chunk, freq))
        print(f"  обработано кусков: {i + 1}", end="\r")

    print()
    print("Объединяем и финальная агрегация...")
    combined = pd.concat(parts).sort_index()

    # финальная агрегация (стыки кусков могут дать дубли)
    final = (
        combined.resample(freq)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open"])
    )

    final.to_parquet(out_path)
    print(f"Готово. Строк: {len(final):,}")
    print(final.head())


if __name__ == "__main__":
    main()
