
"""
bpla_pipeline.py — расчёт трендов по заявкам Минобороны РФ
и экспорт данных для веб-фронтенда.

Использование:
    python bpla_pipeline.py --input data.csv --outdir public/data
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Расчёты
# ----------------------------------------------------------------------

def rolling_quadratic(y: np.ndarray, window: int = 14):
    """
    Скользящая квадратичная регрессия окном `window`.

    Для каждого полного окна [i-window+1 .. i] решается
    y = a*x^2 + b*x + c  (x = 0..window-1) и возвращаются:
      level — значение кривой в последней точке окна,
      slope — производная в последней точке окна.

    Возвращает (level, slope) длиной len(y), с NaN в первых window-1 позициях.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)

    if n < window:
        return level, slope

    # Все окна сразу: (n-window+1, window)
    from numpy.lib.stride_tricks import sliding_window_view
    Yw = sliding_window_view(y, window)

    # Дизайн-матрица решается один раз для всех окон
    x = np.arange(window, dtype=float)
    X = np.stack([x ** 2, x, np.ones(window)], axis=1)
    # coefs[0]=a, coefs[1]=b, coefs[2]=c  — по столбцам окон
    coefs = np.linalg.lstsq(X, Yw.T, rcond=None)[0]

    last = window - 1
    level[window - 1:] = coefs[0] * last ** 2 + coefs[1] * last + coefs[2]
    slope[window - 1:] = 2 * coefs[0] * last + coefs[1]
    return level, slope


def whole_period_trend(y: np.ndarray):
    """Квадратичный тренд всего периода + наклон в последней точке."""
    x = np.arange(len(y), dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    trend = np.polyval([a, b, c], x)
    slope_end = 2 * a * x[-1] + b
    return trend, slope_end


def compute(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    y = df["bpla_24h"].to_numpy(float)

    level, slope = rolling_quadratic(y, window)
    trend, slope_end = whole_period_trend(y)

    df["rolling_level"] = level
    df["rolling_slope"] = slope
    df["whole_trend"] = trend
    df.attrs["whole_slope_end"] = slope_end
    return df


# ----------------------------------------------------------------------
# Экспорт для веб-фронтенда
# ----------------------------------------------------------------------

def export_json(df: pd.DataFrame, path: Path):
    """JSON, понятный JS без pandas-типов. NaN/inf -> null."""
    records = []
    for r in df.to_dict(orient="records"):
        rec = {}
        for k, v in r.items():
            if isinstance(v, float) and not math.isfinite(v):
                rec[k] = None          # None -> null в JSON
            else:
                rec[k] = v
        rec["date"] = pd.Timestamp(rec["date"]).strftime("%Y-%m-%d")
        records.append(rec)
    payload = {
        "meta": {
            "window": 14,
            "whole_slope_end": df.attrs["whole_slope_end"],
            "updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        },
        "data": records,
    }
    # allow_nan=False: если где-то остался NaN — упадём с ошибкой, а не отдадим битый JSON
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                               allow_nan=False), encoding="utf-8")


# ----------------------------------------------------------------------
# График (matplotlib, для отчётов; на сайте — Plotly)
# ----------------------------------------------------------------------

def plot(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(14.5, 8.2))

    ax.plot(df["date"], df["bpla_24h"], alpha=0.25, lw=1.2,
            marker="o", ms=3, label="Суточная сводка Минобороны РФ")
    ax.plot(df["date"], df["rolling_level"], lw=3.2,
            label="Скользящая квадратичная регрессия (14 дн.)")
    ax.plot(df["date"], df["whole_trend"], lw=2, ls="--",
            label="Квадратичный тренд всего периода")

    rubicon = pd.Timestamp("2026-08-13")
    ax.axvline(rubicon, ls=":", lw=2.2, alpha=0.9)
    ax.text(rubicon + pd.Timedelta(hours=8), 0.965,
            "13 августа 2026\nусловный «Рубикон»",
            transform=ax.get_xaxis_transform(), rotation=90,
            va="top", ha="left", fontsize=10, fontweight="bold")

    last = df.iloc[-1]
    arr = float(np.clip(abs(last["rolling_slope"]) * 1.6, 25, 120))
    arr = arr if last["rolling_slope"] > 0 else -arr
    ax.annotate("", xy=(last["date"] + pd.Timedelta(days=2),
                        last["rolling_level"] + arr),
                xytext=(last["date"], last["rolling_level"]),
                arrowprops=dict(arrowstyle="->", lw=2.8))

    info = (f'На {last["date"]:%d %B %Y}:\n'
            f'14-дн. уровень: {last["rolling_level"]:.1f} БПЛА/сутки\n'
            f'14-дн. вектор: {last["rolling_slope"]:+.1f} БПЛА/сутки\n'
            f'Долгосрочный наклон: {df.attrs["whole_slope_end"]:+.1f} БПЛА/сутки')
    ax.text(0.985, 0.965, info, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=.45", facecolor="white", alpha=0.88))

    ax.set_title("БПЛА, заявленные Минобороны РФ как уничтоженные ПВО\n"
                 "Динамика и оценочный «вектор»", fontsize=15, pad=14)
    ax.set_ylabel("БПЛА за 24-часовое оперативное окно")
    ax.set_xlabel("Дата окончания оперативного окна")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(df["date"].iloc[0] - pd.Timedelta(days=1),
                df["date"].iloc[-1] + pd.Timedelta(days=3))
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="public/data")
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--no-plot", action="store_true", help="не строить PNG")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["date"])
    df = compute(df, args.window)

    df.to_csv(outdir / "bpla_daily.csv", index=False, encoding="utf-8-sig")
    export_json(df, outdir / "bpla_daily.json")
    if not args.no_plot:
        plot(df, outdir / "bpla_daily.png")

    print(f"OK: {len(df)} строк; вектор {df['rolling_slope'].iloc[-1]:+.1f}")


if __name__ == "__main__":
    main()
