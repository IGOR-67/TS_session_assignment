# stage_2_5.py
"""
Этапы 2–5:
- Этап 2: обучение моделей (Ridge, RandomForest, ARIMA, LSTM*), выбор лучшей по RMSE
- Этап 3: прогноз по рабочим дням на HORIZON_DAYS, визуализация всех моделей на одном графике
- Этап 4: простая стратегия (покупка/продажа в локальных экстремумах) и оценка прибыли (по лучшей модели)
- Этап 5: логирование результатов

Особенности:
- Прогнозы строятся по рабочим дням (pandas.bdate_range).
- Визуализация: история + все модели на одном графике; лучшая — выделена.
- LSTM опциональна. Если torch недоступен — модель пропускается.
- Исправлена ошибка LSTM при авто-прогнозе: удалён лишний unsqueeze, вход строго 3D [1, T, 1].

Совместимо с python-telegram-bot >= 20.6.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# Рендер без GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import InputFile, Update
from telegram.ext import CommandHandler, ContextTypes, MessageHandler, filters

# ================== Константы ==================

HORIZON_DAYS: int = 30          # горизонт прогноза (рабочих дней)
TEST_SIZE: int = 60             # окно теста для метрик
DATA_DIR: Path = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

LOG_PATH: str = "logs.txt"
SEED: int = 42

# Цвета/стили для моделей на графике
MODEL_STYLES: Dict[str, Dict[str, Any]] = {
    "ridge": {"color": "#d62728", "linestyle": "-", "label": "Ridge"},
    "random_forest": {"color": "#2ca02c", "linestyle": "--", "label": "RandomForest"},
    "arima": {"color": "#9467bd", "linestyle": "-.", "label": "ARIMA(1,1,1)"},
    "lstm": {"color": "#ff7f0e", "linestyle": ":", "label": "LSTM"},
}

# ================== Чтение CSV ==================

def _is_iso_date(s: str) -> bool:
    return isinstance(s, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", s.strip()) is not None


def _load_series_from_multiline_header_csv(path: Path) -> pd.Series:
    raw = pd.read_csv(path, header=None)
    if raw.empty:
        raise ValueError("CSV пустой.")

    date_row_idx: Optional[int] = None
    for i in range(min(10, len(raw))):
        val = str(raw.iat[i, 0]) if not pd.isna(raw.iat[i, 0]) else ""
        if val.strip().lower() == "date":
            date_row_idx = i
            break
    if date_row_idx is None:
        for i in range(len(raw)):
            val = raw.iat[i, 0]
            if isinstance(val, str) and _is_iso_date(val):
                date_row_idx = i - 1
                break
        if date_row_idx is None:
            raise ValueError("Не удалось определить 'Date' или начало данных.")

    header_metrics: List[str] = [str(x).strip() for x in list(raw.iloc[0].values)]
    metrics_clean: List[str] = [m for m in header_metrics if m and m.lower() != "nan"]
    columns: List[str] = ["Date"] + metrics_clean

    data = raw.iloc[date_row_idx + 1:].reset_index(drop=True).copy()
    max_cols = max(len(columns), data.shape[1])
    if data.shape[1] < max_cols:
        for _ in range(max_cols - data.shape[1]):
            data[data.shape[1]] = np.nan
    elif data.shape[1] > max_cols:
        data = data.iloc[:, :max_cols]
    data.columns = columns[:data.shape[1]]

    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d", errors="coerce")
    data = data.dropna(subset=["Date"]).set_index("Date").sort_index()
    for c in data.columns:
        if c != "Date":
            data[c] = pd.to_numeric(data[c], errors="coerce")

    for col in ("Adj Close", "Close", "Price"):
        if col in data.columns:
            s = data[col].dropna()
            if not s.empty:
                return s.astype(float)

    raise ValueError("В CSV нет 'Adj Close'/'Close'/'Price'.")


def load_series_from_csv(path: Path) -> pd.Series:
    try:
        df = pd.read_csv(path, parse_dates=[0], date_format="%Y-%m-%d", index_col=0)
        df = df.sort_index()
        for col in ("Adj Close", "Close", "Price"):
            if col in df.columns:
                s = df[col].astype(float).dropna()
                if not s.empty:
                    return s
        return _load_series_from_multiline_header_csv(path)
    except Exception:
        return _load_series_from_multiline_header_csv(path)


def resolve_user_csv(user_id: int, ticker: Optional[str]) -> Path:
    if ticker:
        p = DATA_DIR / f"user_{user_id}_{ticker.upper()}_last_2y.csv"
        if p.exists():
            return p
    candidates = sorted(DATA_DIR.glob(f"user_{user_id}_*_last_2y.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("Не найден CSV. Сначала выполните /getdata и пришлите тикер и сумму.")

# ================== Метрики и фичи ==================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def make_lag_features(series: pd.Series, lags: int = 30) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    return df.dropna()

# ================== Модели ==================

@dataclass
class ModelResult:
    name: str
    rmse: float
    mape: float
    test_pred: pd.Series   # предсказания на тесте (для метрик)
    model: Any
    meta: Dict[str, Any]


def train_ridge(series: pd.Series, test_size: int, lags: int = 30) -> ModelResult:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = make_lag_features(series, lags)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    pred = pd.Series(y_pred, index=df.index[-test_size:])
    return ModelResult("ridge", rmse(y_test, y_pred), mape(y_test, y_pred), pred, pipe, {"lags": lags})


def train_rf(series: pd.Series, test_size: int, lags: int = 30) -> ModelResult:
    from sklearn.ensemble import RandomForestRegressor

    df = make_lag_features(series, lags)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    pred = pd.Series(y_pred, index=df.index[-test_size:])
    return ModelResult("random_forest", rmse(y_test, y_pred), mape(y_test, y_pred), pred, rf, {"lags": lags})


def train_arima(series: pd.Series, test_size: int) -> ModelResult:
    from statsmodels.tsa.arima.model import ARIMA

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    fit = ARIMA(train, order=(1, 1, 1)).fit()
    forecast = fit.forecast(steps=test.shape[0])
    forecast.index = test.index
    y_true, y_pred = test.values, forecast.values
    return ModelResult("arima", rmse(y_true, y_pred), mape(y_true, y_pred), forecast, fit, {})


def train_lstm(series: pd.Series, test_size: int, window: int = 30, epochs: int = 30, batch_size: int = 32) -> Optional[ModelResult]:
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler
    except Exception:
        return None

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    values = series.values.reshape(-1, 1).astype(np.float32)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).astype(np.float32)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    for i in range(window, len(scaled)):
        X_list.append(scaled[i - window:i, 0])
        y_list.append(scaled[i, 0])

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)
    idx_all = series.index[window:]

    X_train, X_test = X_all[:-test_size], X_all[-test_size:]
    y_train, y_test = y_all[:-test_size], y_all[-test_size:]
    idx_test = idx_all[-test_size:]

    X_train_t = torch.tensor(X_train).unsqueeze(-1).to(device)  # [N, window, 1]
    y_train_t = torch.tensor(y_train).unsqueeze(-1).to(device)
    X_test_t = torch.tensor(X_test).unsqueeze(-1).to(device)

    class LSTMReg(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 64) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)      # x: [B, T, 1]
            out = out[:, -1, :]        # [B, H]
            return self.fc(out)        # [B, 1]

    model = LSTMReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    n = X_train_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_test_scaled = model(X_test_t).cpu().numpy().reshape(-1, 1)

    y_pred_test = scaler.inverse_transform(y_pred_test_scaled).ravel()
    y_true_test = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    pred = pd.Series(y_pred_test, index=idx_test)
    return ModelResult("lstm", rmse(y_true_test, y_pred_test), mape(y_true_test, y_pred_test), pred, model,
                       {"scaler": scaler, "window": window, "device": device})

# ================== Прогнозы (рабочие дни) ==================

def forecast_future_for_model(res: ModelResult, series: pd.Series, horizon: int) -> pd.Series:
    """
    Прогнозирует ряд для конкретной модели на horizon рабочих дней.
    Возвращает Series с индексом рабочих дат.
    """
    last_date: pd.Timestamp = pd.Timestamp(series.index.max())
    start_date: pd.Timestamp = last_date + BDay(1)
    idx: pd.DatetimeIndex = pd.bdate_range(start=start_date, periods=horizon)

    if res.name in ("ridge", "random_forest"):
        lags: int = int(res.meta["lags"])
        hist = series.copy()
        preds: List[float] = []
        current_date = start_date
        for _ in range(horizon):
            feat_row = [hist.iloc[-i] for i in range(1, lags + 1)]
            X_next = np.array(feat_row, dtype=float).reshape(1, -1)
            y_next = float(res.model.predict(X_next)[0])
            preds.append(y_next)
            hist.loc[current_date] = y_next
            current_date = current_date + BDay(1)
        return pd.Series(preds, index=idx, name=f"y_future_pred_{res.name}")

    if res.name == "arima":
        fc = res.model.forecast(steps=horizon)
        return pd.Series(np.asarray(fc), index=idx, name="y_future_pred_arima")

    if res.name == "lstm":
        # ВАЖНО: вход LSTM должен быть 3D [1, T, 1]; убираем лишние измерения
        import torch
        scaler = res.meta["scaler"]
        window: int = int(res.meta["window"])
        device = res.meta["device"]

        values = series.values.reshape(-1, 1).astype(np.float32)
        scaled = scaler.transform(values).astype(np.float32)

        # Последнее окно: строго [1, window, 1]
        seq = torch.tensor(scaled[-window:], dtype=torch.float32, device=device).view(1, window, 1)

        model = res.model
        model.eval()
        preds_scaled: List[float] = []

        with torch.no_grad():
            for _ in range(horizon):
                y_hat = model(seq)  # [1, 1]
                preds_scaled.append(float(y_hat.squeeze().cpu().numpy()))
                # Следующее значение в форму [1,1,1], без лишних осей
                next_val = y_hat.view(1, 1, 1)
                # Сдвигаем окно по времени и добавляем новый шаг
                seq = torch.cat([seq[:, 1:, :], next_val], dim=1)

        preds = scaler.inverse_transform(np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)).ravel()
        return pd.Series(preds, index=idx, name="y_future_pred_lstm")

    raise ValueError(f"Неизвестная модель: {res.name}")


def train_all_and_select_best(series: pd.Series, test_size: int = TEST_SIZE) -> Tuple[ModelResult, Dict[str, ModelResult]]:
    results: Dict[str, ModelResult] = {}
    res_ridge = train_ridge(series, test_size, lags=30); results[res_ridge.name] = res_ridge
    res_rf    = train_rf(series, test_size, lags=30);    results[res_rf.name]    = res_rf
    res_arima = train_arima(series, test_size);          results[res_arima.name] = res_arima
    res_lstm  = train_lstm(series, test_size, window=30, epochs=30, batch_size=32)
    if res_lstm is not None:
        results[res_lstm.name] = res_lstm
    best = min(results.values(), key=lambda r: r.rmse)
    return best, results

# ================== Визуализация ==================

def plot_all_models(history: pd.Series, forecasts: Dict[str, pd.Series], best_name: str) -> bytes:
    """
    Строит общий график: история + все прогнозы моделей.
    Возвращает PNG как bytes.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    history.plot(ax=ax, color="#1f77b4", linewidth=1.8, label="История")

    for name, fc in forecasts.items():
        style = MODEL_STYLES.get(name, {})
        label = style.get("label", name)
        color = style.get("color", None)
        ls = style.get("linestyle", "-")
        lw = 2.2 if name == best_name else 1.8
        ax.plot(fc.index, fc.values, linestyle=ls, color=color, linewidth=lw,
                label=f"{label}{' (best)' if name == best_name else ''}")

    ax.set_title(f"Прогноз на {len(next(iter(forecasts.values())))} рабочих дней (все модели)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axvline(history.index.max(), color="gray", linestyle="--", alpha=0.7)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_summary(history: pd.Series, forecast: pd.Series, model_name: str) -> str:
    last_actual = float(history.iloc[-1])
    last_forecast = float(forecast.iloc[-1])
    delta_abs = last_forecast - last_actual
    delta_pct = (delta_abs / last_actual) * 100.0 if last_actual != 0 else np.nan
    direction = "вырастут" if delta_abs > 0 else ("упадут" if delta_abs < 0 else "не изменятся")
    sign = "" if delta_abs < 0 else "+"
    return (
        f"Модель: {model_name}\n"
        f"Горизонт: {len(forecast)} рабочих дней\n"
        f"Текущая цена: {last_actual:.2f}\n"
        f"Ожидается, что через {len(forecast)} рабочих дней акции {direction} на {sign}{abs(delta_abs):.2f} "
        f"({sign}{abs(delta_pct):.2f}%)."
    )

# ================== Стратегия и логирование ==================

def find_local_extrema(s: pd.Series) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    mins: List[pd.Timestamp] = []
    maxs: List[pd.Timestamp] = []
    for i in range(1, len(s) - 1):
        if s.iloc[i] < s.iloc[i - 1] and s.iloc[i] < s.iloc[i + 1]:
            mins.append(s.index[i])
        if s.iloc[i] > s.iloc[i - 1] and s.iloc[i] > s.iloc[i + 1]:
            maxs.append(s.index[i])
    return mins, maxs


def simulate_strategy(history_last_price: float, forecast: pd.Series, amount: float) -> Tuple[float, str]:
    if amount <= 0:
        return 0.0, "Сумма не указана или равна 0 — расчёт прибыли пропущен."

    mins, maxs = find_local_extrema(forecast)
    min_set, max_set = set(mins), set(maxs)

    cash: float = amount
    shares: float = 0.0
    trades: List[str] = []

    for dt, price in forecast.items():
        p = float(price)
        if shares == 0 and dt in min_set:
            shares = cash / p
            cash = 0.0
            trades.append(f"Покупка {shares:.4f} по {p:.2f} на {dt.date()}")
        elif shares > 0 and dt in max_set:
            cash = shares * p
            trades.append(f"Продажа {shares:.4f} по {p:.2f} на {dt.date()}")
            shares = 0.0

    if shares > 0:
        end_price = float(forecast.iloc[-1])
        cash = shares * end_price
        trades.append(f"Закрытие позиции по {end_price:.2f} на {forecast.index[-1].date()}")
        shares = 0.0

    profit = cash - amount
    descr = "Сделки:\n" + ("\n".join(trades) if trades else "Подходящих локальных экстремумов не найдено.")
    return float(profit), descr


def append_log(user_id: int, ticker: str, amount: float, best: ModelResult, profit: float) -> None:
    from datetime import datetime, timezone
    dt = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = (
        f"{dt}\tuser_id={user_id}\t"
        f"ticker={ticker}\tamount={amount}\t"
        f"best_model={best.name}\tRMSE={best.rmse:.4f}\tMAPE={best.mape:.2f}%\t"
        f"profit={profit:.2f}\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)

# ================== Telegram-хэндлеры ==================

def parse_args(args: List[str]) -> Tuple[str, float]:
    ticker: str = "Unknown"
    amount: float = 0.0
    if len(args) >= 1:
        ticker = args[0]
    if len(args) >= 2:
        try:
            amount = float(args[1])
        except Exception:
            amount = 0.0
    return ticker, amount


async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_id: int = update.effective_user.id if update.effective_user else 0
        ticker, amount = parse_args(context.args or [])

        csv_path: Path = resolve_user_csv(user_id, ticker if ticker != "Unknown" else None)
        series: pd.Series = load_series_from_csv(csv_path).sort_index()

        if len(series) < TEST_SIZE + 60:
            raise ValueError(f"Слишком короткий ряд ({len(series)}). Нужно >= {TEST_SIZE + 60} точек.")

        # Этап 2: обучение всех моделей и выбор лучшей
        best, all_results = train_all_and_select_best(series, test_size=TEST_SIZE)

        # Этап 3: прогноз по рабочим дням для каждой модели
        forecasts: Dict[str, pd.Series] = {}
        for name, res in all_results.items():
            forecasts[name] = forecast_future_for_model(res, series, HORIZON_DAYS)

        # Общий график всех моделей
        png_all: bytes = plot_all_models(series, forecasts, best_name=best.name)

        # Краткая сводка по лучшей модели
        summary: str = build_summary(series, forecasts[best.name], best.name)

        # Этап 4: стратегия и прибыль по лучшей модели
        profit, trade_descr = simulate_strategy(float(series.iloc[-1]), forecasts[best.name], amount)
        reco: str = (
            "Инвестиционные рекомендации (на прогнозном участке):\n"
            "- Локальные минимумы: покупать\n- Локальные максимумы: продавать\n"
            f"{trade_descr}\nОриентировочная прибыль на сумму {amount:.2f}: {profit:.2f}"
        )

        # Этап 5: логирование
        append_log(user_id=user_id, ticker=ticker, amount=amount, best=best, profit=profit)

        # Ответы пользователю
        await update.message.reply_text(summary)
        await update.message.reply_photo(photo=InputFile(io.BytesIO(png_all), filename="forecast_all_models.png"))

        metrics_text = "Метрики моделей (меньше — лучше):\n" + "\n".join(
            f"- {m.name}: RMSE={m.rmse:.4f}, MAPE={m.mape:.2f}%"
            for m in sorted(all_results.values(), key=lambda r: r.rmse)
        )
        await update.message.reply_text(metrics_text)
        await update.message.reply_text(reco)

    except Exception as e:
        if update.message:
            await update.message.reply_text(f"Ошибка: {e}")


def register_stage2_5_handlers(app: Any) -> None:
    app.add_handler(CommandHandler("analyze", analyze_cmd))


# Опционально: загрузка CSV пользователем

async def handle_csv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not doc or not doc.file_name.lower().endswith(".csv"):
        await update.message.reply_text("Пришлите CSV-файл.")
        return
    file = await doc.get_file()
    user_id = update.effective_user.id if update.effective_user else 0
    save_name = DATA_DIR / f"user_{user_id}_history_custom.csv"
    await file.download_to_drive(str(save_name))
    await update.message.reply_text(f"CSV сохранён как {save_name.name}. Теперь используйте /analyze <тикер> <сумма>.")


def register_optional_csv_handler(app: Any) -> None:
    # ВАЖНО: PTB 20.6
    app.add_handler(MessageHandler(filters.Document.FileExtension("csv"), handle_csv))
