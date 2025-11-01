# stage_1.py
"""
Этап 1. Взаимодействие с пользователем и загрузка котировок за 2 года (yfinance).

Что делает:
- Диалог: /getdata → бот просит тикер → затем сумму → загружает котировки за 2 года → присылает CSV и краткую сводку.
- Сохраняет CSV в data/ под именем user_<id>_<TICKER>_last_2y.csv.
- Подсказывает команду анализа: /analyze <TICKER> <AMOUNT> (для модулей Этапов 2–5).
- Работает в обычном скрипте и в средах с активным event loop (Jupyter/Spyder) — поддержан фоновый запуск.

Зависимости:
- python-telegram-bot==20.6
- yfinance
- pandas
- python-dotenv

Окружение:
- Создайте файл my_bot.env (или .env) рядом со скриптом:
  TELEGRAM_BOT_TOKEN=ВАШ_ТОКЕН

Подключение Этапов 2–5:
- Модуль stage_2_5.py должен лежать рядом. Регистрация его хэндлеров выполняется внутри build_app().

Автор: Медведев Игорь
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import InputFile, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

# ----------------------------- Константы и состояния -----------------------------

ASK_TICKER, ASK_AMOUNT = range(2)  # состояния ConversationHandler
LOOKBACK_YEARS: int = 2            # глубина истории (лет)

DATA_DIR: Path = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Логирование -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------------------------- Утилиты окружения -----------------------------

def get_script_dir() -> Path:
    """
    Возвращает директорию текущего скрипта. В интерактивных средах (где __file__ отсутствует) — текущую рабочую.
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def load_token_from_env() -> str:
    """
    Загружает TELEGRAM_BOT_TOKEN из файлов my_bot.env → .env → переменных окружения ОС (в таком порядке).
    Бросает RuntimeError, если токен не найден.
    """
    script_dir = get_script_dir()

    # 1) my_bot.env рядом со скриптом
    my_env = script_dir / "my_bot.env"
    if my_env.exists():
        load_dotenv(dotenv_path=my_env, override=True)
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            logging.info("Токен загружен из my_bot.env")
            return token

    # 2) .env рядом со скриптом
    default_env = script_dir / ".env"
    if default_env.exists():
        load_dotenv(dotenv_path=default_env, override=True)
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            logging.info("Токен загружен из .env")
            return token

    # 3) переменные окружения ОС
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if token:
        logging.info("Токен загружен из переменных окружения ОС")
        return token

    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN. Укажите его в my_bot.env/.env или окружении ОС.")

def is_event_loop_running() -> bool:
    """
    Возвращает True, если в текущем потоке уже запущен asyncio event loop (актуально для Jupyter/Spyder).
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.is_running()
    except RuntimeError:
        return False

async def run_bot_async(app: Application) -> None:
    """
    Асинхронный запуск бота для сред с активным event loop.
    Запускает initialize/start/start_polling и не блокирует текущий поток.
    """
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logging.info("Бот запущен (async, среда с активным event loop).")

# ----------------------------- Типы и загрузка данных -----------------------------

@dataclass
class PriceDataSummary:
    """
    Краткая сводка по загруженным котировкам.
    """
    ticker: str
    rows: int
    start_date: datetime
    end_date: datetime
    last_close: float

def load_last_two_years_history(ticker: str) -> Tuple[pd.DataFrame, PriceDataSummary]:
    """
    Загружает дневные котировки за последние 2 года для указанного тикера через yfinance.
    Возвращает:
      - DataFrame с колонками OHLCV (как у yfinance) и индексом datetime (без tz)
      - PriceDataSummary с агрегированной информацией
    Бросает ValueError, если данных недостаточно или тикер некорректен.
    """
    end = datetime.now(timezone.utc)
    # +7 дней — запас на выходные и праздники
    start = end - timedelta(days=365 * LOOKBACK_YEARS + 7)

    df = yf.download(
        tickers=ticker,
        start=start.date(),
        end=end.date(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df.empty:
        raise ValueError("Не удалось загрузить данные. Проверьте тикер или соединение.")

    df = df.dropna(how="all").copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if df.shape[0] < 30:
        raise ValueError("Слишком мало данных за период (меньше 30 дней).")

    # Предпочитаем Adj Close, если есть
    last_close = float(df.get("Adj Close", df.get("Close")).iloc[-1])

    summary = PriceDataSummary(
        ticker=ticker.upper(),
        rows=df.shape[0],
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        last_close=last_close,
    )
    return df, summary

# ----------------------------- Валидация ввода -----------------------------

def is_valid_ticker(text: str) -> bool:
    """
    Простая валидация тикера: латинские буквы/цифры, длина 1..10.
    """
    t = text.strip().upper()
    return t.isalnum() and (1 <= len(t) <= 10)

# ----------------------------- Хэндлеры Telegram -----------------------------

async def cmd_start(update: Update, context: CallbackContext) -> None:
    """
    Приветствие и краткая инструкция.
    """
    if update.message:
        await update.message.reply_text(
            "Привет! Я загружу котировки за последние два года.\n"
            "Начните командой /getdata"
        )

async def getdata_entry(update: Update, context: CallbackContext) -> int:
    """
    Точка входа в разговор: просим тикер.
    """
    if update.message:
        await update.message.reply_text(
            "Введите тикер компании (например, AAPL):",
            reply_markup=ReplyKeyboardRemove(),
        )
    return ASK_TICKER

async def ask_amount(update: Update, context: CallbackContext) -> int:
    """
    Принимаем тикер, валидируем и просим сумму для условной инвестиции.
    """
    if not update.message:
        return ASK_TICKER

    ticker = update.message.text.strip().upper()
    if not is_valid_ticker(ticker):
        await update.message.reply_text("Некорректный тикер. Пример: AAPL. Повторите ввод:")
        return ASK_TICKER

    context.user_data["ticker"] = ticker
    await update.message.reply_text(
        f"Тикер: {ticker}. Введите сумму для условной инвестиции в USD (например, 1000):"
    )
    return ASK_AMOUNT

async def load_and_reply(update: Update, context: CallbackContext) -> int:
    """
    Принимаем сумму, загружаем котировки через yfinance, сохраняем CSV в data/,
    отправляем CSV пользователю и подсказываем команду анализа.

    Имя файла: data/user_<user_id>_<TICKER>_last_2y.csv
    """
    if not update.message:
        return ConversationHandler.END

    # Парсим сумму
    text_val = update.message.text.strip().replace(",", ".")
    try:
        amount = float(text_val)
        if not pd.notna(amount) or amount <= 0:
            raise ValueError
    except Exception:
        await update.message.reply_text("Сумма должна быть положительным числом. Попробуйте снова:")
        return ASK_AMOUNT

    ticker: Optional[str] = context.user_data.get("ticker")
    if not ticker:
        await update.message.reply_text("Не удалось определить тикер. Начните заново: /getdata")
        return ConversationHandler.END

    await update.message.reply_text("Загружаю котировки за последние два года, подождите...")

    try:
        # offload I/O и преобразования в поток, чтобы не блокировать event loop
        df, summary = await asyncio.to_thread(load_last_two_years_history, ticker)
    except Exception as e:
        logging.exception("Ошибка загрузки данных: %s", e)
        await update.message.reply_text(f"Не удалось загрузить данные: {e}")
        return ConversationHandler.END

    # Сохраняем CSV локально для последующего анализа (Этапы 2–5)
    user_id = update.effective_user.id if update.effective_user else 0
    filename = f"user_{user_id}_{summary.ticker}_last_{LOOKBACK_YEARS}y.csv"
    local_path = DATA_DIR / filename
    df.to_csv(local_path, index=True)

    # Готовим CSV к отправке пользователю (в виде файла)
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=True)
    csv_bytes.seek(0)

    caption = (
        f"✅ Данные загружены для {summary.ticker}\n"
        f"- Дней: {summary.rows}\n"
        f"- Период: {summary.start_date.date()} — {summary.end_date.date()}\n"
        f"- Последняя цена: {summary.last_close:.2f} USD\n"
        f"- Условная сумма: {amount:.2f} USD\n\n"
        f"Теперь запустите анализ: /analyze {summary.ticker} {amount}"
    )

    await update.message.reply_document(
        document=InputFile(csv_bytes, filename=filename),
        caption=caption,
    )

    return ConversationHandler.END

async def cancel(update: Update, context: CallbackContext) -> int:
    """
    Отмена диалога.
    """
    if update.message:
        await update.message.reply_text("Диалог отменён.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

# ----------------------------- Обработчик необработанных ошибок -----------------------------

async def on_error(update: object, context) -> None:
    logging.exception("Unhandled error", exc_info=context.error)

# ----------------------------- Сборка приложения -----------------------------

def build_app(token: str) -> Application:
    """
    Создаёт и настраивает Telegram Application:
      - /start — приветствие
      - /getdata — разговор: тикер → сумма → выгрузка CSV
      - /cancel — отмена разговора

    Подключение Этапов 2–5 выполняется здесь (если рядом есть stage_2_5.py).
    """
    app: Application = ApplicationBuilder().token(token).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("getdata", getdata_entry)],
        states={
            ASK_TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_amount)],
            ASK_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, load_and_reply)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True,
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(conv)
    app.add_handler(CommandHandler("cancel", cancel))

    # Подключаем Этапы 2–5, если модуль доступен
    try:
        from stage_2_5 import register_stage2_5_handlers, register_optional_csv_handler
        register_stage2_5_handlers(app)        # команда /analyze <тикер> <сумма>
        register_optional_csv_handler(app)     # опционально: приём CSV от пользователя
        logging.info("Хэндлеры Этапов 2–5 подключены.")
    except Exception as e:
        logging.warning("Не удалось подключить Этапы 2–5: %s", e)

    # Глобальный обработчик ошибок
    app.add_error_handler(on_error)

    return app

def main() -> None:
    """
    Точка входа. Загружает токен, собирает приложение и запускает бота.
    В средах с активным event loop (Jupyter/Spyder) — запускает в фоне.
    В обычном сценарии PyCharm — запускает polling синхронно.
    """
    token: str = load_token_from_env()
    app: Application = build_app(token)

    if is_event_loop_running():
        # Среда с активным event loop — запускаем в фоне
        asyncio.get_running_loop().create_task(run_bot_async(app))
        logging.info("Бот запущен. Ожидание сообщений... (async)")
    else:
        logging.info("Бот запущен. Ожидание сообщений...")
        app.run_polling(close_loop=True)

if __name__ == "__main__":
    main()