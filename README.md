# BTC TradingAI

Multi-model BTC/USD 1-minute prediction server with replay and paper trading UI.

Three TensorFlow Flatten+Net models (close, volume, high/low) predict 1 candle ahead based on 200 lookback candles with 10 features (close, high, low, volume, MA10, MA20, BB upper/lower, EMA10, OBV).

## Quick Start

```bash
pip install -r requirements.txt
python trading_server.py
```

Open http://localhost:8765/trading_ui.html

## Files

| File | Purpose |
|------|---------|
| `trading_server.py` | HTTP server on port 8765 |
| `trading_ui.html` | Web UI with replay, AI predictions, paper trading |
| `models/tradenetV3_btc.keras` | Trained close model |
| `models/tradenetV3_volume.keras` | Trained volume model |
| `models/tradenetV3_hl.keras` | Trained high/low model |

## Data

`btcusd_binance_enriched.csv` (253k rows, 2024-09-07 to 2025-03-02).

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/data?offset=&limit=` | OHLCV candle data |
| `/api/predict?idx=&n=` | AI prediction for n candles |
| `/api/export_csv?idx=&n=` | Export predictions as CSV |
| `/api/info` | Dataset info (total rows, window, models) |

## Disclaimer

For educational purposes only. Not financial advice.
