FROM python:3.11-slim
WORKDIR /backtest
RUN pip install --no-cache-dir yfinance pandas numpy requests matplotlib
COPY backtests/ /backtest/
ENTRYPOINT ["python3", "/backtest/backtest_engine.py"]
