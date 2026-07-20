FROM python:3.11-slim
WORKDIR /backtest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt
COPY backtests/ /backtest/
COPY manifest.py /backtest/
COPY manifest_runner.py /backtest/
RUN useradd --create-home --shell /bin/bash sandbox
USER sandbox
ENTRYPOINT ["python3", "/backtest/backtest_engine.py"]
