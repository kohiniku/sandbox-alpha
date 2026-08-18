FROM python:3.11-slim
WORKDIR /backtest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Engine version handshake (issue #101): bake git SHA into the image
ARG GIT_REV=unknown
RUN echo "${GIT_REV}" > /backtest/git_rev

COPY backtests/ /backtest/
COPY manifest.py /backtest/
COPY manifest_runner.py /backtest/
COPY data_adapters/ /backtest/data_adapters/
COPY evaluators/ /backtest/evaluators/
RUN useradd --create-home --shell /bin/bash sandbox
USER sandbox
ENTRYPOINT ["python3", "/backtest/backtest_engine.py"]
