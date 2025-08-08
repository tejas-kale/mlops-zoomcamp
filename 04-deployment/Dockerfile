FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY ./pyproject.toml ./
COPY ./starter.py ./
RUN pip install uv

# Prevent uv from creating a virtual environment
RUN uv sync --dev

ENTRYPOINT ["uv", "run", "starter.py", "--year", "2023", "--month", "5"]