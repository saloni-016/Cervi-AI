FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8080 \
    MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-base.txt requirements-torch.txt ./
# CPU-only ML wheels reduce the image size on a host without a GPU.
RUN python -m pip install --no-cache-dir --only-binary=:all: pip==26.3 setuptools==84.1.0 \
    && python -m pip install --no-cache-dir --only-binary=:all: -r requirements-base.txt xgboost-cpu \
    && python -m pip install --no-cache-dir --only-binary=:all: \
         --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt \
    && python -m pip check
RUN useradd -m -U -u 10001 cerviai && chown cerviai:cerviai /app
COPY app.py explain_xgboost.py explain_vit.py train_vit.py ./
COPY model.pkl imputer.pkl vit_sipakmed.pth sipakmed_classes.json ./
COPY templates/*.html ./templates/
COPY static/*.css static/chatbot.js ./static/
COPY static/images/ai_doctor.png static/images/cervical_ai.png ./static/images/
USER cerviai
EXPOSE 8080
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 --access-logfile - app:app"]
