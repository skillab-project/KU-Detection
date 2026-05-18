FROM nikosnikolaidis/skillab-ku-backend-base

# Εγκατάσταση postgresql-client (για το initialize_database που χρησιμοποιεί psql)
RUN apt-get update && \
    apt-get install -y --no-install-recommends postgresql-client && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/.
WORKDIR /app

RUN pip install -r /app/requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]