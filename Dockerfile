FROM nikosnikolaidis/skillab-ku-backend-base

# Εγκατάσταση postgresql-client — ΜΕ auto-retry για να αντιμετωπίζει
# τα διαστήματα που το Debian security repo δεν έχει cached τις νέες εκδόσεις.
# Το --fix-missing συνεχίζει ακόμα κι αν ένα mirror είναι stale.
RUN set -eux; \
    for i in 1 2 3; do \
        apt-get update && \
        apt-get install -y --no-install-recommends --fix-missing postgresql-client && \
        rm -rf /var/lib/apt/lists/* && \
        break || { echo "apt attempt $i failed, retrying..."; sleep 5; }; \
    done

COPY . /app/.
WORKDIR /app

# Φάκελος για το HF cache (persistent μέσω docker volume)
RUN mkdir -p /app/.cache/huggingface

RUN pip install -r /app/requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]
