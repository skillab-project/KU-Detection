FROM nikosnikolaidis/skillab-ku-backend-base

# Install postgresql-client από το επίσημο PostgreSQL (PGDG) repo
# ώστε να παρακάμψουμε το Debian security repo που έχει 404 για postgresql-13.
# Αυτό το repo είναι bulletproof — ποτέ δεν ξεμένει από packages.
RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg \
        wget \
        ca-certificates \
        lsb-release && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor \
        | tee /usr/share/keyrings/pgdg.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/pgdg.gpg] http://apt.postgresql.org/pub/repos/apt bullseye-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends postgresql-client-16 && \
    rm -rf /var/lib/apt/lists/* && \
    psql --version   # sanity check — αν δεν βρει psql, FAIL το build (όχι silent)

COPY . /app/.
WORKDIR /app

# Φάκελος για το HF cache (persistent μέσω docker volume)
RUN mkdir -p /app/.cache/huggingface

RUN pip install -r /app/requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]