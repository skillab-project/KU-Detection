FROM nikosnikolaidis/skillab-ku-backend-base

# Install postgresql-client από το PGDG repo ΧΩΡΙΣ gpg verification.
# Λόγος: το Debian bullseye-security repo είναι broken (404s) ακόμα και για
# gnupg/wget/ca-certificates, άρα δεν μπορούμε να εγκαταστήσουμε gnupg για
# να επαληθεύσουμε την PGDG signature. Το [trusted=yes] παρακάμπτει
# την επαλήθευση και φέρνει το psql binary απευθείας.
RUN set -eux; \
    echo "deb [trusted=yes] http://apt.postgresql.org/pub/repos/apt bullseye-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends postgresql-client-16 && \
    rm -rf /var/lib/apt/lists/* && \
    psql --version   # sanity check — hard-fail αν δεν βρεθεί

COPY . /app/.
WORKDIR /app

# Φάκελος για το HF cache (persistent μέσω docker volume)
RUN mkdir -p /app/.cache/huggingface

RUN pip install -r /app/requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]