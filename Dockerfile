# Stage 1: πηγή — επίσημο postgres image με psql pre-installed
FROM postgres:16-bullseye AS pg_source

# Stage 2: το δικό μας backend image
FROM nikosnikolaidis/skillab-ku-backend-base

# Αντιγραφή του psql binary και ΟΛΩΝ των shared libraries που χρειάζεται,
# απευθείας από το postgres:16-bullseye image. Και τα δύο images είναι
# Debian bullseye, άρα binary compatibility είναι εγγυημένο.
# Αυτό παρακάμπτει εντελώς το broken apt (Fastly CDN 404s σε debian-security και PGDG).
COPY --from=pg_source /usr/lib/postgresql/16/bin/psql /usr/local/bin/psql
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libpq.so.5 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libldap-2.4.so.2 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/liblber-2.4.so.2 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libsasl2.so.2 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libgssapi_krb5.so.2 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libkrb5.so.3 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libk5crypto.so.3 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libcom_err.so.2 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libkrb5support.so.0 /usr/lib/x86_64-linux-gnu/
COPY --from=pg_source /usr/lib/x86_64-linux-gnu/libkeyutils.so.1 /usr/lib/x86_64-linux-gnu/

# Rebuild ldconfig cache & sanity check — hard fail αν δεν τρέξει το psql
RUN ldconfig && psql --version

COPY . /app/.
WORKDIR /app

# Φάκελος για το HF cache (persistent μέσω docker volume)
RUN mkdir -p /app/.cache/huggingface

RUN pip install -r /app/requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]