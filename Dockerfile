# Stage 1: πηγή — pre-stage το psql + ΟΛΕΣ τις runtime deps του σε ένα folder
FROM postgres:16-bullseye AS pg_source

RUN set -eux; \
    mkdir -p /pg_deps/lib; \
    cp /usr/lib/postgresql/16/bin/psql /pg_deps/psql; \
    # Auto-detect ALL shared libraries that psql links against, από οπουδήποτε
    # στο filesystem (/usr/lib, /lib, /lib64) — όχι χειροκίνητη λίστα.
    ldd /usr/lib/postgresql/16/bin/psql \
        | awk '/=>/ && $3 != "" {print $3}' \
        | xargs -I {} cp -L {} /pg_deps/lib/; \
    ls -la /pg_deps/lib/   # debug: δείξε τι μαζέψαμε

# Stage 2: το δικό μας backend image
FROM nikosnikolaidis/skillab-ku-backend-base

# Copy ONLY τα pre-staged files από το stage 1 (όχι random paths).
COPY --from=pg_source /pg_deps/psql /usr/local/bin/psql
COPY --from=pg_source /pg_deps/lib/ /usr/lib/x86_64-linux-gnu/

# Rebuild dynamic linker cache & sanity check
RUN ldconfig && psql --version

COPY . /app/.
WORKDIR /app

# Φάκελος για το HF cache (persistent μέσω docker volume)
RUN mkdir -p /app/.cache/huggingface

RUN pip install -r /app/requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]