FROM nikosnikolaidis/skillab-ku-backend-base

# ΔΕΝ χρειαζόμαστε πλέον apt install — το seed loading γίνεται
# μέσω psycopg2 στο api/data_db.py (αντί για psql CLI).
# Αυτό παρακάμπτει το broken Debian bullseye-security repo (404s).

COPY . /app/.
WORKDIR /app

# Φάκελος για το HF cache (persistent μέσω docker volume)
RUN mkdir -p /app/.cache/huggingface

RUN pip install -r /app/requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]