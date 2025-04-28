FROM nikosnikolaidis/skillab-ku-backend-base
COPY . /app/.
WORKDIR /app

RUN pip install -r /app/requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]