FROM nikosnikolaidis/skillab-ku-backend-base
COPY . /.

RUN pip install -r /requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]