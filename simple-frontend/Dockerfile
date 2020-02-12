FROM python:3.6

WORKDIR /app
RUN pip install flask requests

COPY . /app/
EXPOSE 80
ENTRYPOINT [ "python", "app.py" ]
