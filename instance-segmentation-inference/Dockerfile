FROM python:3.6

RUN pip install cmake && \
    pip install \
    numpy==1.18.1 \
    opencv-python==3.4.0.12 \
    keras==2.3.0 \
    tensorflow==1.14.0 \
    dlib==19.19.0
RUN pip install flask pillow
    

COPY . /app/
WORKDIR /app
RUN mv test.py backend/ && \
    mv model.py backend/

EXPOSE 80
ENTRYPOINT [ "python", "app.py" ]
