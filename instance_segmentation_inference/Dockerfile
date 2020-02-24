FROM python:3.6

RUN pip install cmake && \
    pip install \
    cython==0.29.15 \
    torch==1.4.0 \
    torchvision==0.5.0 \
    pycocotools==2.0.0 \
    matplotlib==3.1.3 \
    numpy==1.18.1 \
    opencv-python==4.2.0.32
RUN pip install flask pillow
    

COPY . /app/
WORKDIR /app
RUN mv test.py backend/ && \
    mv model.py backend/

EXPOSE 80
ENTRYPOINT [ "python", "app.py" ]

