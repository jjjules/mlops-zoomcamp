FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "starter.py", "./" ]
RUN chmod +x starter.py

# ENTRYPOINT [ "/app/start.py", "2021", "4" ]
ENTRYPOINT [ "bash" ]
