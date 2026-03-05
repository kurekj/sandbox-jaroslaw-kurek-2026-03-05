FROM python:3.10.13-bullseye

RUN apt-get update && \
    apt-get clean && \
    mkdir /web_recommendation_rp

RUN pip install --upgrade pip poetry==1.6.1

RUN useradd -ms /bin/bash web_recs
USER web_recs

COPY --chown=web_recs pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root

WORKDIR /home/web_recs/web_recommendation_rp
ENV PYTHONPATH "${PYTHONPATH}:/home/web_recs/web_recommendation_rp"

COPY --chown=web_recs docker_conf/app.conf .
COPY --chown=web_recs docker_conf/app.ini .
COPY --chown=web_recs docker_conf/var.env .
COPY --chown=web_recs docker_conf/wsgi.py .
COPY --chown=web_recs --chmod=744 docker_conf/script_web_recs_server.sh .

COPY --chown=web_recs sql ./sql
COPY --chown=web_recs web_recommendation/ ./web_recommendation

EXPOSE 8000

CMD ["poetry", "run", "uwsgi", "--ini", "app.ini"]
