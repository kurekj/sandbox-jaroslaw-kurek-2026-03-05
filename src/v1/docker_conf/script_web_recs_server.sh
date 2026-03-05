#!/bin/bash

poetry run python web_recommendation/managers/init_pickle_update.py
poetry run uwsgi --ini app.ini
