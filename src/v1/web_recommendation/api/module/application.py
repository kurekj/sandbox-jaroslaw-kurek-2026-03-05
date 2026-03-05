from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from src.v1.web_recommendation.consts import config

from .sentry import initialize_sentry

initialize_sentry()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = config.DB_ENGINE_STRING
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app, session_options={"autocommit": True})
