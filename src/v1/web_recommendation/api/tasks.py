import requests

from src.v1.web_recommendation.celery import app
from src.v1.web_recommendation.managers import data_generation
from src.v1.web_recommendation.model import prediction


@app.task(name="generate_recommendations")
def generate_recommendations(user, webhook_url, portal, logged_in):
    recommendations = prediction.main(user, portal, logged_in)

    push_prediction_results.delay(webhook_url=webhook_url, recommendations=recommendations)

    return recommendations


@app.task(
    name="push_prediction_results",
    autoretry_for=(requests.RequestException,),
    retry_backoff=2,
    max_retries=5,
)
def push_prediction_results(webhook_url, recommendations):
    """Send recommendations to webhook_url.

    Task can be automatically retried if webhook returns an error using exponential backoff.
    """
    requests.post(webhook_url, json={"success": True, "recommendations": recommendations})


@app.task(
    name="push_prediction_error",
    autoretry_for=(requests.RequestException,),
    retry_backoff=2,
    max_retries=5,
)
def push_prediction_error(request, exc, traceback, webhook_url):
    """Send an error message to webhook_url.

    Task can be automatically retried if webhook returns an error using exponential backoff.
    """
    requests.post(
        webhook_url,
        json={
            "success": False,
        },
    )


@app.task(name="update_aggregate_files")
def update_aggregates():
    data_generation.update_applications_views_dm()
