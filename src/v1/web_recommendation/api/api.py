from flask import request

from src.v1.web_recommendation.api.module.application import app
from src.v1.web_recommendation.managers.redis_up_to_date import set_redis_current_date

from .helpers import schedule_recommendations_generation
from .tasks import update_aggregates


@app.route("/web-recommendation/update-agg", methods=["GET"])
def update_agg():
    set_redis_current_date()
    update_aggregates.delay()
    return "Updated"


@app.route("/web-recommendation/user", methods=["POST"])
def fetch_users():
    """Fetch recommendations for a user.

    This endpoint is used to fetch recommendations for a user. It accepts a POST request with the following parameters:
    - user_id: user identifier
    - logged_in: whether the user is logged in or not
    - webhook_url: URL to which the results will be sent
    - portal: optional parameter.
    In default is set as both, but there is an option to select only recommendations for rp or gh


    Input parameters are validated and a task is scheduled to generate recommendations. Any errors during the
    recommendation generation process are sent asynchronously to the webhook_url.

    It returns a JSON object with the following fields:
    - success: whether the request was successful
    - message: error message if success is False
    """
    user_id = request.form.get("user_id")
    logged_in = request.form.get("logged_in")
    webhook_url = request.form.get("webhook_url")
    portal = request.form.get("portal")

    if not user_id:
        return {"success": False, "message": "user_id is required"}, 400

    if not logged_in:
        return {"success": False, "message": "logged_in is required"}, 400

    if not webhook_url:
        return {"success": False, "message": "webhook_url is required"}, 400

    if not portal:
        portal = "both"
    elif portal not in ("both", "rp", "gh"):
        return {"success": False, "message": "portal is incorrect"}, 400

    try:
        logged_in = {
            "true": True,
            "false": False,
        }[logged_in.lower()]
    except KeyError:
        return {"success": False, "message": "logged_in must be a boolean"}, 400

    schedule_recommendations_generation(user=user_id, logged_in=logged_in, portal=portal, webhook_url=webhook_url)

    return {"success": True}, 200


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint used by Kubernetes probes."""
    return "pong"


if __name__ == "__main__":
    app.run(port=8000, debug=True, use_reloader=False)
