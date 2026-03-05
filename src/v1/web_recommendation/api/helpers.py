from src.v1.web_recommendation.api.tasks import generate_recommendations, push_prediction_error


def schedule_recommendations_generation(user, webhook_url, portal, logged_in):
    """Schedule recommendation generation for user.

    The task is executed asynchronously and returned. Errors are handled by push_prediction_error task.
    """
    task = generate_recommendations.apply_async(
        kwargs={"user": user, "webhook_url": webhook_url, "portal": portal, "logged_in": logged_in},
        link_error=push_prediction_error.s(webhook_url=webhook_url),
    )

    return task
