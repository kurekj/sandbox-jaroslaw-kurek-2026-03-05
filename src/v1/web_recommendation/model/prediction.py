import logging

from src.v1.web_recommendation.consts.paths import DIR_CONST_DATA_EXEC_TIME_LOGS
from src.v1.web_recommendation.managers.data_generation import is_datamart_up_to_date
from src.v1.web_recommendation.managers.manager import find_user_uuid, find_uuid
from src.v1.web_recommendation.model.gh_model import get_best_properties_from_gh
from src.v1.web_recommendation.model.rp_model import get_best_properties_from_rp


def main(user_id, portal, logged_in):
    """
    The function verifies whether the entered user ID is valid. If it is an integer, we assume it refers to an
    ID from the "km.users_user" database table. Otherwise, we try to assign a UUID for users who are not logged in
    and prepare recommendations for them.
    Currently, we are fetching 5 properties from the primary market RynekPierwotny.pl and 2 properties from Gethome.
    :param user_id: id of the km.users_user if user is looged in, session_id if isn't
    :param logged_in: If true the function will search for the id from km.users_user if false, will be looking for
    the session_id from the tracker for unauthenticated users.
    :param portal: here it is possible to select the portal for which the recommendations will be generated.
        In default is set as both, but there is an option to select only rp or gh
    :return: TOP 5 recommendations from RynekPierwotny.pl and TOP 2 from Gethome
    """
    logging.basicConfig(
        filename=DIR_CONST_DATA_EXEC_TIME_LOGS,
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    is_datamart_up_to_date()

    user_uuid = find_user_uuid(user_id) if logged_in else find_uuid(user_id)
    if user_uuid is not None:
        recommendation = {
            "rp_properties": get_best_properties_from_rp(user_uuid) if portal in ("both", "rp") else None,
            "gh_properties": get_best_properties_from_gh(user_uuid) if portal in ("both", "gh") else None,
        }
        return recommendation
