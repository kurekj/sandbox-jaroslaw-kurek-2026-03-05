from sql.scripts import FIND_ALGOLYTICS_UUID, FIND_USER_UUID
from src.v1.web_recommendation.managers.db_manager import query_fetch


def find_user_uuid(user_id):
    """
    The funcion is searching for user_id based on session_id.
    :param user_id: id of user
    :return: uuid from km.users_user
    """
    user_uuid = query_fetch(FIND_USER_UUID, {"user_id": user_id})
    if len(user_uuid) > 0:
        return user_uuid[0][0]


def find_uuid(session_id):
    """
    The function is searching for uuid based on session_id.
    :param session_id: uuid of user
    :return: uuid from km.users_user if id is in the sceuid_map table.
    If not, returns input value - session_id
    """
    user_uuid = query_fetch(FIND_ALGOLYTICS_UUID, {"session_id": session_id})
    if len(user_uuid) > 0:
        return user_uuid[0][0]
    else:
        return session_id
