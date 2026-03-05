import psycopg2

from src.v1.web_recommendation.consts import config


def query_fetch(sql, params=None, db="RP"):
    conn = psycopg2.connect(config.DB[db])
    cursor = conn.cursor()
    if params is None:
        cursor.execute(sql)
    else:
        cursor.execute(sql, params)
    sql_output = cursor.fetchall()
    cursor.close()
    conn.close()
    return sql_output
