from typing import Any, Optional

import psycopg
import psycopg.abc

from src.v2.config import get_config


async def execute_query(query: str, params: Optional[psycopg.abc.Params] = None) -> list[dict[str, Any]]:
    db_config = get_config().db

    connection_config = (
        f"dbname={db_config.dbname} "
        + f"user={db_config.user} "
        + f"password={db_config.password} "
        + f"host={db_config.host} "
        + f"port={db_config.port}"
    )

    async with await psycopg.AsyncConnection.connect(connection_config) as conn:
        await conn.set_read_only(True)
        async with conn.cursor() as cursor:
            await cursor.execute(query, params)
            result = await cursor.fetchall()
            if cursor.description is None:
                raise Exception(f"No description available for the query {query} and args {params}")
            columns = [desc[0] for desc in cursor.description]

    return [dict(zip(columns, row)) for row in result]
