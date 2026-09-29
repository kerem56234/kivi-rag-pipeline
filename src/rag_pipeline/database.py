# database.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from rag_pipeline.retriever import DatabaseClient, RetrieverError


class PsycopgDatabaseClient:
    """Synchronous PostgreSQL implementation of DatabaseClient."""

    def __init__(
        self,
        dsn: str,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        timeout: float = 10.0,
    ) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("dsn must be a non-empty string.")

        if (
            not isinstance(min_pool_size, int)
            or isinstance(min_pool_size, bool)
            or min_pool_size < 0
        ):
            raise ValueError(
                "min_pool_size must be a non-negative integer."
            )

        if (
            not isinstance(max_pool_size, int)
            or isinstance(max_pool_size, bool)
            or max_pool_size <= 0
        ):
            raise ValueError(
                "max_pool_size must be a positive integer."
            )

        if min_pool_size > max_pool_size:
            raise ValueError(
                "min_pool_size must not exceed max_pool_size."
            )

        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number.")

        self._closed = False

        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_pool_size,
            max_size=max_pool_size,
            timeout=float(timeout),
            open=False,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
            },
            configure=self._configure_connection,
        )

        try:
            self._pool.open(wait=True)
        except Exception as exc:
            self._pool.close()
            self._closed = True
            raise RetrieverError(
                "Failed to open the PostgreSQL connection pool."
            ) from exc

    @staticmethod
    def _configure_connection(
        connection: psycopg.Connection[Any],
    ) -> None:
        """Register pgvector adapters on each pooled connection."""
        register_vector(connection)

    def fetch_all(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a query and return all result rows as dictionaries."""
        if self._closed:
            raise RetrieverError(
                "Cannot execute a query because the database client is closed."
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        try:
            with self._pool.connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()

            return [dict(row) for row in rows]

        except psycopg.Error as exc:
            raise RetrieverError(
                "PostgreSQL query execution failed."
            ) from exc

    def close(self) -> None:
        """Close the connection pool."""
        if not self._closed:
            self._pool.close()
            self._closed = True

    def __enter__(self) -> "PsycopgDatabaseClient":
        if self._closed:
            raise RetrieverError(
                "Cannot enter a closed database client."
            )

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()