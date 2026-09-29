# test_database.py

from unittest.mock import MagicMock, patch

import psycopg
import pytest

from rag_pipeline.database import PsycopgDatabaseClient
from rag_pipeline.retriever import RetrieverError


@pytest.fixture
def pool_mock():
    with patch("rag_pipeline.database.ConnectionPool") as pool_class:
        pool = MagicMock()
        pool_class.return_value = pool

        connection = MagicMock()
        cursor = MagicMock()

        pool.connection.return_value.__enter__.return_value = connection
        connection.cursor.return_value.__enter__.return_value = cursor

        yield pool_class, pool, connection, cursor


def test_initializes_and_opens_connection_pool(pool_mock):
    pool_class, pool, _, _ = pool_mock

    client = PsycopgDatabaseClient(
        dsn="postgresql://user:password@localhost:5432/test_db",
        min_pool_size=2,
        max_pool_size=5,
        timeout=15.0,
    )

    pool_class.assert_called_once()
    kwargs = pool_class.call_args.kwargs

    assert kwargs["conninfo"] == (
        "postgresql://user:password@localhost:5432/test_db"
    )
    assert kwargs["min_size"] == 2
    assert kwargs["max_size"] == 5
    assert kwargs["timeout"] == 15.0
    assert kwargs["open"] is False
    assert kwargs["kwargs"]["autocommit"] is True

    pool.open.assert_called_once_with(wait=True)

    client.close()


def test_fetch_all_executes_query_and_returns_dictionaries(pool_mock):
    _, _, _, cursor = pool_mock

    cursor.fetchall.return_value = [
        {"id": 1, "chunk": "First chunk", "score": 0.1},
        {"id": 2, "chunk": "Second chunk", "score": 0.2},
    ]

    client = PsycopgDatabaseClient(
        dsn="postgresql://user:password@localhost:5432/test_db"
    )

    params = [[0.1, 0.2], 5]
    rows = client.fetch_all(
        "SELECT id, chunk, score FROM documents LIMIT %s",
        params,
    )

    cursor.execute.assert_called_once_with(
        "SELECT id, chunk, score FROM documents LIMIT %s",
        params,
    )

    assert rows == [
        {"id": 1, "chunk": "First chunk", "score": 0.1},
        {"id": 2, "chunk": "Second chunk", "score": 0.2},
    ]


def test_fetch_all_raises_retriever_error_on_database_error(pool_mock):
    _, _, _, cursor = pool_mock

    cursor.execute.side_effect = psycopg.OperationalError(
        "database unavailable"
    )

    client = PsycopgDatabaseClient(
        dsn="postgresql://user:password@localhost:5432/test_db"
    )

    with pytest.raises(
        RetrieverError,
        match="PostgreSQL query execution failed",
    ) as error:
        client.fetch_all("SELECT 1")

    assert isinstance(error.value.__cause__, psycopg.OperationalError)


def test_fetch_all_rejects_empty_query(pool_mock):
    client = PsycopgDatabaseClient(
        dsn="postgresql://user:password@localhost:5432/test_db"
    )

    with pytest.raises(
        ValueError,
        match="query must be a non-empty string",
    ):
        client.fetch_all("   ")


def test_fetch_all_rejects_closed_client(pool_mock):
    _, pool, _, _ = pool_mock

    client = PsycopgDatabaseClient(
        dsn="postgresql://user:password@localhost:5432/test_db"
    )
    client.close()

    with pytest.raises(
        RetrieverError,
        match="database client is closed",
    ):
        client.fetch_all("SELECT 1")

    pool.close.assert_called_once()


def test_context_manager_closes_pool(pool_mock):
    _, pool, _, _ = pool_mock

    with PsycopgDatabaseClient(
        dsn="postgresql://user:password@localhost:5432/test_db"
    ) as client:
        assert isinstance(client, PsycopgDatabaseClient)

    pool.close.assert_called_once()


def test_pool_open_failure_is_wrapped(pool_mock):
    _, pool, _, _ = pool_mock
    pool.open.side_effect = RuntimeError("pool unavailable")

    with pytest.raises(
        RetrieverError,
        match="Failed to open the PostgreSQL connection pool",
    ) as error:
        PsycopgDatabaseClient(
            dsn="postgresql://user:password@localhost:5432/test_db"
        )

    assert isinstance(error.value.__cause__, RuntimeError)
    pool.close.assert_called_once()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"dsn": ""}, "dsn must be a non-empty string"),
        (
            {
                "dsn": "postgresql://localhost/test",
                "min_pool_size": -1,
            },
            "min_pool_size must be a non-negative integer",
        ),
        (
            {
                "dsn": "postgresql://localhost/test",
                "max_pool_size": 0,
            },
            "max_pool_size must be a positive integer",
        ),
        (
            {
                "dsn": "postgresql://localhost/test",
                "min_pool_size": 5,
                "max_pool_size": 2,
            },
            "min_pool_size must not exceed max_pool_size",
        ),
        (
            {
                "dsn": "postgresql://localhost/test",
                "timeout": 0,
            },
            "timeout must be a positive number",
        ),
    ],
)
def test_rejects_invalid_configuration(arguments, message):
    with pytest.raises(ValueError, match=message):
        PsycopgDatabaseClient(**arguments)