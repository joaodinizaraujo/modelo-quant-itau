import json
from pathlib import Path

import httpx
import pytest

from modelo_quant.threads import BASE_URL, ThreadsClient

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _sem_token_do_dev(monkeypatch):
    """Impede que um .env real do desenvolvedor influencie os testes."""
    monkeypatch.delenv("THREADS_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("THREADS_WEB_DOC_ID", raising=False)


@pytest.fixture(autouse=True)
def _sem_sleep(monkeypatch):
    """Retry e rate limit nao devem deixar a suite lenta."""
    monkeypatch.setattr("time.sleep", lambda _s: None)


@pytest.fixture
def client():
    with ThreadsClient("test-token", client=httpx.Client(base_url=BASE_URL)) as c:
        yield c


@pytest.fixture
def fx():
    def _ler(nome: str) -> dict:
        return json.loads((FIXTURES / f"{nome}.json").read_text(encoding="utf-8"))

    return _ler
