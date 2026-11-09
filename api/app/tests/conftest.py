import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # carpeta api/

from app.main import app  # noqa: E402
from app.schemas.predict import EJEMPLO  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def anuncio():
    return dict(EJEMPLO)
