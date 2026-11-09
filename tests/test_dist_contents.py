"""Build the sdist and wheel and assert what ships (supply-chain hygiene)."""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / "pyproject.toml").exists(),
    reason="needs a source checkout",
)


@pytest.fixture(scope="module")
def dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("hatchling")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(out), str(ROOT)],
        check=True,
        capture_output=True,
    )
    return out


def test_wheel_ships_only_the_package(dist: Path) -> None:
    (wheel,) = dist.glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    assert "llm_gateway/py.typed" in names
    for n in names:
        assert n.startswith(("llm_gateway/", "llm_gateway-")), n
        assert not n.startswith("llm_gateway/tests"), n
    assert any(n.endswith("licenses/LICENSE") for n in names)


def test_sdist_excludes_infra_secrets_and_ci(dist: Path) -> None:
    (sdist,) = dist.glob("*.tar.gz")
    names = tarfile.open(sdist).getnames()
    rel = ["/".join(n.split("/")[1:]) for n in names]
    banned_prefixes = ("infra/", ".github/", "examples/", "constraints/", "scripts/")
    banned_names = {"Dockerfile", ".env", ".env.example", "secrets.yaml", "secrets.enc.yaml"}
    for r in rel:
        assert not r.startswith(banned_prefixes), r
        assert Path(r).name not in banned_names, r
        assert not r.endswith((".pem", ".key", ".p12", ".tfstate")), r
    for needed in ("LICENSE", "SECURITY.md", "pyproject.toml", "src/llm_gateway/py.typed"):
        assert needed in rel
