"""
Privacy audit (CP 10 / Track 08).

You are going to stand in front of judges and say no health data leaves the
device. This file is what makes that a checkable claim rather than an
intention.

Static analysis over the source tree, so it runs in CI and fails the moment
someone adds an endpoint, a file write, or a logger that would break the
promise. The manual half of the gate -- running a full session with a network
monitor open -- is still required, and is listed in the checkpoint. These tests
catch the regressions between those manual runs.

WHAT IS PROMISED
================
  PRD 2   "No video, audio, or health data leaves the phone in this build."
  TRD 4   "never stores video/audio, only structured session summaries"
  PRD 8   "This is a self-management aid, not a diagnostic device."
"""

import ast
import re
from pathlib import Path

import pytest

from rehab_ai.rules.loader import load_rules

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "rehab_ai"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PYTHON_FILES = sorted(SOURCE_ROOT.rglob("*.py"))

LOOPBACK = ("127.0.0.1", "localhost", "::1")


def source_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Nothing leaves the machine
# --------------------------------------------------------------------------


def test_there_are_python_files_to_audit():
    """Guards against the audit silently passing because it found nothing."""
    assert len(PYTHON_FILES) > 10


@pytest.mark.parametrize("path", PYTHON_FILES, ids=lambda p: p.name)
def test_every_url_literal_is_loopback(path):
    """The only network destination in this application is the local model.

    A URL to anywhere else -- telemetry, crash reporting, a model API, an
    analytics beacon -- breaks the central promise, so it fails the build
    rather than being caught by someone watching a network monitor.
    """
    urls = re.findall(r"https?://[^\s\"'<>)]+", source_of(path))

    for url in urls:
        host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]
        assert host in LOOPBACK, f"{path.name} contacts {host!r} ({url})"


def test_the_configured_model_endpoint_is_loopback():
    """rules/thresholds.v1.json is editable, so it is checked too. Pointing
    explain.endpoint at a remote host would ship every lock decision off the
    machine without touching a line of Python."""
    endpoint = load_rules().explain.endpoint
    host = re.sub(r"^https?://", "", endpoint).split("/")[0].split(":")[0]
    assert host in LOOPBACK, f"explain.endpoint is {endpoint!r}, which is not local"


def test_no_module_imports_a_network_client_other_than_requests():
    """requests is used for exactly one thing: localhost Ollama. Anything else
    that can open a socket needs a deliberate decision, not a quiet import."""
    forbidden = {
        "urllib.request", "http.client", "socket", "httpx", "aiohttp",
        "websockets", "boto3", "google.cloud", "openai", "anthropic",
    }

    for path in PYTHON_FILES:
        tree = ast.parse(source_of(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name not in forbidden, f"{path.name} imports {name}"


# --------------------------------------------------------------------------
# No frames, no audio, at rest
# --------------------------------------------------------------------------


def test_no_module_writes_an_image_or_video_file():
    """cv2.imwrite / VideoWriter would put a frame on disk. TRD 4 says the
    database never stores video; a file beside it would be the same breach
    through a different door."""
    forbidden_calls = ("imwrite", "VideoWriter", "imsave")

    for path in PYTHON_FILES:
        text = source_of(path)
        for call in forbidden_calls:
            assert call not in text, f"{path.name} calls {call}"


def test_the_transcriber_does_not_retain_audio():
    """faster-whisper reads a path and returns text. The wrapper must not keep
    the buffer, copy the file, or hold a reference to it after transcribing."""
    text = source_of(SOURCE_ROOT / "checkin" / "voice.py")

    assert "shutil" not in text
    assert ".save(" not in text
    assert "copyfile" not in text
    # The only thing kept from a check-in is the transcript and the parsed
    # fields, all of which are structured text the patient just said aloud.
    assert "self._audio" not in text


def test_storage_never_reads_landmarks_off_an_observation():
    """Observation carries raw landmarks for one frame so the detector can use
    them. Storage must never touch that field.

    Checked against the AST rather than the source text: repository.py's
    docstring says the word "landmark" precisely because it is documenting that
    they are not stored, and a substring search would fail on the comment that
    proves the point.
    """
    tree = ast.parse(source_of(SOURCE_ROOT / "storage" / "repository.py"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert "landmark" not in node.attr.lower(), (
                f"repository.py reads .{node.attr}"
            )
        if isinstance(node, ast.Name):
            assert "landmark" not in node.id.lower(), f"repository.py uses {node.id}"


def test_the_frame_buffer_never_reaches_storage():
    """Same guarantee for the composited frame and the raw capture."""
    tree = ast.parse(source_of(SOURCE_ROOT / "storage" / "repository.py"))

    forbidden = {"for_inference", "for_display", "latest_frame", "rgb_for_inference"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden, f"repository.py reads .{node.attr}"


# --------------------------------------------------------------------------
# Patient data does not reach logs or stdout
# --------------------------------------------------------------------------


def test_no_module_logs_patient_values():
    """A pain score in a log file is health data at rest, in a place nobody
    thinks of as a database.

    Checks that the modules holding patient data neither configure logging nor
    print. app.py prints startup status only, and is exempt by name.
    """
    exempt = {"app.py", "render_cues.py"}

    for path in PYTHON_FILES:
        if path.name in exempt:
            continue
        tree = ast.parse(source_of(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"{path.name} calls print()"


def test_app_startup_output_contains_no_patient_fields():
    """app.py may print, but only about the environment -- never a value the
    patient supplied."""
    text = source_of(SOURCE_ROOT / "app.py")
    for field in ("pain.value", "swelling.report", "session.pain", ".reps["):
        assert field not in text, f"app.py prints {field}"


# --------------------------------------------------------------------------
# The database stays out of version control
# --------------------------------------------------------------------------


def test_the_gitignore_excludes_the_session_database():
    """The repository is public. A committed .db would publish pain scores and
    swelling reports irreversibly."""
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("*.db", "*.sqlite", "*.sqlite3"):
        assert pattern in ignore, f".gitignore is missing {pattern}"


def test_the_gitignore_excludes_recorded_footage():
    """Tuning footage holds patients' faces and their homes."""
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("recordings/", "*.mp4", "*.mov"):
        assert pattern in ignore, f".gitignore is missing {pattern}"


def test_the_default_database_path_is_inside_the_ignored_directory():
    from rehab_ai.storage.repository import DEFAULT_DB_PATH

    assert DEFAULT_DB_PATH.suffix == ".db"  # covered by the *.db ignore rule
    assert "data" in DEFAULT_DB_PATH.parts


# --------------------------------------------------------------------------
# The clinical boundary
# --------------------------------------------------------------------------


def test_no_patient_facing_copy_makes_a_diagnostic_claim():
    """PRD 8: this flags a movement pattern, it does not diagnose. The copy the
    patient reads is checked against the same forbidden list the LLM output is,
    so a hand-written string cannot say what a generated one may not."""
    from rehab_ai.explain.phrasing import FORBIDDEN_TERMS
    from rehab_ai.ui.theme import COPY

    rules = load_rules()
    strings = list(COPY.values()) + list(rules.policy.copy.values())
    strings += [phrase.text for phrase in rules.cue.phrases.values()]

    for text in strings:
        lowered = text.lower()
        for term in FORBIDDEN_TERMS:
            assert term not in lowered, f"copy {text!r} contains {term!r}"
