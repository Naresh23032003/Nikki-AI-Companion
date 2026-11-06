import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from app.config import CONFIG_PATH, load_settings

ROOT = Path(__file__).resolve().parent.parent


class ConfigEnvTests(unittest.TestCase):
    def setUp(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.yaml_data = yaml.safe_load(f) or {}
        self.yaml_extract_model = (
            self.yaml_data.get("ollama", {}).get("extract_model")
            or self.yaml_data.get("ollama", {}).get("model")
        )

    def test_extract_model_follows_ollama_model_when_extract_not_set(self):
        # Given OLLAMA_MODEL=llama3.2:3b and no OLLAMA_EXTRACT_MODEL,
        # When load_settings() runs, Then ollama_extract_model == "llama3.2:3b".
        env = dict(os.environ)
        env["OLLAMA_MODEL"] = "llama3.2:3b"
        env.pop("OLLAMA_EXTRACT_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()
            self.assertEqual(settings.ollama_model, "llama3.2:3b")
            self.assertEqual(settings.ollama_extract_model, "llama3.2:3b")

    def test_extract_model_override_wins_over_ollama_model(self):
        # Given OLLAMA_EXTRACT_MODEL=llama3.2:1b and OLLAMA_MODEL=llama3.2:3b,
        # When load_settings() runs, Then ollama_extract_model == "llama3.2:1b".
        env = dict(os.environ)
        env["OLLAMA_MODEL"] = "llama3.2:3b"
        env["OLLAMA_EXTRACT_MODEL"] = "llama3.2:1b"
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()
            self.assertEqual(settings.ollama_model, "llama3.2:3b")
            self.assertEqual(settings.ollama_extract_model, "llama3.2:1b")

    def test_no_model_env_vars_preserves_config_yaml_value(self):
        # Given no model env vars,
        # When load_settings() runs, Then ollama_extract_model equals extract_model value in config.yaml.
        env = dict(os.environ)
        env.pop("OLLAMA_MODEL", None)
        env.pop("OLLAMA_EXTRACT_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()
            self.assertEqual(settings.ollama_extract_model, self.yaml_extract_model)


class EntrypointPullTests(unittest.TestCase):
    def _run_entrypoint_with_mock_commands(self, env_vars: dict[str, str]) -> list[str]:
        """Runs docker/entrypoint.sh with mocked curl, uvicorn, and sleep commands,
        returning the arguments passed to curl during model pulls."""
        entrypoint_script = ROOT / "docker" / "entrypoint.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            recorded_pulls = tmp_path / "pulls.log"

            # Mock curl: handles /api/tags (returns success) and /api/pull (records the payload)
            mock_curl = tmp_path / "curl"
            mock_curl.write_text(
                f"""#!/usr/bin/env sh
for arg in "$@"; do
    if [ "$arg" = "http://ollama:11434/api/tags" ] || echo "$arg" | grep -q "/api/tags"; then
        exit 0
    fi
done
# Check for payload in -d
prev=""
for arg in "$@"; do
    if [ "$prev" = "-d" ]; then
        echo "$arg" >> "{recorded_pulls}"
    fi
    prev="$arg"
done
exit 0
""",
                encoding="utf-8",
            )
            mock_curl.chmod(0o755)

            # Mock uvicorn: exits cleanly without starting server
            mock_uvicorn = tmp_path / "uvicorn"
            mock_uvicorn.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
            mock_uvicorn.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{tmpdir}:{env.get('PATH', '')}"
            env.update(env_vars)

            res = subprocess.run(
                ["sh", str(entrypoint_script)],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"entrypoint failed: {res.stderr}")

            if recorded_pulls.exists():
                return recorded_pulls.read_text(encoding="utf-8").splitlines()
            return []

    def test_entrypoint_pulls_once_when_extract_equals_chat_model(self):
        pulls = self._run_entrypoint_with_mock_commands({
            "OLLAMA_MODEL": "llama3.2:3b",
            "OLLAMA_EMBED_MODEL": "nomic-embed-text",
            "OLLAMA_EXTRACT_MODEL": "llama3.2:3b",
        })
        self.assertEqual(
            pulls,
            ['{"name":"llama3.2:3b"}', '{"name":"nomic-embed-text"}'],
        )

    def test_entrypoint_pulls_extract_when_different(self):
        pulls = self._run_entrypoint_with_mock_commands({
            "OLLAMA_MODEL": "llama3.2:3b",
            "OLLAMA_EMBED_MODEL": "nomic-embed-text",
            "OLLAMA_EXTRACT_MODEL": "llama3.2:1b",
        })
        self.assertEqual(
            pulls,
            [
                '{"name":"llama3.2:3b"}',
                '{"name":"nomic-embed-text"}',
                '{"name":"llama3.2:1b"}',
            ],
        )


if __name__ == "__main__":
    unittest.main()
