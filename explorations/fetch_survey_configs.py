"""Download Hugging Face config.json for each survey model into configs/."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_EXPL = Path(__file__).resolve().parent
for p in (_ROOT, _EXPL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from survey_load import CSV_PATH

CONFIG_DIR = _ROOT / "configs"
REVISIONS = ("main", "master")
TIMEOUT = 45
MAX_WORKERS = 8


def repo_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def config_url(repo_id: str, revision: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/config.json"


def fetch_config(repo_id: str) -> tuple[dict | None, str | None]:
    last_err: str | None = None
    for revision in REVISIONS:
        url = config_url(repo_id, revision)
        req = urllib.request.Request(url, headers={"User-Agent": "Ablations-survey-fetch/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read()), None
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code} ({revision})"
        except urllib.error.URLError as exc:
            last_err = f"URL error ({revision}): {exc.reason}"
        except json.JSONDecodeError as exc:
            last_err = f"Invalid JSON ({revision}): {exc}"
    return None, last_err


def process_repo(repo_id: str) -> dict:
    config, err = fetch_config(repo_id)
    if config is None:
        return {"url": repo_id, "error": err}

    out_dir = CONFIG_DIR / repo_dir_name(repo_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"url": repo_id, "error": None}


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    repos = df["url"].astype(str).tolist()
    results: dict[str, dict] = {}

    print(f"Fetching configs for {len(repos)} models -> {CONFIG_DIR}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_repo, repo): repo for repo in repos}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results[row["url"]] = row
            status = "ok" if row["error"] is None else row["error"]
            print(f"  [{i}/{len(repos)}] {row['url']}: {status}")

    ok = sum(1 for r in results.values() if r["error"] is None)
    failed = [r for r in results.values() if r["error"] is not None]

    print(f"\nSaved {ok}/{len(repos)} configs under {CONFIG_DIR}")
    if failed:
        print(f"\n{len(failed)} failed:")
        for row in failed:
            print(f"  {row['url']}: {row['error']}")


if __name__ == "__main__":
    main()
