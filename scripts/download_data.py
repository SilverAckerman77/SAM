from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Direct ZIP URL or local ZIP path")
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.url)
    temporary: Path | None = None
    if source.is_file():
        archive = source
    elif urlparse(args.url).scheme in {"http", "https"}:
        temporary = Path(tempfile.mkstemp(suffix=".zip")[1])
        urlretrieve(args.url, temporary)
        archive = temporary
    else:
        raise FileNotFoundError(
            "Provide a direct ZIP URL or an existing local ZIP path; a Roboflow landing page is not enough."
        )
    try:
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"Downloaded input is not a ZIP archive: {archive}")
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Extracted dataset to {output.resolve()}")


if __name__ == "__main__":
    main()
