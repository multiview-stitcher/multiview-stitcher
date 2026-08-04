"""
Assemble the browser app that ships with the documentation.

The static files in ``docs/browser`` are checked in; this script adds the two
pieces that cannot be:

* the multiview-stitcher wheel the Pyodide runtime installs, plus a manifest
  the page reads to find it, and
* a Neuroglancer build, vendored so the viewer is served from the same origin
  as the app (which is what lets a service worker answer its chunk requests).

Usage::

    python scripts/build_browser_app.py                # wheel + manifest
    python scripts/build_browser_app.py --neuroglancer # also vendor the viewer
    python scripts/build_browser_app.py --check        # verify, change nothing
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "docs" / "browser"
PACKAGES_DIR = APP_DIR / "packages"
NEUROGLANCER_DIR = APP_DIR / "neuroglancer"

#: Prebuilt Neuroglancer bundle published on npm.
NEUROGLANCER_PACKAGE = "neuroglancer"
NEUROGLANCER_VERSION = "2.41.0"

#: Static files that must exist for the app to work.
REQUIRED_FILES = (
    "index.html",
    "app.js",
    "app.css",
    "config.json",
    "sw.js",
    "py-runtime.js",
    "session-worker.js",
    "compute-worker.js",
    "fs-worker.js",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_wheel():
    """Build a wheel of the working tree and place it next to the app."""
    PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

    for stale in PACKAGES_DIR.glob("multiview_stitcher-*.whl"):
        stale.unlink()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(PACKAGES_DIR),
            str(REPO_ROOT),
        ],
        check=True,
    )

    wheels = sorted(PACKAGES_DIR.glob("multiview_stitcher-*.whl"))
    if not wheels:
        raise RuntimeError("pip wheel produced no multiview-stitcher wheel.")

    return wheels[-1]


def write_manifest(wheel):
    """Record the wheel the page should install, with its checksum."""
    config = json.loads((APP_DIR / "config.json").read_text())

    manifest = {
        "wheel": wheel.name,
        "sha256": _sha256(wheel),
        "size": wheel.stat().st_size,
        "pyodide_version": config["pyodide_version"],
        "browser_dependencies": config["browser_dependencies"],
    }

    path = PACKAGES_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def vendor_neuroglancer(version=NEUROGLANCER_VERSION):
    """Download a Neuroglancer build and unpack it below the app.

    Serving the viewer from our own origin is what allows the service worker to
    intercept its chunk requests; a hosted instance could not read the user's
    local files.
    """
    url = (
        f"https://registry.npmjs.org/{NEUROGLANCER_PACKAGE}/-/"
        f"{NEUROGLANCER_PACKAGE}-{version}.tgz"
    )
    print(f"downloading {url}")

    with urllib.request.urlopen(url) as response:  # noqa: S310 - pinned URL
        payload = response.read()

    if NEUROGLANCER_DIR.exists():
        shutil.rmtree(NEUROGLANCER_DIR)
    NEUROGLANCER_DIR.mkdir(parents=True)

    prefix = "package/dist/client/"
    extracted = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = Path(member.name[len(prefix) :])
            target = NEUROGLANCER_DIR / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source:
                target.write_bytes(source.read())
            extracted += 1

    if not extracted:
        raise RuntimeError(
            f"no client bundle found under '{prefix}' in the "
            f"{NEUROGLANCER_PACKAGE} {version} package."
        )

    print(f"vendored {extracted} Neuroglancer file(s) into {NEUROGLANCER_DIR}")
    return NEUROGLANCER_DIR


def check():
    """Report whether the app directory is complete. Returns True when it is."""
    ok = True

    for name in REQUIRED_FILES:
        if not (APP_DIR / name).is_file():
            print(f"missing: docs/browser/{name}")
            ok = False

    manifest_path = PACKAGES_DIR / "manifest.json"
    if not manifest_path.is_file():
        print("missing: docs/browser/packages/manifest.json (run this script)")
        ok = False
    else:
        manifest = json.loads(manifest_path.read_text())
        wheel = PACKAGES_DIR / manifest["wheel"]
        if not wheel.is_file():
            print(f"missing: docs/browser/packages/{manifest['wheel']}")
            ok = False
        elif _sha256(wheel) != manifest["sha256"]:
            print("stale: the wheel does not match its manifest checksum")
            ok = False

    if not (NEUROGLANCER_DIR / "index.html").is_file():
        print(
            "missing: docs/browser/neuroglancer/index.html "
            "(run with --neuroglancer)"
        )
        ok = False

    print("browser app is complete" if ok else "browser app is incomplete")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neuroglancer",
        action="store_true",
        help="also download and vendor the Neuroglancer client",
    )
    parser.add_argument(
        "--neuroglancer-version",
        default=NEUROGLANCER_VERSION,
        help=f"Neuroglancer version to vendor (default {NEUROGLANCER_VERSION})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that the app directory is complete",
    )
    args = parser.parse_args(argv)

    if args.check:
        return 0 if check() else 1

    wheel = build_wheel()
    manifest = write_manifest(wheel)
    print(f"wheel:    docs/browser/packages/{wheel.name}")
    print(f"manifest: {manifest.relative_to(REPO_ROOT)}")

    if args.neuroglancer:
        vendor_neuroglancer(args.neuroglancer_version)
    elif not (NEUROGLANCER_DIR / "index.html").is_file():
        print(
            "note: no Neuroglancer build found - "
            "run again with --neuroglancer to vendor one"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
