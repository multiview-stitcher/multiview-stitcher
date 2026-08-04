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
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "docs" / "browser"
PACKAGES_DIR = APP_DIR / "packages"
NEUROGLANCER_DIR = APP_DIR / "neuroglancer"

#: Official hosted Neuroglancer build, mirrored into the app.
#:
#: The `neuroglancer` npm package ships ES module sources only, so vendoring a
#: *client* from it would mean running a bundler. Mirroring the build Google
#: already publishes keeps this script dependency-free, and the app needs the
#: viewer on its own origin so the service worker can answer its requests.
NEUROGLANCER_URL = "https://neuroglancer-demo.appspot.com/"
NEUROGLANCER_LICENSE_URL = (
    "https://raw.githubusercontent.com/google/neuroglancer/master/LICENSE"
)

#: Entry points that the bundle links to but never names in its own code.
NEUROGLANCER_EXTRA_PAGES = ("bossauth.html", "google_oauth2_redirect.html")

#: Assets referenced from markup: <script src>, <link href>.
_MARKUP_ASSET_RE = re.compile(
    r"""(?:src|href)\s*=\s*["']([^"'#?]+\.(?:js|css|wasm))["']""",
    re.IGNORECASE,
)
#: Asset names appearing as string literals inside the bundle itself.
_LITERAL_ASSET_RE = re.compile(
    r"""["'`]([A-Za-z0-9_.-]+\.(?:js|css|wasm))["'`]"""
)
#: Webpack's chunk-url template, e.g. `.u=e=>""+e+".4aba060a495df99f.js"`.
_CHUNK_TEMPLATE_RE = re.compile(
    r"""\.u\s*=\s*\w+\s*=>\s*["']([^"']*)["']\s*\+\s*\w+\s*\+\s*["']([^"']*)["']"""
)
#: The other form Webpack emits when chunks do not share one content hash:
#: a chunk id -> hash table, e.g. `({34:"f9e3...",586:"deac..."})[e]+".js"`.
_CHUNK_TABLE_RE = re.compile(r"""(\d+)\s*:\s*["']([0-9a-f]{8,})["']""")
#: Call sites naming a chunk id: `i.u("145")` and `r.e("586")`.
_CHUNK_ID_RE = re.compile(r"""\.[ue]\(\s*["']?(\d+)["']?\s*[,)]""")

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

    digest = _sha256(wheel)

    manifest = {
        "wheel": wheel.name,
        # The page appends this to the wheel URL. A rebuild of the same commit
        # produces an identically named wheel, and micropip fetches it from
        # inside a worker, where a page reload does not bypass the HTTP cache -
        # so without a changing URL the browser keeps running the previous
        # build's Python behind the current JavaScript.
        "sha256": digest,
        "build": digest[:12],
        "size": wheel.stat().st_size,
        "pyodide_version": config["pyodide_version"],
        "browser_dependencies": config["browser_dependencies"],
    }

    path = PACKAGES_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _download(url, optional=False):
    """Fetch a URL, returning its bytes (or None when optional and missing)."""
    request = urllib.request.Request(
        url,
        # Ask for no transfer encoding so responses need no post-processing;
        # some fronts ignore this, so gzip is still handled below.
        headers={"Accept-Encoding": "identity", "User-Agent": "multiview-stitcher"},
    )

    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310
            payload = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                payload = gzip.decompress(payload)
            return payload
    except urllib.error.HTTPError as error:
        if optional and error.code in (403, 404):
            return None
        raise


def _referenced_assets(payload, from_markup):
    """Return ``(names, chunk_templates, chunk_ids)`` referenced by one file.

    Bundled code names its lazily loaded pieces in two ways: directly, as
    string literals (the WebAssembly decoders live in the lazy chunks, which is
    why the crawl has to be iterative), and indirectly, by passing a chunk id
    to a url template that each entry bundle defines for itself.
    """
    text = payload.decode("utf-8", errors="ignore")

    if from_markup:
        return set(_MARKUP_ASSET_RE.findall(text)), set(), set()

    names = {
        name
        for name in _LITERAL_ASSET_RE.findall(text)
        if not name.startswith((".", "/"))
    }
    # A chunk id -> hash table names its chunks outright.
    names |= {
        f"{chunk_id}.{digest}.js"
        for chunk_id, digest in _CHUNK_TABLE_RE.findall(text)
    }
    return (
        names,
        set(_CHUNK_TEMPLATE_RE.findall(text)),
        set(_CHUNK_ID_RE.findall(text)),
    )


def vendor_neuroglancer(base_url=NEUROGLANCER_URL):
    """Mirror a hosted Neuroglancer build into the app.

    Serving the viewer from our own origin is what allows the service worker to
    intercept its chunk requests; a hosted instance could not read the user's
    local files.

    Starting from ``index.html``, every asset the bundle names is followed:
    markup references first, then the string literals and Webpack chunk table
    inside the downloaded code, which is where the lazily loaded chunks and the
    WebAssembly decoders appear. Candidates that do not exist are skipped, so a
    changed build layout degrades to a smaller mirror rather than a crash.
    """
    base_url = base_url if base_url.endswith("/") else base_url + "/"
    print(f"mirroring {base_url}")

    index = _download(urljoin(base_url, "index.html"))

    names, _, _ = _referenced_assets(index, from_markup=True)
    if not names:
        raise RuntimeError(f"no assets referenced by {base_url}index.html")

    pending = list(names) + list(NEUROGLANCER_EXTRA_PAGES)
    downloaded = {"index.html": index}
    seen = {"index.html"}
    templates = set()
    chunk_ids = set()

    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)

        payload = _download(urljoin(base_url, name), optional=True)
        if payload is None:
            continue

        downloaded[name] = payload
        names, new_templates, new_ids = _referenced_assets(
            payload, from_markup=name.endswith(".html")
        )
        templates |= new_templates
        chunk_ids |= new_ids

        # Any template may be paired with any chunk id: each entry bundle
        # carries its own runtime, and they share the require object at run
        # time. Combinations that do not exist simply 404 and are skipped.
        for prefix, suffix in templates:
            names |= {
                f"{prefix}{chunk_id}{suffix}" for chunk_id in chunk_ids
            }

        pending += [
            candidate for candidate in names if candidate not in seen
        ]

    license_text = _download(NEUROGLANCER_LICENSE_URL, optional=True)
    if license_text is not None:
        downloaded["LICENSE"] = license_text

    # Recorded beside the mirror rather than injected into it, so every file
    # stays byte-identical to upstream and can be diffed against it.
    downloaded["PROVENANCE.txt"] = (
        "Neuroglancer client mirrored by scripts/build_browser_app.py\n"
        f"source:  {base_url}\n"
        f"files:   {len(downloaded)}\n"
        "license: Apache-2.0 (see LICENSE)\n"
    ).encode()

    if NEUROGLANCER_DIR.exists():
        shutil.rmtree(NEUROGLANCER_DIR)
    NEUROGLANCER_DIR.mkdir(parents=True)

    for name, payload in sorted(downloaded.items()):
        target = NEUROGLANCER_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    missing = _missing_chunks(downloaded)
    if missing:
        # A chunk the viewer loads at run time but that we did not mirror shows
        # up only as "failed to execute importScripts" once a user opens an
        # image, so refuse to ship an incomplete mirror.
        raise RuntimeError(
            "the Neuroglancer mirror is incomplete: no file for chunk id(s) "
            + ", ".join(sorted(missing))
            + ". The build's chunk naming has probably changed; update the "
            "chunk regexes in this script."
        )

    print(
        f"vendored {len(downloaded)} Neuroglancer file(s) into "
        f"{NEUROGLANCER_DIR.relative_to(REPO_ROOT)}"
    )
    return NEUROGLANCER_DIR


def _missing_chunks(downloaded):
    """Chunk ids the mirrored code loads at run time but that we do not have.

    Every lazily loaded chunk is fetched as ``<id>.<hash>.js``, so a chunk id
    is covered when some mirrored filename starts with ``<id>.``.
    """
    requested = set()
    for name, payload in downloaded.items():
        if not name.endswith(".js"):
            continue
        requested |= set(
            _CHUNK_ID_RE.findall(payload.decode("utf-8", errors="ignore"))
        )

    prefixes = {name.split(".", 1)[0] for name in downloaded if name.endswith(".js")}
    return requested - prefixes


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
        "--neuroglancer-url",
        default=NEUROGLANCER_URL,
        help=(
            "hosted Neuroglancer build to mirror "
            f"(default {NEUROGLANCER_URL})"
        ),
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
        vendor_neuroglancer(args.neuroglancer_url)
    elif not (NEUROGLANCER_DIR / "index.html").is_file():
        print(
            "note: no Neuroglancer build found - "
            "run again with --neuroglancer to vendor one"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
