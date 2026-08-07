"""
Assemble the browser app that ships with the documentation.

The static files in ``docs/browser`` are checked in; this script adds the two
pieces that cannot be:

* the multiview-stitcher wheel the Pyodide runtime installs, plus a manifest
  the page reads to find it, and
* a Neuroglancer bundle, built from its npm package so the app can import its
  public API directly, and served from our own origin (neuroglancer creates
  Web Workers from these files, which is only possible same-origin).

Usage::

    python scripts/build_browser_app.py                # wheel + manifest
    python scripts/build_browser_app.py --neuroglancer # also bundle the viewer
    python scripts/build_browser_app.py --check        # verify, change nothing
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "docs" / "browser"
PACKAGES_DIR = APP_DIR / "packages"
NEUROGLANCER_DIR = APP_DIR / "neuroglancer"

#: Neuroglancer is bundled from its npm package rather than mirrored from a
#: hosted build. The package ships ES modules and expects a bundler, which is
#: what lets the app import its public API directly instead of embedding a
#: prebuilt viewer application in an iframe.
NEUROGLANCER_PACKAGE = "neuroglancer@2.41.2"
ESBUILD_PACKAGE = "esbuild@0.25.0"

#: Worker entry points bundled separately from the module the app imports.
#: neuroglancer starts each itself with `new Worker(new URL(..., import.meta.url))`,
#: so each has to sit exactly where the bundle that starts it looks for it -
#: which is not the same directory for both. `_missing_bundle_assets` checks
#: the resolved locations; this is only the list of what to build.
NEUROGLANCER_WORKER_ENTRIES = (
    "chunk_worker.bundle.js",
    "async_computation.bundle.js",
)

#: What the app imports from the bundle. Kept small and explicit so that a
#: neuroglancer upgrade fails loudly here rather than somewhere in the UI.
NEUROGLANCER_ENTRY = """\
// Side-effect import: registers neuroglancer's datasource, layer and kvstore
// front ends. Without it a viewer has no way to open a data source.
import "neuroglancer";

export { setupDefaultViewer } from "neuroglancer/unstable/ui/default_viewer_setup.js";
export { Viewer } from "neuroglancer/unstable/viewer.js";
export { StatusMessage } from "neuroglancer/unstable/status.js";

// Builds one layer from a layer specification, so a layer can be added to a
// running viewer instead of restoring a whole `layers` array - which clears
// the list and rebuilds every layer, taking the layout and each layer's
// shader settings with it.
export { makeLayer } from "neuroglancer/unstable/layer/index.js";
"""

#: esbuild has no built-in loader for the assets neuroglancer imports.
ESBUILD_LOADERS = (
    "--loader:.svg=text",
    "--loader:.css=css",
    "--loader:.html=text",
    "--loader:.wasm=file",
)

#: Assets the bundles fetch at run time by URL. The specifier is captured
#: whole, including any leading `../`: neuroglancer writes these for its own
#: output layout, and a pattern that only matched `./` silently passed the one
#: reference that pointed outside the bundle directory.
_URL_ASSET_RE = re.compile(r"""new URL\(\s*["'](\.{1,2}/[^"']+)["']""")


#: Static files that must exist for the app to work.
REQUIRED_FILES = (
    "index.html",
    "app.js",
    "app.css",
    "logo.svg",
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


#: The page's own sources. Their contents feed the build id, so that editing
#: the app without touching Python still produces a URL the browser treats as
#: new - a wheel-only id would leave every JavaScript fix hidden behind the
#: cached copy of the file that was meant to change.
_APP_SOURCES = (
    "app.js",
    "app.css",
    "camera.js",
    "compute-worker.js",
    "fs-worker.js",
    "highlight.js",
    "index.html",
    "jobs.js",
    "placement.js",
    "py-runtime.js",
    "routes.js",
    "session-worker.js",
    "sw.js",
    "viewer.js",
)


def _app_digest(wheel):
    """One digest over everything a running page consists of."""
    digest = hashlib.sha256()
    digest.update(_sha256(wheel).encode())

    for name in sorted(_APP_SOURCES):
        path = APP_DIR / name
        if not path.exists():
            raise RuntimeError(f"The browser app is missing {name}.")
        # Hash the name too, so moving code between files still registers.
        digest.update(name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())

    return digest.hexdigest()


#: Dependencies Pyodide's lockfile declares that nothing here imports, as
#: ``{package: (dependency, ...)}``. Pyodide lists matplotlib as a dependency
#: of networkx, so pulling in scikit-image drags matplotlib and its font stack
#: along: about 10 MB per worker, downloaded and unpacked every boot, for a
#: module only ``networkx.drawing`` would use.
_UNUSED_DEPENDENCIES = {"networkx": ("matplotlib",)}

#: Name of the trimmed lockfile the page loads instead of Pyodide's own.
PYODIDE_LOCK_NAME = "pyodide-lock.json"


def write_pyodide_lock():
    """Write a copy of Pyodide's lockfile without the unused dependencies.

    Only the dependency graph changes; every package is still fetched from
    Pyodide's own CDN, so this cannot pull in a package that Pyodide did not
    build. Returns None when the lockfile cannot be fetched - the page then
    falls back to Pyodide's, which costs download size but still works.
    """
    config = json.loads((APP_DIR / "config.json").read_text())
    url = f"{config['pyodide_index_url']}{PYODIDE_LOCK_NAME}"

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            lock = json.loads(response.read().decode())
    except Exception as error:  # noqa: BLE001 - any failure is non-fatal
        print(f"warning: could not fetch {url}: {error}")
        print("         the app will use Pyodide's own lockfile")
        return None

    trimmed = []
    for name, unused in _UNUSED_DEPENDENCIES.items():
        package = lock.get("packages", {}).get(name)
        if package is None:
            continue
        kept = [dep for dep in package["depends"] if dep not in unused]
        if kept != package["depends"]:
            trimmed.append(f"{name} -= {', '.join(sorted(set(package['depends']) - set(kept)))}")
            package["depends"] = kept

    path = PACKAGES_DIR / PYODIDE_LOCK_NAME
    path.write_text(json.dumps(lock))
    print(f"pyodide lock: {path.relative_to(REPO_ROOT)} ({'; '.join(trimmed) or 'unchanged'})")
    return path


def write_manifest(wheel, pyodide_lock=None):
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
        "build": _app_digest(wheel)[:12],
        # Absent when the lockfile could not be fetched; the page then lets
        # Pyodide use its own.
        "pyodide_lock": pyodide_lock.name if pyodide_lock else None,
        "size": wheel.stat().st_size,
        "pyodide_version": config["pyodide_version"],
        "browser_dependencies": config["browser_dependencies"],
    }

    path = PACKAGES_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


#: Places npm commonly lives when it is not on PATH, e.g. under nvm or a
#: package manager that does not touch the login shell's PATH.
_NPM_SEARCH_GLOBS = (
    "~/.nvm/versions/node/*/bin/npm",
    "~/.local/share/fnm/node-versions/*/installation/bin/npm",
    "/usr/local/bin/npm",
    "/opt/homebrew/bin/npm",
)


def find_npm(explicit=None):
    """Locate npm, or return None.

    Checked in order: an explicit path, ``MVS_NPM``, ``PATH``, then a few
    common install locations - version managers routinely leave npm off the
    PATH of a non-login shell.
    """
    candidates = [explicit, os.environ.get("MVS_NPM")]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)

    found = shutil.which("npm")
    if found:
        return found

    for pattern in _NPM_SEARCH_GLOBS:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            return matches[-1]

    return None


class NpmNotFound(RuntimeError):
    """Raised when the viewer cannot be bundled because npm is missing."""

    def __init__(self):
        super().__init__(
            "npm was not found, and it is needed to bundle the Neuroglancer "
            "viewer from its npm package.\n"
            "\n"
            "  - install Node.js 18 or newer (https://nodejs.org), or\n"
            "  - point this script at an existing npm:\n"
            "        python scripts/build_browser_app.py --neuroglancer "
            "--npm /path/to/npm\n"
            "        MVS_NPM=/path/to/npm python scripts/build_browser_app.py "
            "--neuroglancer\n"
            "\n"
            "The rest of the app builds without Node: drop --neuroglancer to "
            "rebuild only the wheel and keep the viewer bundle you already "
            "have (run --check to confirm it is complete)."
        )


def _npm(args, cwd, npm=None):
    subprocess.run([npm or "npm", *args], cwd=str(cwd), check=True)


def bundle_neuroglancer(work_dir=None, npm=None):
    """Bundle neuroglancer from npm into the app.

    The npm package ships ES module sources and expects a bundler; esbuild
    produces one module the app can import. Three entry points are built: the
    module the app talks to, and the two Web Worker entry points neuroglancer
    starts itself by URL. Everything lands in one directory so that those URLs,
    which are relative to the bundle, resolve.

    Serving it from our own origin still matters: neuroglancer creates its
    workers from these files, and a worker cannot be created from another
    origin.
    """
    npm = find_npm(npm)
    if npm is None:
        raise NpmNotFound

    # Outside docs/, or mkdocs would publish node_modules with the site.
    work_dir = Path(work_dir) if work_dir else REPO_ROOT / ".ng-build"
    work_dir.mkdir(parents=True, exist_ok=True)

    (work_dir / "package.json").write_text('{"private":true,"type":"module"}\n')
    print(f"installing {NEUROGLANCER_PACKAGE}")
    # Not --silent: when an install fails, its reason is the only useful
    # thing on the screen.
    _npm(["install", "--no-audit", "--no-fund", "--loglevel=error",
          NEUROGLANCER_PACKAGE, ESBUILD_PACKAGE], work_dir, npm=npm)

    entry = work_dir / "neuroglancer-entry.js"
    entry.write_text(NEUROGLANCER_ENTRY)

    if NEUROGLANCER_DIR.exists():
        shutil.rmtree(NEUROGLANCER_DIR)
    NEUROGLANCER_DIR.mkdir(parents=True)

    def esbuild(source, outfile):
        _npm([
            "exec", "--", "esbuild", str(source),
            "--bundle", "--format=esm", "--minify",
            f"--outfile={outfile}", *ESBUILD_LOADERS,
        ], work_dir, npm=npm)

    print("bundling neuroglancer")
    esbuild(entry, NEUROGLANCER_DIR / "neuroglancer.js")

    package_lib = work_dir / "node_modules" / "neuroglancer" / "lib"

    # neuroglancer.js creates the chunk worker from a URL relative to itself,
    # so it belongs beside it.
    esbuild(
        package_lib / "chunk_worker.bundle.js",
        NEUROGLANCER_DIR / "chunk_worker.bundle.js",
    )

    # The chunk worker in turn creates the async-computation worker - the one
    # that decodes compressed chunks - from a URL relative to *itself*, and
    # neuroglancer's source puts that a directory up. esbuild leaves such URLs
    # alone, so the file goes where the reference points rather than where it
    # would be tidier. Getting this wrong is close to invisible: metadata is
    # JSON and still loads, so a layer keeps its bounding box and contrast
    # range and simply never shows a pixel of compressed data.
    esbuild(
        package_lib / "async_computation.bundle.js",
        _asset_target(
            NEUROGLANCER_DIR / "chunk_worker.bundle.js",
            "async_computation.bundle.js",
        ),
    )

    # Assets the bundles fetch by URL at run time; esbuild leaves those URLs
    # alone, so the files have to be placed beside the bundles by hand.
    for asset in sorted(package_lib.rglob("*")):
        if asset.is_file() and asset.suffix in (".wasm", ".html"):
            shutil.copy2(asset, NEUROGLANCER_DIR / asset.name)

    license_path = work_dir / "node_modules" / "neuroglancer" / "LICENSE"
    if license_path.is_file():
        shutil.copy2(license_path, NEUROGLANCER_DIR / "LICENSE")

    (NEUROGLANCER_DIR / "PROVENANCE.txt").write_text(
        "Neuroglancer bundled by scripts/build_browser_app.py\n"
        f"package: {NEUROGLANCER_PACKAGE}\n"
        f"bundler: {ESBUILD_PACKAGE}\n"
        "license: Apache-2.0 (see LICENSE)\n"
    )

    missing = _missing_bundle_assets()
    if missing:
        # A file the viewer only fetches once a user opens an image would
        # otherwise fail at run time, long after the build looked fine.
        raise RuntimeError(
            "the neuroglancer bundle is missing runtime asset(s): "
            + ", ".join(sorted(missing))
        )

    built = sorted(path.name for path in NEUROGLANCER_DIR.iterdir())
    print(f"bundled {len(built)} file(s) into "
          f"{NEUROGLANCER_DIR.relative_to(REPO_ROOT)}")
    return NEUROGLANCER_DIR


def _asset_target(bundle, asset):
    """Where ``bundle`` will look for ``asset``, as an absolute path.

    The reference is read out of the bundle rather than assumed: neuroglancer
    writes these URLs for its own output layout, and ours is flatter.
    """
    if bundle.is_file():
        text = bundle.read_text(encoding="utf-8", errors="ignore")
        for reference in _URL_ASSET_RE.findall(text):
            if PurePosixPath(reference).name == asset:
                return (bundle.parent / reference).resolve()

    return bundle.parent / asset


def _missing_bundle_assets():
    """Assets the bundles fetch by URL that are not where they look for them.

    Every reference is resolved against the bundle that makes it, because they
    are not all relative to the bundle's own directory - the chunk worker asks
    for the async-computation worker a level up.
    """
    if not NEUROGLANCER_DIR.is_dir():
        return set()

    missing = set()
    for bundle in NEUROGLANCER_DIR.glob("*.js"):
        text = bundle.read_text(encoding="utf-8", errors="ignore")
        for reference in _URL_ASSET_RE.findall(text):
            target = (bundle.parent / reference).resolve()
            if not target.is_file():
                try:
                    shown = target.relative_to(REPO_ROOT)
                except ValueError:
                    shown = target
                missing.add(f"{shown} (wanted by {bundle.name})")

    return missing


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

    if not (NEUROGLANCER_DIR / "neuroglancer.js").is_file():
        print(
            "missing: docs/browser/neuroglancer/neuroglancer.js "
            "(run with --neuroglancer)"
        )
        ok = False
    else:
        # The viewer's stylesheet used to come with the iframe; an embedded
        # viewer needs it served alongside the bundle.
        if not (NEUROGLANCER_DIR / "neuroglancer.css").is_file():
            print("missing: docs/browser/neuroglancer/neuroglancer.css")
            ok = False

        # Only the chunk worker is addressed relative to the main bundle;
        # where the others belong is decided by the references themselves,
        # which `_missing_bundle_assets` resolves.
        if not (NEUROGLANCER_DIR / "chunk_worker.bundle.js").is_file():
            print("missing: docs/browser/neuroglancer/chunk_worker.bundle.js")
            ok = False

        missing = _missing_bundle_assets()
        if missing:
            print(
                "missing neuroglancer runtime asset(s): "
                + ", ".join(sorted(missing))
            )
            ok = False

    print("browser app is complete" if ok else "browser app is incomplete")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neuroglancer",
        action="store_true",
        help="also bundle the Neuroglancer viewer from npm (needs Node)",
    )
    parser.add_argument(
        "--npm",
        default=None,
        help="path to npm, when it is not on PATH (or set MVS_NPM)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that the app directory is complete",
    )
    args = parser.parse_args(argv)

    if args.check:
        return 0 if check() else 1

    # Fail before doing any work if the viewer cannot be bundled: rebuilding
    # the wheel first and only then discovering npm is missing wastes a minute
    # and leaves the tree half-updated.
    if args.neuroglancer and find_npm(args.npm) is None:
        print(NpmNotFound(), file=sys.stderr)
        return 1

    wheel = build_wheel()
    pyodide_lock = write_pyodide_lock()
    manifest = write_manifest(wheel, pyodide_lock)
    print(f"wheel:    docs/browser/packages/{wheel.name}")
    print(f"manifest: {manifest.relative_to(REPO_ROOT)}")

    if args.neuroglancer:
        bundle_neuroglancer(npm=args.npm)
    elif not (NEUROGLANCER_DIR / "neuroglancer.js").is_file():
        print(
            "note: no Neuroglancer bundle found - "
            "run again with --neuroglancer to build one"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
