/**
 * The trimmed Pyodide lockfile, loaded the way the app loads it.
 *
 * The app serves its own lockfile from `packages/` while the wheels stay on
 * Pyodide's distribution. Pyodide resolves package files against
 * `packageBaseUrl`, which it *defaults to the directory the lockfile came
 * from* - so a lockfile served from anywhere else sends it looking for every
 * wheel next to that copy. The failure is a wall of SRI errors and a runtime
 * with no micropip, nowhere near the lockfile that caused it.
 *
 * Keeping the lockfile in a directory of its own is the whole point here: a
 * lockfile sitting next to the wheels passes whether or not `packageBaseUrl`
 * is set, which is exactly how the bug got shipped.
 *
 *   node --test tests/browser/pyodide_lock.test.mjs
 */

import assert from "node:assert/strict";
import { copyFileSync, existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const distDir = join(here, "node_modules", "pyodide");
const builtLock = join(
  repoRoot, "docs", "browser", "packages", "pyodide-lock.json",
);

const available = existsSync(builtLock) && existsSync(distDir);

/** The app's lockfile, in a directory that holds nothing else. */
function lockAwayFromWheels() {
  const dir = mkdtempSync(join(tmpdir(), "mvs-lock-"));
  const path = join(dir, "pyodide-lock.json");
  copyFileSync(builtLock, path);
  return path;
}

test("packages load from the distribution, not from beside the lockfile", async (t) => {
  if (!available) {
    t.skip("built artefacts not present");
    return;
  }

  const { loadPyodide } = await import("pyodide");
  const pyodide = await loadPyodide({
    indexURL: `${distDir}/`,
    lockFileURL: lockAwayFromWheels(),
    // Without this, Pyodide looks for every wheel next to the lockfile.
    packageBaseUrl: `${distDir}/`,
    packages: ["micropip", "numpy", "scikit-image"],
  });

  const loaded = Object.keys(pyodide.loadedPackages);
  assert.ok(loaded.includes("micropip"), "micropip should be installed");
  assert.ok(loaded.includes("scikit-image"), "scikit-image should load");

  // The point of shipping our own lockfile: networkx no longer drags in
  // matplotlib and its font stack, ~10 MB per worker that nothing imports.
  assert.ok(
    !loaded.includes("matplotlib"),
    `matplotlib should not be loaded, got: ${loaded.join(", ")}`,
  );

  const edges = await pyodide.runPythonAsync(`
import networkx as nx, skimage.transform, scipy.ndimage
from skimage.filters import threshold_otsu
g = nx.Graph(); g.add_edge(0, 1)
g.number_of_edges()
`);
  assert.equal(edges, 1, "networkx and scikit-image must still work");
});
