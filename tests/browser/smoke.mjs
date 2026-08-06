/**
 * Pyodide end-to-end smoke test.
 *
 * Boots the same runtime the browser app uses (same Pyodide version, same
 * pinned dependencies, the freshly built multiview-stitcher wheel), then runs
 * `smoke.py` inside it. Catches the integration failures that only appear in
 * WebAssembly - zarr v2, an older xarray, no ngff-zarr / ome-zarr-py - long
 * before any of it reaches the UI.
 *
 *   node tests/browser/smoke.mjs [path/to/wheel.whl]
 */

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

const config = JSON.parse(
  readFileSync(join(repoRoot, "docs", "browser", "config.json"), "utf8"),
);

function findWheel() {
  if (process.argv[2]) return resolve(process.argv[2]);

  for (const dir of ["dist", join("docs", "browser", "packages")]) {
    const full = join(repoRoot, dir);
    if (!existsSync(full)) continue;
    const wheels = readdirSync(full)
      .filter((name) => name.startsWith("multiview_stitcher-") && name.endsWith(".whl"))
      .sort();
    if (wheels.length) return join(full, wheels[wheels.length - 1]);
  }

  throw new Error(
    "No multiview-stitcher wheel found. Build one with " +
      "`python -m build --wheel` or pass its path as an argument.",
  );
}

const wheelPath = findWheel();
const wheelName = wheelPath.split("/").pop();
console.log(`pyodide ${config.pyodide_version}, wheel ${wheelName}`);

const pyodide = await loadPyodide({ packages: config.pyodide_packages });

pyodide.FS.writeFile(`/${wheelName}`, readFileSync(wheelPath));

const started = Date.now();
await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(config.browser_dependencies)})
await micropip.install("emfs:/${wheelName}", deps=False)
`);
console.log(`dependencies installed in ${((Date.now() - started) / 1000).toFixed(1)}s`);

pyodide.FS.writeFile(
  "/smoke.py",
  readFileSync(join(here, "smoke.py")),
);

let report;
try {
  report = JSON.parse(
    await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, "/")
import smoke
smoke.main()
`),
  );
} catch (error) {
  console.error("smoke test raised:\n" + error.message);
  process.exit(1);
}

// Call serve_route the way the browser workers do: across the JavaScript
// boundary, with the session spec as a JSON string. Handing over a live JS
// object instead turns its nulls into `JsNull` proxies, which pass Python's
// `is not None` checks and then fail deep inside numeric code - a failure
// that only exists on this boundary and cannot be reproduced from Python.
const jsBoundary = await (async () => {
  const worker = pyodide.pyimport("multiview_stitcher.browser.worker");
  const setup = JSON.parse(
    await pyodide.runPythonAsync(`
import json
from multiview_stitcher.browser import example_data, worker as w
w._runtime = w.WorkerRuntime()
w.handle_json("load", json.dumps({"sources": example_data.example_sources("tiles-3d")}))
preview = json.loads(w.handle_json("fuse_preview", json.dumps({"options": {}})))["result"]
json.dumps({"route": preview["route"], "spec": json.loads(w.handle_json("spec", "{}"))["result"]})
`),
  );

  // A fresh runtime, so the spec is the only way to reach the image - exactly
  // a compute worker's situation.
  await pyodide.runPythonAsync(
    "from multiview_stitcher.browser import worker as w; w._runtime = w.WorkerRuntime()",
  );

  const results = {};
  for (const key of [".zattrs", "0/.zarray"]) {
    const result = worker.serve_route(setup.route, key, JSON.stringify(setup.spec));
    const [status] = result.toJs();
    result.destroy();
    results[key] = status;
  }
  return results;
})();

report.checks.serve_route_across_the_js_boundary = {
  ok: Object.values(jsBoundary).every((status) => status === 200),
  detail: JSON.stringify(jsBoundary),
};

// The app mounts the user's CZI with WORKERFS, which is what turns a file of
// any size into an ordinary path without reading it into memory. It is a
// link-time option of the Pyodide build rather than something the app can add,
// so an upgrade that dropped it would take CZI support with it - silently,
// since nothing else here uses that filesystem.
//
// Mounting itself cannot be exercised: WORKERFS refuses to mount outside a Web
// Worker, and Node has none. What is checked is that the build still offers it.
report.checks.workerfs_available = {
  ok: Boolean(pyodide.FS.filesystems && pyodide.FS.filesystems.WORKERFS),
  detail: Object.keys(pyodide.FS.filesystems || {}).join(", "),
};

const checks = Object.entries(report.checks);
for (const [name, result] of checks) {
  console.log(`  ${result.ok ? "ok  " : "FAIL"} ${name}${result.detail ? ` - ${result.detail}` : ""}`);
}

const failed = checks.filter(([, result]) => !result.ok);
console.log(
  `\n${checks.length - failed.length}/${checks.length} checks passed ` +
    `(python ${report.python}, zarr ${report.runtime.zarr}, ` +
    `xarray ${report.runtime.xarray}, dask ${report.runtime.dask})`,
);

process.exit(failed.length ? 1 : 0);
