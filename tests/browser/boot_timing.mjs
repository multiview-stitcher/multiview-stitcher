/**
 * Where the Python runtime's boot time goes.
 *
 *   node tests/browser/boot_timing.mjs
 *
 * Node has no browser HTTP cache, so every phase here is a cold one - which is
 * the case that matters: the workers boot concurrently, so they all miss the
 * cache together.
 */

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const appDir = join(repoRoot, "docs", "browser");

const config = JSON.parse(readFileSync(join(appDir, "config.json"), "utf8"));
const manifest = JSON.parse(
  readFileSync(join(appDir, "packages", "manifest.json"), "utf8"),
);
const wheelPath = join(appDir, "packages", manifest.wheel);

const timings = [];
async function phase(label, fn) {
  const started = performance.now();
  const value = await fn();
  const ms = performance.now() - started;
  timings.push([label, ms]);
  console.log(`  ${(ms / 1000).toFixed(1)}s  ${label}`);
  return value;
}

console.log("cold boot, one worker:");

const pyodide = await phase(
  `loadPyodide + ${config.pyodide_packages.length} bundled packages`,
  () => loadPyodide({ packages: config.pyodide_packages }),
);

await phase(
  `micropip.install(${config.browser_dependencies.join(", ")})`,
  () =>
    pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(config.browser_dependencies)})
`),
);

await phase("micropip.install(multiview-stitcher wheel)", () =>
  pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:${wheelPath}", deps=False)
`),
);

await phase("import multiview_stitcher.browser.worker", () =>
  pyodide.runPythonAsync("import multiview_stitcher.browser.worker"),
);

const total = timings.reduce((sum, [, ms]) => sum + ms, 0);
console.log(`\n  ${(total / 1000).toFixed(1)}s  total, per worker`);
console.log("\nshare of total:");
for (const [label, ms] of timings.sort((a, b) => b[1] - a[1])) {
  console.log(`  ${((ms / total) * 100).toFixed(0).padStart(3)}%  ${label}`);
}

// What did micropip actually have to go and fetch?
const installed = await pyodide.runPythonAsync(`
import json, micropip
json.dumps(sorted(micropip.list().keys()))
`);
console.log("\nmicropip installed:", JSON.parse(installed).join(", "));
