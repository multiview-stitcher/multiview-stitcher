/**
 * The page's half of the dispatch protocol, with no browser.
 *
 * `docs/browser/jobs.js` is what makes a long registration survivable: the
 * request that starts the work returns immediately, so no service worker is
 * ever holding a fetch event open while Python registers a timelapse. What
 * matters is the bookkeeping around that - a retried request must join a job
 * rather than run it twice, and a finished job must still be collectable if
 * the answer, not the work, was what got lost.
 *
 *   node --test tests/browser/jobs.test.mjs
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const source = join(repoRoot, "docs", "browser", "jobs.js");

// Imported through a data: URL rather than by path; see placement.test.mjs for
// why `docs/browser/` cannot declare itself as ES modules to Node.
const { DispatchJobs, DEFAULT_TTL_MS } = await import(
  `data:text/javascript;base64,${readFileSync(source).toString("base64")}`
);

/** A job whose completion this test decides. */
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

test("a job runs once, however often it is submitted", async () => {
  const jobs = new DispatchJobs();
  let runs = 0;
  const gate = deferred();

  const run = () => {
    runs += 1;
    return gate.promise;
  };

  const first = jobs.start("abc", run);
  const second = jobs.start("abc", run);

  assert.equal(first, second);
  assert.equal(jobs.size, 1);

  gate.resolve(["result"]);
  assert.deepEqual(await jobs.poll("abc", 1000), {
    done: true,
    results: ["result"],
  });
  assert.equal(runs, 1, "a repeated submit must join, not restart");
});

test("polling reports progress while the work is still running", async () => {
  const jobs = new DispatchJobs();
  const gate = deferred();

  jobs.start(
    "abc",
    (job) => {
      job.completed += 4;
      return gate.promise;
    },
    { completed: 10 },
  );

  // The poll gives up waiting long before the job finishes, and says how far
  // it has got rather than nothing at all.
  assert.deepEqual(await jobs.poll("abc", 5), { done: false, completed: 14 });

  gate.resolve([]);
  assert.equal((await jobs.poll("abc", 1000)).done, true);
});

test("a finished job can be collected more than once", async () => {
  // The reply to a poll can be lost just as a request can, and the work behind
  // it is far too expensive to run again.
  const jobs = new DispatchJobs();
  jobs.start("abc", () => Promise.resolve([1, 2]));

  assert.deepEqual(await jobs.poll("abc", 1000), { done: true, results: [1, 2] });
  assert.deepEqual(await jobs.poll("abc", 1000), { done: true, results: [1, 2] });
});

test("a failure belongs to whoever polls for the job", async () => {
  // Not to whoever started it: that request has long since been answered, and
  // an unhandled rejection there would take down the message handler.
  const jobs = new DispatchJobs();
  jobs.start("abc", () => Promise.reject(new Error("worker 2 died")));

  await assert.rejects(() => jobs.poll("abc", 1000), /worker 2 died/);
});

test("a job the page never heard of is an error, not a wait", async () => {
  // Otherwise a Python worker outliving a reloaded tab would poll for ever.
  const jobs = new DispatchJobs();
  await assert.rejects(() => jobs.poll("nope", 0), /unknown dispatch job/);
});

test("finished jobs are forgotten once nobody can still be waiting", async () => {
  let clock = 0;
  const jobs = new DispatchJobs({ now: () => clock });

  jobs.start("first", () => Promise.resolve([]));
  await jobs.poll("first", 0);

  clock += DEFAULT_TTL_MS + 1;
  jobs.start("second", () => Promise.resolve([]));

  assert.equal(jobs.has("first"), false, "the stale job must be dropped");
  assert.equal(jobs.has("second"), true);
});

test("a running job is never pruned, however long it takes", async () => {
  // Which is the whole point: a pairwise registration over a long timelapse
  // outlasts any timeout that would have been reasonable to pick.
  let clock = 0;
  const jobs = new DispatchJobs({ now: () => clock });
  const gate = deferred();

  jobs.start("slow", () => gate.promise);

  clock += DEFAULT_TTL_MS * 100;
  jobs.start("other", () => Promise.resolve([]));

  assert.equal(jobs.has("slow"), true);
  gate.resolve(["done"]);
  assert.deepEqual(await jobs.poll("slow", 1000), {
    done: true,
    results: ["done"],
  });
});
