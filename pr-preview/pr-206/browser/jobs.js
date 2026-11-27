/**
 * Work the page is running on behalf of a blocked Python worker.
 *
 * A Pyodide worker spreads registration and fusion over the pool by blocking
 * on a synchronous request, which the service worker forwards here. It cannot
 * *wait* on one of those requests, though: a browser terminates a service
 * worker whose fetch event outlives its budget - a few minutes in Chrome - and
 * the request it was answering then fails with a bare `NetworkError`. That is
 * a wall the work will hit sooner or later, since one pairwise registration
 * over a long timelapse legitimately runs for minutes.
 *
 * So the two are separated: a request *starts* a job and returns at once, and
 * later requests ask how it is going. Nothing here is on a clock, and the
 * requests that carry the conversation each last seconds.
 *
 * That makes both halves idempotent, which is the other half of the point -
 * Python may retry a request whose answer never arrived:
 *
 *   - starting a job under an id that is already running joins it instead of
 *     running the work a second time;
 *   - a finished job keeps its results for `ttlMs`, so a poll whose reply was
 *     the part that got lost can be asked again.
 *
 * Deliberately free of the DOM and of the worker pool, so the bookkeeping can
 * be exercised under `node --test`.
 */

/** How long a finished job's results stay available to a repeated poll. */
export const DEFAULT_TTL_MS = 5 * 60 * 1000;

export class DispatchJobs {
  constructor({
    ttlMs = DEFAULT_TTL_MS,
    now = () => Date.now(),
    sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  } = {}) {
    this.ttlMs = ttlMs;
    this.now = now;
    this.sleep = sleep;
    this.jobs = new Map();
  }

  get size() {
    return this.jobs.size;
  }

  has(id) {
    return this.jobs.has(id);
  }

  /**
   * Start `run()` under `id`, or return the job already running under it.
   *
   * `completed` is what the caller counts from - the work finished before this
   * request - and `run` is handed the job so it can add to it as tasks
   * complete. A failure is recorded rather than thrown: it belongs to whoever
   * polls for the job, not to whoever happened to start it.
   */
  start(id, run, { completed = 0 } = {}) {
    this.prune();

    const running = this.jobs.get(id);
    if (running) return running;

    const job = { id, done: false, completed, results: null, error: null };
    job.promise = Promise.resolve()
      .then(() => run(job))
      .then(
        (results) => {
          job.results = results;
        },
        (error) => {
          job.error = String((error && error.message) || error);
        },
      )
      .finally(() => {
        job.done = true;
        job.finishedAt = this.now();
      });

    this.jobs.set(id, job);
    return job;
  }

  /**
   * Wait up to `timeoutMs` for a job, then say where it has got to.
   *
   * Throws for a job this page has never heard of - which means it was never
   * started here, or the tab reloaded under a still-running Python worker.
   * Saying so beats answering "not finished yet" for ever.
   */
  async poll(id, timeoutMs) {
    const job = this.jobs.get(id);
    if (!job) throw new Error(`unknown dispatch job '${id}'`);

    await Promise.race([job.promise, this.sleep(Math.max(0, timeoutMs || 0))]);

    if (!job.done) return { done: false, completed: job.completed };
    if (job.error) throw new Error(job.error);
    return { done: true, results: job.results };
  }

  /** Forget finished jobs whose results nobody can still be waiting for. */
  prune() {
    const now = this.now();
    for (const [id, job] of this.jobs) {
      if (job.finishedAt !== undefined && now - job.finishedAt > this.ttlMs) {
        this.jobs.delete(id);
      }
    }
  }
}
