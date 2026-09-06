import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const source = readFileSync(new URL('../../docs/browser/app.js', import.meta.url), 'utf8');
const saving = source.slice(source.indexOf('const saveHandles ='), source.indexOf('const OUTPUT_NAME ='));

function setup({ existing = false, selected = false } = {}) {
  const calls = [];
  const controls = new Map();
  const $ = (key) => {
    if (!controls.has(key)) controls.set(key, { value: '', textContent: '' });
    return controls.get(key);
  };
  $('#save-scope').value = selected ? 'selected' : 'all';
  const handle = {
    async requestPermission() { calls.push('permission'); return 'granted'; },
    async getDirectoryHandle() {
      if (existing) return {};
      throw Object.assign(new Error(), { name: 'NotFoundError' });
    },
  };
  const context = vm.createContext({
    $, state: { session: { views: [{ url: '/input', name: 'tile' }] },
      selectedViewUrls: new Set(), transformKey: 'registered', mounts: [] },
    window: { showDirectoryPicker: async () => handle },
    crypto: { randomUUID: () => '12345678' }, API_BASE: '/__mvs__',
    fsRequest: async () => ({ mount: 'output' }),
    command: async (name, payload) => { calls.push([name, payload]); },
    pool: { size: 2 }, withPool: async (action) => action(),
    setBusy() {}, setStatus() {}, log() {}, clearProgress() {},
  });
  vm.runInContext(saving, context);
  return { context, calls, handle };
}

test('conversion validates before writing and uses the displayed transform', async () => {
  const { context, calls } = setup();
  await vm.runInContext('doSave(false)', context);
  assert.equal(calls[0][0], 'prepare_save');
  assert.equal(calls[0][1].transform_key, 'registered');
  assert.equal(calls[1][0], 'save_image');
  assert.equal(calls[1][1].destination, '/__mvs__/fs/output/1-tile.ome.zarr');
});

test('existing output is refused without writes', async () => {
  const { context, calls } = setup({ existing: true });
  await assert.rejects(vm.runInContext('doSave(false)', context), /already exists/);
  assert.equal(calls.length, 1);
});

test('empty selection and remote metadata updates cannot write', async () => {
  const { context, calls } = setup({ selected: true });
  await assert.rejects(vm.runInContext('doSave(false)', context), /Select at least/);
  assert.equal(calls.length, 0);
  context.$('#save-scope').value = 'all';
  await assert.rejects(vm.runInContext('doSave(true)', context), /local OME-Zarr/);
  assert.equal(calls.length, 0);
});

test('metadata update requests source permission and sends no output path', async () => {
  const { context, calls, handle } = setup();
  context.handle = handle;
  vm.runInContext("saveHandles.set('/input', handle)", context);
  await vm.runInContext('doSave(true)', context);
  assert.equal(calls[0], 'permission');
  assert.equal(calls[1][1].metadata_only, true);
  assert.equal(calls[2][1].destination, null);
});
