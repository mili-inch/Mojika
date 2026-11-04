import assert from 'node:assert/strict';
import test from 'node:test';
import { ActionQueue } from '../src/ActionQueue.ts';

function gate() {
  let release!: () => void;
  const wait = new Promise<void>(resolve => { release = resolve; });
  return { wait, release };
}

test('a delayed request finishes before the next value is sent', async () => {
  const queue = new ActionQueue();
  const first = gate();
  const sent: number[] = [];
  const a = queue.submit('noise_mix', async () => { sent.push(0); await first.wait; });
  const b = queue.submit('noise_mix', async () => { sent.push(1); });
  await Promise.resolve();
  assert.deepEqual(sent, [0]);
  first.release();
  await Promise.all([a, b]);
  assert.deepEqual(sent, [0, 1]);
});

test('waiting edits keep the newest value of each field without building a backlog', async () => {
  const queue = new ActionQueue();
  const first = gate();
  const sent: string[] = [];
  const active = queue.submit('gpt', () => first.wait);
  const tasks = [];
  for (let i = 0; i < 100; i++) {
    tasks.push(queue.submit('noise_mix', async () => { sent.push(`mix:${i}`); }));
  }
  tasks.push(queue.submit('transition_seconds', async () => { sent.push('duration'); }));
  first.release();
  await Promise.all([active, ...tasks]);
  assert.deepEqual(sent, ['mix:99', 'duration']);
});

test('CLIP text and image share one pending edit', async () => {
  const queue = new ActionQueue();
  const first = gate();
  const sent: string[] = [];
  const active = queue.submit('gpt', () => first.wait);
  const text = queue.submit('clip_text', async () => { sent.push('text'); });
  const image = queue.submit('clip_image', async () => { sent.push('image'); });
  first.release();
  await Promise.all([active, text, image]);
  assert.deepEqual(sent, ['image']);
});

test('manual update and clear actions preserve the edits on either side', async () => {
  for (const kind of ['noise', 'clear_condition', 'clear_conditions'] as const) {
    const queue = new ActionQueue();
    const first = gate();
    const sent: string[] = [];
    const active = queue.submit('gpt', () => first.wait);
    const before = queue.submit('noise_mix', async () => { sent.push('before'); });
    const button = queue.submit(kind, async () => { sent.push(kind); });
    const after = queue.submit('noise_mix', async () => { sent.push('after'); });
    first.release();
    await Promise.all([active, before, button, after]);
    assert.deepEqual(sent, ['before', kind, 'after']);
  }
});

test('losing control cancels unsent edits and allows a new operation afterward', async () => {
  const queue = new ActionQueue();
  const first = gate();
  const sent: string[] = [];
  const active = queue.submit('gpt', () => first.wait);
  const stale = queue.submit('noise_mix', async () => { sent.push('stale'); });
  queue.cancel();
  await stale;
  const fresh = queue.submit('noise_mix', async () => { sent.push('fresh'); });
  first.release();
  await Promise.all([active, fresh]);
  assert.deepEqual(sent, ['fresh']);
});

test('a failed request rejects its caller and does not block the next request', async () => {
  const queue = new ActionQueue();
  const first = gate();
  const sent: string[] = [];
  const failed = queue.submit('noise_mix', async () => { await first.wait; throw new Error('network failed'); });
  const checked = assert.rejects(failed, /network failed/);
  const next = queue.submit('noise_mix', async () => { sent.push('next'); });
  first.release();
  await Promise.all([checked, next]);
  assert.deepEqual(sent, ['next']);
});
