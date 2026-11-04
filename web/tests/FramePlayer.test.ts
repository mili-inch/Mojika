import assert from 'node:assert/strict';
import test from 'node:test';
import { FramePlayer } from '../src/FramePlayer.ts';
import { packet } from './frameFixture.ts';

class Bitmap {
  width = 256; height = 256; closed = false; id: number;
  constructor(id: number) { this.id = id; }
  close() { assert.equal(this.closed, false); this.closed = true; }
}
const tick = () => new Promise(resolve => setImmediate(resolve));
function fixture() {
  let now = 10000, rafId = 0, recovered = 0;
  const sent: { type: string; id?: number; client_ms?: number; visible?: boolean }[] = [];
  const jobs: { resolve: (bitmap: Bitmap) => void; reject: (reason: Error) => void }[] = [];
  const rafs = new Map<number, () => void>(), drawn: number[] = [], errors: string[] = [];
  const player = new FramePlayer({
    now: () => now,
    decode: () => new Promise<Bitmap>((resolve, reject) => jobs.push({ resolve, reject })),
    requestFrame: callback => { rafs.set(++rafId, callback); return rafId; },
    cancelFrame: id => { rafs.delete(id); },
    draw: bitmap => { drawn.push((bitmap as Bitmap).id); },
    send: message => { sent.push(message as typeof sent[number]); },
    recovered: () => { recovered++; },
    failed: error => { errors.push(error.reason); },
  }, true);
  return { player, jobs, sent, drawn, errors, rafs, now: (value: number) => { now = value; },
    recovered: () => recovered,
    paint: () => { const callbacks = [...rafs.values()]; rafs.clear(); callbacks.forEach(callback => callback()); } };
}

test('bounds decoding, coalesces pending PNGs and renders only the newest ready bitmap on RAF', async () => {
  const f = fixture();
  f.player.receive(packet(1)); f.player.receive(packet(2)); f.player.receive(packet(3));
  assert.equal(f.jobs.length, 1);
  assert.deepEqual(f.sent, [{ type: 'ack', id: 2 }]);
  const first = new Bitmap(1); f.jobs[0].resolve(first); await tick();
  assert.equal(f.jobs.length, 2); assert.equal(f.rafs.size, 1); assert.deepEqual(f.drawn, []);
  const third = new Bitmap(3); f.jobs[1].resolve(third); await tick();
  assert.equal(first.closed, true); assert.equal(f.rafs.size, 1);
  f.paint();
  assert.deepEqual(f.drawn, [3]); assert.equal(third.closed, true);
  assert.deepEqual(f.sent.map(message => message.id), [2, 1, 3]);
  assert.equal(f.recovered(), 1);
  f.now(11000); const metrics = f.player.metrics();
  assert.equal(metrics.receivedFps, 3); assert.equal(metrics.renderedFps, 1);
  assert.equal(metrics.discarded, 2); assert.equal(metrics.skipped, 0);
  assert.deepEqual(f.errors, []);
});

test('hidden tabs release held images and cannot draw an old decode on resume', async () => {
  const f = fixture(); f.player.receive(packet(1));
  f.player.visibility(false); f.player.receive(packet(2)); f.player.visibility(true);
  const old = new Bitmap(1); f.jobs[0].resolve(old); await tick();
  assert.equal(old.closed, true); assert.equal(f.rafs.size, 0);
  f.player.receive(packet(6)); const current = new Bitmap(6); f.jobs[1].resolve(current); await tick();
  f.player.visibility(false);
  assert.equal(current.closed, true); f.paint(); assert.deepEqual(f.drawn, []);
  assert.equal(f.player.metrics().skipped, 3);
  assert.deepEqual(f.sent.filter(message => message.type === 'ack').map(message => message.id), [2, 1, 6]);
});

test('closing retires asynchronous callbacks, frees resources and never ACKs a new connection', async () => {
  const f = fixture(); f.player.receive(packet(1)); f.player.close();
  const bitmap = new Bitmap(1); f.jobs[0].resolve(bitmap); await tick();
  f.player.receive(new ArrayBuffer(0)); f.paint();
  assert.equal(bitmap.closed, true); assert.deepEqual(f.sent, []);
  assert.deepEqual(f.errors, []); assert.deepEqual(f.drawn, []);
});

test('a decoder failure closes the stream with a diagnostic and frees the queued bitmap', async () => {
  const f = fixture(); f.player.receive(packet(1));
  const first = new Bitmap(1); f.jobs[0].resolve(first); await tick();
  f.player.receive(packet(2)); f.jobs[1].reject(new Error('bad PNG')); await tick();
  assert.deepEqual(f.errors, ['invalid-frame png-decode']); assert.equal(first.closed, true);
  assert.equal(f.rafs.size, 0); assert.equal(f.recovered(), 0);
});

test('latency uses clock offset and excludes server reply handling time from RTT', async () => {
  const f = fixture(); f.player.clock();
  f.now(10022); f.player.receive(JSON.stringify({ type: 'clock', id: 0, client_ms: 10000, received_ms: 15010, sent_ms: 15012 }));
  f.now(10030); f.player.receive(packet(1, [14000, 14900, 14901, 15020]));
  f.now(10033); f.jobs[0].resolve(new Bitmap(1)); await tick();
  f.now(10045); f.paint();
  f.now(11000); const metrics = f.player.metrics();
  assert.equal(metrics.rttMs, 20); assert.equal(metrics.clockUncertaintyMs, 10);
  assert.equal(metrics.decodeMs, 3); assert.equal(metrics.receiveToDrawMs, 15);
  assert.equal(metrics.completeToDrawMs, 145); assert.equal(metrics.startToDrawMs, 1045);
  assert.equal(metrics.transportMs, 10); assert.deepEqual(f.errors, []);
});

test('out-of-order frames fail without acknowledging the invalid ID', () => {
  const f = fixture(); f.player.receive(packet(4)); f.player.receive(packet(4));
  assert.deepEqual(f.errors, ['invalid-frame order=4']); assert.deepEqual(f.sent, []);
});
