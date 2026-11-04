import assert from 'node:assert/strict';
import test from 'node:test';
import { FrameProtocolError, parseFrame } from '../src/api.ts';
import { packet } from './frameFixture.ts';

test('decodes a PNG frame and all pipeline timestamps', () => {
  const data = packet(42), frame = parseFrame(data);
  assert.equal(frame.id, 42); assert.equal(frame.revision, 7);
  assert.deepEqual([frame.startedMs, frame.completedMs, frame.encodedMs, frame.sentMs], [1000, 2000, 2001, 2002]);
  assert.equal(frame.png.byteLength + 52, data.byteLength);
  assert.equal(new Uint8Array(frame.png)[0], 137);
});

function rejected(data: unknown, detail: string) {
  assert.throws(() => parseFrame(data), (error: unknown) => {
    assert.ok(error instanceof FrameProtocolError);
    assert.equal(error.reason, `invalid-frame ${detail}`);
    assert.ok(new TextEncoder().encode(error.reason).length <= 123);
    return true;
  });
}

test('rejects truncated and oversized messages before PNG decode', () => {
  for (const length of [0, 23, 51]) rejected(new ArrayBuffer(length), `header-bytes=${length}`);
  const data = packet(), length = data.byteLength;
  rejected(data.slice(0, -1), `bytes=${length - 1} payload=${length - 52}`);
  const oversized = new Uint8Array(length + 1); oversized.set(new Uint8Array(data));
  rejected(oversized.buffer, `bytes=${length + 1} payload=${length - 52}`);
});

test('rejects wrong message types and invalid headers', () => {
  rejected('private text', 'type=[object String]');
  rejected(new Blob(['private text']), 'type=[object Blob]');
  const badMagic = packet(); new DataView(badMagic).setUint32(0, 0, true);
  rejected(badMagic, 'magic=0');
  const badSize = packet(); new DataView(badSize).setUint16(44, 128, true);
  rejected(badSize, 'size=128x256');
  const badTime = packet(); new DataView(badTime).setFloat64(20, NaN, true);
  rejected(badTime, 'timestamp');
});

test('rejects damaged PNG signature, dimensions and end marker', () => {
  for (const offset of [52, 52 + 16, 52 + 24, packet().byteLength - 5]) {
    const data = packet(); new Uint8Array(data)[offset] ^= 255;
    rejected(data, 'png-header-or-end');
  }
});
