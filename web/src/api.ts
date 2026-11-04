import type { Action, ActionResult, Snapshot } from './api-types';

export type Operation = Action['operation'];
export interface Frame {
  id: number; revision: number; startedMs: number; completedMs: number;
  encodedMs: number; sentMs: number; png: ArrayBuffer; bytes: number;
}

function requestId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('');
}

export async function request<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, credentials: 'same-origin' });
  const body = await response.json();
  if (!response.ok) {
    const detail = body.detail;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return body as T;
}

export function initialize(signal: AbortSignal) {
  return request<Snapshot>('/api/session', { method: 'POST', signal });
}

export function command(operation: Operation) {
  return request<ActionResult>('/api/actions', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId(), operation }) });
}

export function sendImage(image: Blob) {
  return request<ActionResult>(`/api/clip-image?request_id=${requestId()}`, {
    method: 'POST', headers: { 'Content-Type': image.type }, body: image });
}

export class FrameProtocolError extends Error {
  readonly reason: string;

  constructor(detail: string) {
    super('画像配信の形式が正しくありません。再接続しています。');
    this.reason = `invalid-frame ${detail}`;
  }
}

export function parseFrame(buffer: unknown): Frame {
  if (!(buffer instanceof ArrayBuffer)) {
    throw new FrameProtocolError(`type=${Object.prototype.toString.call(buffer)}`);
  }
  if (buffer.byteLength < 52) throw new FrameProtocolError(`header-bytes=${buffer.byteLength}`);
  const header = new DataView(buffer);
  const magic = header.getUint32(0, true);
  if (magic !== 0x504b4a4d) throw new FrameProtocolError(`magic=${magic.toString(16)}`);
  const length = header.getUint32(48, true);
  if (length < 67 || length > 100000 || buffer.byteLength !== 52 + length) {
    throw new FrameProtocolError(`bytes=${buffer.byteLength} payload=${length}`);
  }
  const width = header.getUint16(44, true), height = header.getUint16(46, true);
  if (width !== 256 || height !== 256) throw new FrameProtocolError(`size=${width}x${height}`);
  const times = [12, 20, 28, 36].map(offset => header.getFloat64(offset, true));
  if (times.some(value => !Number.isFinite(value) || value <= 0)) throw new FrameProtocolError('timestamp');
  const id = header.getUint32(4, true);
  if (id === 0) throw new FrameProtocolError('id=0');
  const png = buffer.slice(52), view = new DataView(png);
  if (view.getUint32(0) !== 0x89504e47 || view.getUint32(4) !== 0x0d0a1a0a ||
      view.getUint32(8) !== 13 || view.getUint32(12) !== 0x49484452 ||
      view.getUint32(16) !== width || view.getUint32(20) !== height ||
      view.getUint8(24) !== 8 || view.getUint8(25) !== 0 ||
      view.getUint32(length - 12) !== 0 || view.getUint32(length - 8) !== 0x49454e44 ||
      view.getUint32(length - 4) !== 0xae426082) throw new FrameProtocolError('png-header-or-end');
  return { id, revision: header.getUint32(8, true), startedMs: times[0], completedMs: times[1],
    encodedMs: times[2], sentMs: times[3], png, bytes: buffer.byteLength };
}
