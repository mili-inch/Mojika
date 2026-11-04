import { FrameProtocolError, parseFrame } from './api.ts';
import type { Frame } from './api.ts';

interface Bitmap { width: number; height: number; close(): void }
interface Received { frame: Frame; received: number }
interface Ready extends Received { bitmap: Bitmap; decodeMs: number }
interface Hooks {
  now(): number;
  decode(png: ArrayBuffer): Promise<Bitmap>;
  requestFrame(callback: () => void): number;
  cancelFrame(id: number): void;
  draw(bitmap: Bitmap): void;
  send(message: object): void;
  recovered(): void;
  failed(error: FrameProtocolError): void;
}
export interface PlaybackMetrics {
  receivedFps: number; renderedFps: number; mbps: number;
  skipped: number; discarded: number; decodeMs: number | null;
  receiveToDrawMs: number | null; completeToDrawMs: number | null;
  startToDrawMs: number | null; transportMs: number | null;
  rttMs: number | null; clockUncertaintyMs: number | null;
}
export const emptyMetrics: PlaybackMetrics = {
  receivedFps: 0, renderedFps: 0, mbps: 0, skipped: 0, discarded: 0,
  decodeMs: null, receiveToDrawMs: null, completeToDrawMs: null,
  startToDrawMs: null, transportMs: null, rttMs: null, clockUncertaintyMs: null,
};

/** One decode, one replaceable pending PNG and one replaceable decoded bitmap. */
export class FramePlayer {
  private hooks: Hooks;
  private closed = false;
  private visible: boolean;
  private epoch = 0;
  private pending: Received | null = null;
  private ready: Ready | null = null;
  private decoding = false;
  private raf: number | null = null;
  private lastId = 0;
  private received = 0;
  private rendered = 0;
  private bytes = 0;
  private skipped = 0;
  private discarded = 0;
  private measuredAt: number;
  private durations: number[][] = [[], [], [], [], []];
  private clockId = 0;
  private clocks = new Map<number, number>();
  private samples: { offset: number; rtt: number }[] = [];

  constructor(hooks: Hooks, visible: boolean) {
    this.hooks = hooks;
    this.visible = visible;
    this.measuredAt = hooks.now();
  }

  visibility(visible: boolean) {
    if (this.closed) return;
    this.visible = visible;
    this.hooks.send({ type: 'visibility', visible });
    if (!visible) {
      this.epoch++;
      this.clearImages(true);
    }
  }

  clock() {
    if (this.closed || this.clocks.size >= 8) return;
    const id = this.clockId++, client_ms = this.hooks.now();
    this.clocks.set(id, client_ms);
    this.hooks.send({ type: 'clock', id, client_ms });
  }

  receive(data: unknown) {
    if (this.closed) return;
    const received = this.hooks.now();
    try {
      if (typeof data === 'string') {
        const message = JSON.parse(data);
        const sent = this.clocks.get(message.id);
        if (message.type !== 'clock' || sent === undefined || message.client_ms !== sent ||
            !Number.isFinite(message.received_ms) || !Number.isFinite(message.sent_ms) ||
            message.sent_ms < message.received_ms) throw new FrameProtocolError('clock');
        this.clocks.delete(message.id);
        const rtt = received - sent - (message.sent_ms - message.received_ms);
        if (rtt < -1) throw new FrameProtocolError('clock-rtt');
        this.samples.push({ rtt: Math.max(0, rtt), offset: (message.received_ms - sent + message.sent_ms - received) / 2 });
        if (this.samples.length > 8) this.samples.shift();
        return;
      }
      const frame = parseFrame(data);
      if (frame.id <= this.lastId) throw new FrameProtocolError(`order=${frame.id}`);
      if (this.lastId) this.skipped += frame.id - this.lastId - 1;
      this.lastId = frame.id;
      this.received++; this.bytes += frame.bytes;
      if (!this.visible) { this.discard(frame.id); return; }
      if (this.pending) this.discard(this.pending.frame.id);
      this.pending = { frame, received };
      this.decode();
    } catch (error) {
      this.fail(error instanceof FrameProtocolError ? error : new FrameProtocolError('control-json'));
    }
  }

  private decode() {
    if (this.closed || this.decoding || !this.pending) return;
    const item = this.pending, epoch = this.epoch, started = this.hooks.now();
    this.pending = null;
    this.decoding = true;
    void this.hooks.decode(item.frame.png).then(bitmap => {
      if (this.closed) { bitmap.close(); return; }
      if (bitmap.width !== 256 || bitmap.height !== 256) {
        bitmap.close(); throw new FrameProtocolError('decoded-size');
      }
      if (epoch !== this.epoch || !this.visible) {
        bitmap.close(); this.discard(item.frame.id); return;
      }
      if (this.ready) { this.ready.bitmap.close(); this.discard(this.ready.frame.id); }
      this.ready = { ...item, bitmap, decodeMs: this.hooks.now() - started };
      if (this.raf === null) this.raf = this.hooks.requestFrame(() => this.paint());
    }).catch(error => {
      this.fail(error instanceof FrameProtocolError ? error : new FrameProtocolError('png-decode'));
    }).finally(() => { this.decoding = false; this.decode(); });
  }

  private paint() {
    this.raf = null;
    const item = this.ready;
    this.ready = null;
    if (this.closed || !item) return;
    try {
      this.hooks.draw(item.bitmap);
      const drawn = this.hooks.now();
      this.rendered++;
      this.durations[0].push(item.decodeMs);
      this.durations[1].push(drawn - item.received);
      const clock = this.bestClock();
      if (clock) {
        this.durations[2].push(drawn + clock.offset - item.frame.completedMs);
        this.durations[3].push(drawn + clock.offset - item.frame.startedMs);
        this.durations[4].push(item.received + clock.offset - item.frame.sentMs);
      }
      this.ack(item.frame.id);
      this.hooks.recovered();
    } catch {
      this.fail(new FrameProtocolError('canvas-draw'));
    } finally { item.bitmap.close(); }
  }

  private bestClock() {
    return this.samples.reduce<{ offset: number; rtt: number } | null>(
      (best, sample) => !best || sample.rtt < best.rtt ? sample : best, null);
  }

  metrics(): PlaybackMetrics {
    const now = this.hooks.now(), seconds = (now - this.measuredAt) / 1000;
    const average = (values: number[]) => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    const clock = this.bestClock();
    const result = {
      receivedFps: seconds > 0 ? this.received / seconds : 0,
      renderedFps: seconds > 0 ? this.rendered / seconds : 0,
      mbps: seconds > 0 ? this.bytes * 8 / seconds / 1e6 : 0,
      skipped: this.skipped, discarded: this.discarded,
      decodeMs: average(this.durations[0]), receiveToDrawMs: average(this.durations[1]),
      completeToDrawMs: average(this.durations[2]), startToDrawMs: average(this.durations[3]),
      transportMs: average(this.durations[4]), rttMs: clock?.rtt ?? null,
      clockUncertaintyMs: clock ? clock.rtt / 2 : null,
    };
    this.received = this.rendered = this.bytes = 0;
    this.measuredAt = now;
    this.durations = [[], [], [], [], []];
    return result;
  }

  private ack(id: number) { this.hooks.send({ type: 'ack', id }); }
  private discard(id: number) { this.discarded++; this.ack(id); }
  private fail(error: FrameProtocolError) {
    if (this.closed) return;
    this.close();
    this.hooks.failed(error);
  }
  private clearImages(ack: boolean) {
    if (this.raf !== null) this.hooks.cancelFrame(this.raf);
    this.raf = null;
    if (this.pending && ack) this.discard(this.pending.frame.id);
    if (this.ready) {
      this.ready.bitmap.close();
      if (ack) this.discard(this.ready.frame.id);
    }
    this.pending = this.ready = null;
  }
  close() { this.closed = true; this.clearImages(false); }
}
