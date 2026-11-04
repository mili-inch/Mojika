import { useEffect, useRef, useState } from 'react';
import type { Snapshot } from './api-types';
import type { PlaybackMetrics } from './FramePlayer';

const ms = (value: number | null) => value === null ? '—' : `${value.toFixed(1)} ms`;

export function FrameStats({ state, playback }: { state: Snapshot; playback: PlaybackMetrics }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.stopImmediatePropagation(); setOpen(false); }
    };
    document.addEventListener('pointerdown', outside);
    window.addEventListener('keydown', escape, true);
    return () => {
      document.removeEventListener('pointerdown', outside);
      window.removeEventListener('keydown', escape, true);
    };
  }, [open]);
  return <div className="frame-stats" ref={root}>
    <button id="frame-stats" className="fps" aria-label="配信の計測" aria-expanded={open}
      aria-controls="frame-statistics" onClick={() => setOpen(!open)}>{playback.renderedFps.toFixed(1)} FPS</button>
    {open && <div id="frame-statistics" className="stats-popover" role="region" aria-label="配信の計測詳細">
      <dl>
        <dt>生成 / 受信 / 描画</dt><dd>{state.fps.toFixed(1)} / {playback.receivedFps.toFixed(1)} / {playback.renderedFps.toFixed(1)} FPS</dd>
        <dt>受信量</dt><dd>{playback.mbps.toFixed(2)} Mbps</dd>
        <dt>PNGサイズ</dt><dd>{(state.pipeline.frame_bytes / 1024).toFixed(1)} KiB</dd>
        <dt>GPU変換・転送</dt><dd>{ms(state.pipeline.transfer_ms)}</dd>
        <dt>PNG圧縮 / 展開</dt><dd>{ms(state.pipeline.encode_ms)} / {ms(playback.decodeMs)}</dd>
        <dt>受信 → 描画</dt><dd>{ms(playback.receiveToDrawMs)}</dd>
        <dt>送信 → 受信（推定）</dt><dd>{ms(playback.transportMs)}</dd>
        <dt>GPU完了確認 → 描画（推定）</dt><dd>{ms(playback.completeToDrawMs)}</dd>
        <dt>生成開始 → 描画（推定）</dt><dd>{ms(playback.startToDrawMs)}</dd>
        <dt>RTT / 時刻推定幅 ±</dt><dd>{ms(playback.rttMs)} / {ms(playback.clockUncertaintyMs)}</dd>
        <dt>送信省略 / 描画省略</dt><dd>{playback.skipped} / {playback.discarded}</dd>
      </dl>
    </div>}
  </div>;
}
