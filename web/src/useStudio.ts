import { useCallback, useEffect, useRef, useState } from 'react';
import { command, initialize, request, sendImage } from './api';
import type { Operation } from './api';
import type { ActionResult, Snapshot } from './api-types';
import { ActionQueue } from './ActionQueue';
import { FramePlayer, emptyMetrics } from './FramePlayer';

export function useStudio() {
  const [state, setState] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const renderer = useRef<CanvasRenderingContext2D | null>(null);
  const [playback, setPlayback] = useState(emptyMetrics);
  const [queue] = useState(() => new ActionQueue());

  useEffect(() => {
    if (!connected || !state?.lease.mine) queue.cancel();
  }, [connected, state?.lease.mine, queue]);

  useEffect(() => {
    let disposed = false;
    let sockets: WebSocket[] = [];
    let retry: ReturnType<typeof setTimeout> | null = null;
    let heartbeat: ReturnType<typeof setInterval> | null = null;
    let player: FramePlayer | null = null;
    const metricsTimer = setInterval(() => setPlayback(player ? player.metrics() : emptyMetrics), 1000);
    let attempts = 0;
    let reconnecting = false;
    const abort = new AbortController();
    const clearSockets = () => {
      player?.close();
      player = null;
      for (const socket of sockets) {
        socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
        socket.close();
      }
      sockets = [];
      if (heartbeat !== null) clearInterval(heartbeat);
    };
    const scheduleReconnect = () => {
      if (disposed || reconnecting) return;
      reconnecting = true;
      setConnected(false);
      queue.cancel();
      clearSockets();
      retry = setTimeout(() => { reconnecting = false; void connect(); }, Math.min(5000, 300 * 2 ** attempts++));
    };
    const visibility = () => player?.visibility(!document.hidden);
    document.addEventListener('visibilitychange', visibility);
    const connect = async () => {
      try {
        const initial = await initialize(abort.signal);
        if (disposed) return;
        setState(initial);
        const origin = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}`;
        const events = new WebSocket(`${origin}/api/events`);
        const frames = new WebSocket(`${origin}/api/frames`);
        frames.binaryType = 'arraybuffer';
        sockets = [events, frames];
        let receivedFrame = false;
        const playback = new FramePlayer({
          now: () => performance.timeOrigin + performance.now(),
          decode: png => createImageBitmap(new Blob([png], { type: 'image/png' })),
          requestFrame: callback => requestAnimationFrame(callback),
          cancelFrame: id => cancelAnimationFrame(id),
          draw: bitmap => {
            if (!renderer.current) throw new Error('Image canvas is not mounted');
            renderer.current.drawImage(bitmap as ImageBitmap, 0, 0);
          },
          send: message => {
            if (frames.readyState === WebSocket.OPEN) frames.send(JSON.stringify(message));
          },
          recovered: () => {
            if (!receivedFrame) { receivedFrame = true; attempts = 0; setConnectionError(null); }
          },
          failed: failure => {
            console.error('Image stream rejected:', failure.reason);
            setConnectionError(failure.message);
            frames.close(4002, failure.reason);
            scheduleReconnect();
          },
        }, !document.hidden);
        player = playback;
        for (const socket of sockets) {
          socket.onopen = () => {
            if (socket === frames) { playback.visibility(!document.hidden); playback.clock(); }
            if (sockets.every(s => s.readyState === WebSocket.OPEN)) setConnected(true);
          };
          socket.onclose = scheduleReconnect;
          socket.onerror = () => socket.close();
        }
        events.onmessage = event => setState(JSON.parse(event.data) as Snapshot);
        frames.onmessage = event => playback.receive(event.data);
        let ticks = 0;
        heartbeat = setInterval(() => {
          if (frames.readyState === WebSocket.OPEN) playback.clock();
          if (++ticks % 5 === 0 && events.readyState === WebSocket.OPEN) events.send('ping');
        }, 2000);
      } catch (failure) {
        if (!disposed) { setConnectionError(String(failure)); scheduleReconnect(); }
      }
    };
    void connect();
    return () => {
      disposed = true; abort.abort(); queue.cancel();
      if (retry !== null) clearTimeout(retry);
      clearInterval(metricsTimer);
      document.removeEventListener('visibilitychange', visibility);
      clearSockets();
    };
  }, [queue]);

  const attachCanvas = useCallback((canvas: HTMLCanvasElement) => {
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Canvas 2D is unavailable');
    renderer.current = context;
    return () => { renderer.current = null; };
  }, []);

  const execute = useCallback(async (operation: () => Promise<ActionResult>) => {
    setError(null);
    try {
      const result = await operation();
      if (result.status === 'error' || result.status === 'cancelled') setError(result.message);
    } catch (failure) { setError(String(failure)); }
  }, []);

  const send = useCallback((operation: Operation) =>
    queue.submit(operation.kind, () => execute(() => command(operation))), [queue, execute]);
  const upload = useCallback((image: Blob) =>
    queue.submit('clip_image', () => execute(() => sendImage(image))), [queue, execute]);
  const clearConditions = async (scope: keyof Snapshot['conditions'] | 'all') => {
    await send(scope === 'all' ? { kind:'clear_conditions' } : { kind:'clear_condition', value:scope });
  };
  const control = useCallback(async (action: 'acquire' | 'release') => {
    setError(null);
    if (action === 'release') queue.cancel();
    try { setState(await request<Snapshot>(`/api/control/${action}`, { method: 'POST' })); }
    catch (failure) { setError(String(failure)); }
  }, [queue]);
  return { state, connected, error: connectionError ?? error, attachCanvas, playback, send, upload, control, clearConditions };
}
