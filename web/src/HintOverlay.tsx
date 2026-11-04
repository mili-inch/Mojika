import { useEffect, useRef, useState } from 'react';
import type { PointerEvent } from 'react';

export type HintTool = 'draw' | 'erase';

function paint(canvas: HTMLCanvasElement, pixels: number[]) {
  const context = canvas.getContext('2d')!;
  const image = context.createImageData(16, 16);
  for (let i = 0; i < 256; i++) {
    image.data[4*i] = 190; image.data[4*i+1] = 235; image.data[4*i+2] = 145;
    image.data[4*i+3] = Math.round((255-pixels[i]) * .85);
  }
  context.putImageData(image, 0, 0);
}

export function HintOverlay({ value, enabled, width, tool, onApply }: {
  value: number[] | null; enabled: boolean; width: number; tool: HintTool;
  onApply: (value: number[]) => Promise<void>;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const raster = useRef<HTMLCanvasElement | null>(null);
  const drawing = useRef<{ id: number; point: [number, number]; color: string; width: number } | null>(null);
  const generation = useRef(0);
  const [busy, setBusy] = useState(false);
  const latest = useRef(value);
  latest.current = value;
  const restore = () => {
    const pixels = latest.current === null ? Array<number>(256).fill(255) : latest.current;
    const context = raster.current!.getContext('2d')!;
    const image = context.createImageData(16, 16);
    pixels.forEach((pixel, i) => { image.data.set([pixel, pixel, pixel, 255], 4*i); });
    context.putImageData(image, 0, 0); paint(canvas.current!, pixels);
  };
  useEffect(() => {
    const source = document.createElement('canvas'); source.width = source.height = 16;
    raster.current = source;
    restore();
  }, []);
  useEffect(() => {
    if (!enabled) { drawing.current = null; generation.current++; setBusy(false); }
    if (!drawing.current && (!busy || !enabled)) restore();
  }, [value, enabled, busy]);
  const pixels = () => {
    const bytes = raster.current!.getContext('2d')!.getImageData(0, 0, 16, 16).data;
    return Array.from({ length:256 }, (_, i) => bytes[4*i]);
  };
  const point = (event: PointerEvent<HTMLCanvasElement>): [number, number] => {
    const box = event.currentTarget.getBoundingClientRect();
    return [(event.clientX-box.left)*16/box.width, (event.clientY-box.top)*16/box.height];
  };
  const move = (event: PointerEvent<HTMLCanvasElement>) => {
    if (!enabled || drawing.current?.id !== event.pointerId) return;
    const next = point(event), context = raster.current!.getContext('2d')!;
    context.strokeStyle = drawing.current.color; context.lineWidth = drawing.current.width; context.lineCap = 'round';
    context.beginPath(); context.moveTo(...drawing.current.point); context.lineTo(...next); context.stroke();
    drawing.current.point = next;
    paint(canvas.current!, pixels());
  };
  return <canvas ref={canvas} width={16} height={16} className="hint-overlay" aria-label="手描きヒント" hidden={!enabled}
    onPointerDown={event => {
      if (!enabled || drawing.current || event.button !== 0) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      const next = point(event), context = raster.current!.getContext('2d')!;
      const color = tool === 'draw' ? '#000' : '#fff';
      drawing.current = { id:event.pointerId, point:next, color, width };
      context.fillStyle = color; context.beginPath(); context.arc(...next, width/2, 0, Math.PI*2); context.fill();
      paint(canvas.current!, pixels());
    }} onPointerMove={move}
    onPointerUp={event => {
      if (drawing.current?.id !== event.pointerId) return;
      move(event); drawing.current = null;
      if (!enabled) return;
      const current = ++generation.current; setBusy(true);
      void onApply(pixels()).finally(() => { if (generation.current === current) setBusy(false); });
    }} onPointerCancel={event => {
      if (drawing.current?.id !== event.pointerId) return;
      drawing.current = null; restore();
    }} />;
}
