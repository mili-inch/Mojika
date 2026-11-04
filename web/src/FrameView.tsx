import { useEffect, useRef } from 'react';

export function FrameView({ attach }: { attach: (canvas: HTMLCanvasElement) => () => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => attach(canvas.current!), [attach]);
  return <canvas className="generated-frame" ref={canvas} width={256} height={256} aria-label="生成画像" />;
}
