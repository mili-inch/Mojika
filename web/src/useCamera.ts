import { useEffect, useRef, useState } from 'react';

export function useCamera({ disabled, upload }: { disabled: boolean; upload: (image: Blob) => Promise<void> }) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const mounted = useRef(true);
  const allowed = useRef(!disabled);
  const generation = useRef(0);
  allowed.current = !disabled;
  const [active, setActive] = useState(false);
  const [opening, setOpening] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (disabled) { generation.current++; stream.current?.getTracks().forEach(track => track.stop()); stream.current = null; setActive(false); setOpening(false); setReady(false); setError(null); }
  }, [disabled]);
  useEffect(() => { mounted.current = true; generation.current++; return () => { mounted.current = false; stream.current?.getTracks().forEach(track => track.stop()); }; }, []);
  const start = async () => {
    if (disabled || opening || active) return;
    const current = generation.current;
    setError(null); setOpening(true);
    try {
      if (!navigator.mediaDevices) throw new Error('カメラはHTTPSまたはlocalhostで利用できます。');
      const media = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (!mounted.current || !allowed.current || generation.current !== current) { media.getTracks().forEach(track => track.stop()); return; }
      stream.current = media;
      video.current!.srcObject = stream.current;
      await video.current!.play();
      if (mounted.current && allowed.current && generation.current === current) setActive(true);
    } catch (failure) {
      if (mounted.current && allowed.current && generation.current === current) {
        stream.current?.getTracks().forEach(track => track.stop()); stream.current = null;
        setReady(false); setError(String(failure));
      }
    } finally {
      if (mounted.current && generation.current === current) setOpening(false);
    }
  };
  const capture = () => {
    if (!ready || disabled) return;
    const current = generation.current;
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 224;
    canvas.getContext('2d')!.drawImage(video.current!, 0, 0, 224, 224);
    canvas.toBlob(blob => {
      if (!mounted.current || !allowed.current || generation.current !== current) return;
      if (blob) void upload(blob); else setError('画像を作成できませんでした。');
    }, 'image/jpeg', .9);
  };
  return { video, active, opening, ready, error, start, capture, loaded:() => { if (allowed.current) setReady(true); } };
}
