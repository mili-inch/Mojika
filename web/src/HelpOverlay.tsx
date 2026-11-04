import { useEffect, useLayoutEffect, useRef, useState } from 'react';

export type HelpItem = { target: string; text: string };
type Box = { x: number; y: number; width: number; height: number };
const overlap = (a: Box, b: Box) => Math.max(0, Math.min(a.x+a.width, b.x+b.width)-Math.max(a.x,b.x))
  * Math.max(0, Math.min(a.y+a.height, b.y+b.height)-Math.max(a.y,b.y));

function placeNotes(anchors: Box[], sizes: Box[], width: number, height: number) {
  const placed: Box[] = [];
  for (let i=0; i<anchors.length; i++) {
    const anchor = anchors[i], size = sizes[i];
    const xs = width < 600 ? [12, width-size.width-12]
      : [anchor.x+anchor.width+12, anchor.x-size.width-12, anchor.x+(anchor.width-size.width)/2, 12, width-size.width-12];
    const ys = [anchor.y-size.height-12, anchor.y+(anchor.height-size.height)/2, anchor.y+anchor.height+12,
      12, height-size.height-12, ...anchors.flatMap(box => [box.y-size.height-12, box.y+box.height+12]),
      ...placed.flatMap(box => [box.y-size.height-8, box.y+box.height+8])];
    const candidates = xs.flatMap(x => ys.map(y => ({ width:size.width, height:size.height,
      x:Math.max(12, Math.min(width-size.width-12, x)), y:Math.max(12, Math.min(height-size.height-12, y)) })));
    const score = (box: Box) => placed.reduce((sum, other) => sum+overlap(box, {
      x:other.x-4, y:other.y-4, width:other.width+8, height:other.height+8,
    }),0)*1000 + anchors.reduce((sum, other) => sum+overlap(box,other),0)*50
      + Math.hypot(box.x+box.width/2-anchor.x-anchor.width/2, box.y+box.height/2-anchor.y-anchor.height/2);
    candidates.sort((a,b) => score(a)-score(b));
    placed.push(candidates[0]);
  }
  return placed;
}

export function HelpOverlay({ items, trigger, onClose }: { items: HelpItem[]; trigger: string; onClose: () => void }) {
  const root = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState<{ items:HelpItem[]; notes:Box[]; anchors:Box[] } | null>(null);
  const positioned = layout?.items === items ? layout : null;
  useEffect(() => {
    root.current!.focus({ preventScroll:true });
    return () => { document.getElementById(trigger)?.focus({ preventScroll:true }); };
  }, [trigger]);
  useLayoutEffect(() => {
    const anchors = items.map(item => {
      const element = document.getElementById(item.target);
      if (!element) throw new Error(`Missing help anchor: ${item.target}`);
      return element;
    });
    const notes = Array.from(root.current!.querySelectorAll<HTMLElement>('.help-note'));
    const measure = () => {
      const boxes = anchors.map(element => element.getBoundingClientRect());
      setLayout({ items, anchors:boxes, notes:placeNotes(boxes, notes.map(element => element.getBoundingClientRect()), innerWidth, innerHeight) });
    };
    measure();
    const observer = new ResizeObserver(measure);
    for (const element of [...anchors,...notes]) observer.observe(element);
    window.addEventListener('resize',measure);
    document.addEventListener('scroll',measure,true);
    return () => {
      observer.disconnect(); window.removeEventListener('resize',measure); document.removeEventListener('scroll',measure,true);
    };
  }, [items]);
  return <div ref={root} className="help-overlay" role="dialog" aria-modal="true" aria-label="操作ヒント。背景タップまたはEscapeで閉じる" tabIndex={-1}
    onKeyDown={event => { if (event.key === 'Tab') event.preventDefault(); }}
    onPointerDown={event => { if (event.target === event.currentTarget) { event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); } }}
    onPointerUp={event => { if (event.target === event.currentTarget) onClose(); }}>
    {positioned && <svg className="help-lines" aria-hidden="true">
      {positioned.notes.map((note,i) => {
        const anchor = positioned.anchors[i];
        return <g key={items[i].target}>
          <line x1={anchor.x+anchor.width/2} y1={anchor.y+anchor.height/2} x2={note.x+note.width/2} y2={note.y+note.height/2} />
          <rect x={anchor.x-3} y={anchor.y-3} width={anchor.width+6} height={anchor.height+6} rx={10} />
        </g>;
      })}
    </svg>}
    {items.map((item,i) => <div className="help-note" role="tooltip" data-anchor={item.target} key={item.target}
      style={positioned ? { left:positioned.notes[i].x, top:positioned.notes[i].y } : { visibility:'hidden' }}>{item.text}</div>)}
  </div>;
}
