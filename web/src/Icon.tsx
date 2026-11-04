import type { ButtonHTMLAttributes } from 'react';
import { useRef } from 'react';

const paths = {
  edit: 'm4 16 11-11 4 4L8 20H4v-4ZM13 7l4 4',
  text: 'M4 5h16M12 5v15M8 20h8',
  image: 'M4 4h16v16H4ZM4 16l5-5 4 4 3-3 4 4M15 8h.01',
  brush: 'm10 13 8-10 3 3-10 8M10 13l1 1c1 5-4 7-8 6 3-1 0-5 7-7Z',
  eraser: 'm4 12 8-9 9 8-8 9H9l-5-4v-4ZM8 8l9 8M12 20h9',
  clear: 'M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7',
  back: 'M20 4v7a4 4 0 0 1-4 4H4m5-5-5 5 5 5',
  help: 'M9 8a3 3 0 0 1 6 0c0 2-3 2-3 5M12 17h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z',
  clock: 'M12 6v6l4 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z',
  mix: 'M15 12a6 6 0 1 1-12 0 6 6 0 0 1 12 0ZM21 12a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z',
  play: 'm8 4 12 8-12 8V4Z',
  pause: 'M8 4v16M16 4v16',
  refresh: 'M20 7v5h-5M20 12a8 8 0 1 0-2.3 5.7',
  camera: 'M3 7h4l2-3h6l2 3h4v13H3ZM16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
  capture: 'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0Z',
  upload: 'M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6',
};

export type IconName = keyof typeof paths;

export function Icon({ name }: { name: IconName }) {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}

export function IconButton({ icon, label, marked = false, onActivate, ...props }: Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'onClick'> & {
  icon: IconName; label: string; marked?: boolean; onActivate: () => void;
}) {
  const press = useRef<{ id:number; x:number; y:number } | null>(null);
  return <button {...props} type="button" className="icon-button" aria-label={label} title={label} data-marked={marked}
    onPointerDown={event => {
      if (event.button !== 0 || !event.isPrimary || props.disabled) return;
      press.current = { id:event.pointerId, x:event.clientX, y:event.clientY };
      event.currentTarget.setPointerCapture(event.pointerId);
    }}
    onPointerUp={event => {
      const start = press.current; press.current = null;
      if (!start || start.id !== event.pointerId || props.disabled) return;
      const box = event.currentTarget.getBoundingClientRect();
      const inside = event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom;
      if (inside && Math.hypot(event.clientX-start.x, event.clientY-start.y) <= 8) onActivate();
    }}
    onPointerCancel={() => { press.current = null; }}
    onClick={event => { if (event.detail === 0 && !props.disabled) onActivate(); }}>
    <Icon name={icon} />
  </button>;
}
