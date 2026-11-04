import { useEffect, useRef, useState } from 'react';
import { Icon } from './Icon';

const settings = {
  transition_seconds: {
    label:'遷移時間', icon:'clock' as const, helpId:'timing-range', min:0.1, max:60,
    format:(value: number) => `${value.toFixed(1)} 秒`,
  },
  noise_mix: {
    label:'新規ノイズの割合', icon:'mix' as const, helpId:'noise-mix-range', min:0, max:1,
    format:(value: number) => value.toFixed(1),
  },
};

export function ControlSlider({ setting, value, disabled, onApply }: {
  setting: keyof typeof settings; value: number; disabled: boolean; onApply: (value: number) => Promise<void>;
}) {
  const config = settings[setting];
  const [draft, setDraft] = useState<number | null>(null);
  const latest = useRef<number | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragging = useRef(false);
  const submitting = useRef(false);
  const generation = useRef(0);
  useEffect(() => () => { if (timer.current !== null) clearTimeout(timer.current); }, []);
  useEffect(() => {
    if (disabled) {
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null; latest.current = null; dragging.current = false;
      submitting.current = false; generation.current++; setDraft(null);
    }
  }, [disabled]);
  const submit = () => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
    const next = latest.current;
    latest.current = null;
    if (next === null) return;
    const current = ++generation.current;
    submitting.current = true;
    void onApply(next).finally(() => {
      if (generation.current !== current) return;
      submitting.current = false;
      if (latest.current === null && !dragging.current) setDraft(null);
    });
  };
  const finish = () => {
    dragging.current = false;
    if (latest.current !== null) submit();
    else if (!submitting.current) setDraft(null);
  };
  const current = draft === null ? value : draft;
  return <label className="field control-slider"><span><span id={config.helpId} title={config.label}><Icon name={config.icon} /></span><output>{config.format(current)}</output></span>
    <input type="range" aria-label={config.label} aria-valuetext={config.format(current)}
      min={config.min} max={config.max} step={0.1} value={current} disabled={disabled}
      onPointerDown={event => { dragging.current = true; event.currentTarget.setPointerCapture(event.pointerId); }}
      onPointerUp={finish} onPointerCancel={finish} onBlur={finish}
      onChange={event => {
        const next = event.target.valueAsNumber;
        setDraft(next); latest.current = next;
        if (timer.current === null) timer.current = setTimeout(submit, 80);
      }} />
  </label>;
}
