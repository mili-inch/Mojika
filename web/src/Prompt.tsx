import { useEffect, useRef, useState } from 'react';

export function Prompt({ label, value, disabled, onApply }: {
  label: string; value: string; disabled: boolean; onApply: (text: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState(value);
  const [focused, setFocused] = useState(false);
  const [busy, setBusy] = useState(false);
  const composing = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const generation = useRef(0);
  useEffect(() => { if (disabled || (!focused && !busy)) setDraft(value); }, [value, focused, busy, disabled]);
  useEffect(() => () => { if (timer.current !== null) clearTimeout(timer.current); }, []);
  useEffect(() => {
    if (disabled) {
      generation.current++;
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null; composing.current = false;
      setBusy(false); setFocused(false);
    }
  }, [disabled]);
  const schedule = (text: string) => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
    const current = ++generation.current;
    if (composing.current || !text.trim()) { setBusy(false); return; }
    setBusy(true);
    timer.current = setTimeout(() => {
      timer.current = null;
      void onApply(text).finally(() => { if (generation.current === current) setBusy(false); });
    }, 300);
  };
  return <textarea className="prompt" aria-label={label} placeholder={label} value={draft} disabled={disabled}
    maxLength={4000} rows={2} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
    onCompositionStart={() => { composing.current = true; if (timer.current !== null) clearTimeout(timer.current); }}
    onCompositionEnd={event => { composing.current = false; schedule(event.currentTarget.value); }}
    onChange={event => { setDraft(event.target.value); schedule(event.target.value); }} />;
}
