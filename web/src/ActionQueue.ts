import type { Operation } from './api';

type Kind = Operation['kind'] | 'clip_image';
type Pending = {
  key: string | null;
  run: () => Promise<void>;
  resolve: () => void;
  reject: (error: unknown) => void;
};

/** One request in flight; coalesce unsent edits without crossing button actions. */
export class ActionQueue {
  private waiting: Pending[] = [];
  private active = false;

  submit(kind: Kind, run: () => Promise<void>): Promise<void> {
    const key = kind === 'noise' || kind === 'clear_condition' || kind === 'clear_conditions'
      ? null : kind === 'clip_text' || kind === 'clip_image' ? 'clip' : kind;
    if (key !== null) {
      for (let i = this.waiting.length - 1; i >= 0; i--) {
        const previous = this.waiting[i];
        if (previous.key === null) break;
        if (previous.key === key) {
          this.waiting.splice(i, 1);
          previous.resolve();
          break;
        }
      }
    }
    const result = new Promise<void>((resolve, reject) => {
      this.waiting.push({ key, run, resolve, reject });
    });
    void this.drain();
    return result;
  }

  cancel() {
    for (const pending of this.waiting.splice(0)) pending.resolve();
  }

  private async drain() {
    if (this.active) return;
    this.active = true;
    try {
      let pending: Pending | undefined;
      while ((pending = this.waiting.shift()) !== undefined) {
        try { await pending.run(); pending.resolve(); }
        catch (error) { pending.reject(error); }
      }
    } finally { this.active = false; }
  }
}
