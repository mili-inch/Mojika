import { useEffect, useMemo, useRef, useState } from 'react';
import { useCamera } from './useCamera';
import { FrameView } from './FrameView';
import { FrameStats } from './FrameStats';
import { HelpOverlay } from './HelpOverlay';
import type { HelpItem } from './HelpOverlay';
import { HintOverlay } from './HintOverlay';
import type { HintTool } from './HintOverlay';
import { Icon, IconButton } from './Icon';
import { ControlSlider } from './ControlSlider';
import { Prompt } from './Prompt';
import { useStudio } from './useStudio';
import type { Conditions } from './api-types';

type Condition = keyof Conditions;
type Tool = Condition | 'timing';
const clearLabels: Record<Condition, string> = { gpt:'GPT条件を解除', clip:'CLIP条件を解除', hint:'手描きヒントを解除' };
const contextHelp: Record<Tool | 'idle' | 'selection', HelpItem[]> = {
  idle: [
    { target:'control-acquire', text:'編集項目を開き、操作を開始します。' },
  ],
  selection: [
    { target:'tool-gpt', text:'文章の意味やイメージを指定します。' },
    { target:'tool-clip', text:'文章・画像で見た目を指定します。' },
    { target:'tool-hint', text:'手描きヒント：漢字の上に線を描き、形のヒントを伝えます。' },
    { target:'all-clear', text:'すべての条件を解除します。' },
    { target:'tool-timing', text:'遷移時間・新規ノイズの割合を設定します。' },
    { target:'control-release', text:'編集を閉じ、操作を終了します。' },
  ],
  gpt: [
    { target:'gpt-clear', text:'GPTの条件だけを解除します。' },
  ],
  clip: [
    { target:'clip-camera', text:'カメラ映像を開きます。' },
    { target:'clip-capture', text:'撮影した画像を条件にします。' },
    { target:'clip-upload', text:'画像ファイルを条件にします。' },
    { target:'clip-clear', text:'CLIPの文章・画像条件を解除します。' },
  ],
  hint: [
    { target:'hint-pen', text:'ペンで手描きヒントを描きます。' },
    { target:'hint-eraser', text:'なぞった部分の手描きヒントを消します。' },
    { target:'hint-width', text:'ペン・消しゴムの太さを調整します。' },
    { target:'hint-clear', text:'手描きヒントをすべて消して条件を解除します。' },
  ],
  timing: [
    { target:'timing-range', text:'ノイズと条件の変化にかける時間を調整します。' },
    { target:'noise-mix-range', text:'次の地点に混ぜる新規ノイズの割合です。小さいほど近い地点へ移り、1で独立したノイズ、0で同じ地点になります。' },
  ],
};
const backHelp: HelpItem = { target:'mode-back', text:'条件を保持して、モード選択へ戻ります。' };

export default function App() {
  const studio = useStudio();
  const { state, connected, send, control } = studio;
  const [tool, setTool] = useState<Tool | null>(null);
  const [help, setHelp] = useState(false);
  const [brushWidth, setBrushWidth] = useState(1);
  const [hintTool, setHintTool] = useState<HintTool>('draw');
  const [clearing, setClearing] = useState<Condition | 'all' | null>(null);
  const [epochs, setEpochs] = useState({ gpt:0, clip:0, hint:0 });
  const [clipUploading, setClipUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const canOperate = Boolean(connected && state?.phase === 'ready' && state.lease.mine);
  const disabled = !canOperate || clearing !== null;
  const fieldDisabled = (condition: Condition) => !canOperate || clearing === 'all' || clearing === condition;
  const uploadClip = async (image: Blob) => {
    setClipUploading(true);
    try { await studio.upload(image); } finally { setClipUploading(false); }
  };
  const camera = useCamera({ disabled:tool !== 'clip' || fieldDisabled('clip'), upload:uploadClip });
  const automatic = state?.controls.noise_auto;
  const hasState = state !== null;
  const helpItems = useMemo(() => {
    const context = !canOperate ? 'idle' : tool === null ? 'selection' : tool;
    const items = [...contextHelp[context]];
    if (canOperate && tool !== null) items.push(backHelp);
    if (hasState) items.push({ target:'frame-stats', text:'生成・受信・描画速度と遅延の計測を開きます。' });
    if (hasState) items.push({ target:'generation-toggle', text:canOperate ? 'ノイズの自動更新を停止・再開します。' : '編集を開始すると、自動更新を停止・再開できます。' });
    if (automatic === false) items.push({ target:'generation-refresh', text:'停止したまま、次のノイズへ一度だけ更新します。' });
    return items;
  }, [tool, automatic, canOperate, hasState]);
  useEffect(() => {
    if (!canOperate) setTool(null);
  }, [canOperate]);
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (help) { setHelp(false); return; }
      if (!canOperate || clearing !== null) return;
      if (tool !== null) setTool(null);
      else void control('release');
    };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, [tool, help, canOperate, clearing, control]);
  const clear = async (scope: Condition | 'all') => {
    setHelp(false); setClearing(scope);
    if (scope === 'all') setTool(null);
    setEpochs(previous => ({
      gpt:previous.gpt+Number(scope === 'all' || scope === 'gpt'),
      clip:previous.clip+Number(scope === 'all' || scope === 'clip'),
      hint:previous.hint+Number(scope === 'all' || scope === 'hint'),
    }));
    try { await studio.clearConditions(scope); } finally { setClearing(null); }
  };
  return <main className="studio">
    <div inert={help}>
    <div className="stage"><div className="frame-view">
      <FrameView attach={studio.attachCanvas} />
      {state && <HintOverlay key={epochs.hint} value={state.conditions.hint} enabled={!fieldDisabled('hint') && tool === 'hint' && !help} width={brushWidth} tool={hintTool}
        onApply={value => send({ kind:'hint', value })} />}
    </div></div>
    {state && <FrameStats state={state} playback={studio.playback} />}
    <div className="global-controls" role="toolbar" aria-label="更新とヒント">
      {state && <>
        <IconButton id="generation-toggle" icon={state.controls.noise_auto ? 'pause' : 'play'} label={state.controls.noise_auto ? '自動更新を停止' : '自動更新を開始'} disabled={disabled}
          onActivate={() => void send({ kind:'noise_auto', value:!state.controls.noise_auto })} />
        {!state.controls.noise_auto && <IconButton id="generation-refresh" icon="refresh" label="ノイズを手動更新" disabled={disabled} onActivate={() => void send({ kind:'noise' })} />}
      </>}
      <IconButton id="global-help" icon="help" label="操作ヒントを表示" onActivate={() => setHelp(true)} />
    </div>
    <div className="tool-dock" role="toolbar" aria-label="条件と更新設定">
      {!canOperate && <IconButton id="control-acquire" icon="edit" label="操作開始" disabled={!connected || state?.phase !== 'ready' || !!state?.lease.held} onActivate={() => void control('acquire')} />}
      {canOperate && state && tool === null && <>
        <IconButton id="tool-gpt" icon="text" label="GPT条件を編集" marked={state.conditions.gpt !== null} disabled={disabled} onActivate={() => setTool('gpt')} />
        <IconButton id="tool-clip" icon="image" label="CLIP条件を編集" marked={state.conditions.clip !== null} disabled={disabled} onActivate={() => setTool('clip')} />
        <IconButton id="tool-hint" icon="brush" label="手描きヒントを描く" marked={state.conditions.hint !== null} disabled={disabled} onActivate={() => setTool('hint')} />
        <IconButton id="all-clear" icon="clear" label="すべての条件を解除" disabled={disabled} onActivate={() => void clear('all')} />
        <span className="tool-divider" aria-hidden="true" />
        <IconButton id="tool-timing" icon="clock" label="更新を設定" disabled={disabled} onActivate={() => setTool('timing')} />
        <span className="tool-divider" aria-hidden="true" />
        <IconButton id="control-release" icon="back" label="操作終了" disabled={disabled} onActivate={() => void control('release')} />
      </>}
      {canOperate && tool !== null && <>
        {tool === 'hint' && <>
          <IconButton id="hint-pen" icon="brush" label="ペンで描く" aria-pressed={hintTool === 'draw'} disabled={disabled} onActivate={() => setHintTool('draw')} />
          <IconButton id="hint-eraser" icon="eraser" label="消しゴムで消す" aria-pressed={hintTool === 'erase'} disabled={disabled} onActivate={() => setHintTool('erase')} />
        </>}
        {tool === 'clip' && <>
          <IconButton id="clip-camera" icon="camera" label="カメラを開始" disabled={disabled || camera.active || camera.opening} onActivate={() => void camera.start()} />
          <IconButton id="clip-capture" icon="capture" label="撮影して適用" disabled={disabled || clipUploading || !camera.ready} onActivate={camera.capture} />
          <IconButton id="clip-upload" icon="upload" label="画像ファイルを選択" disabled={disabled || clipUploading} onActivate={() => fileInput.current!.click()} />
        </>}
        {tool !== 'timing' && <IconButton id={`${tool}-clear`} icon="clear" label={clearLabels[tool]} disabled={disabled} onActivate={() => void clear(tool)} />}
        <IconButton id="mode-back" icon="back" label="モード選択に戻る" disabled={disabled} onActivate={() => setTool(null)} />
      </>}
    </div>
    {canOperate && state && <>
      <div className="tool-popover" role="region" aria-label="GPT条件" hidden={tool !== 'gpt'}>
        <Prompt key={epochs.gpt} label="GPTプロンプト" value={state.conditions.gpt === null ? '' : state.conditions.gpt} disabled={fieldDisabled('gpt')} onApply={value => send({ kind:'gpt', value })} />
      </div>
      <div className="tool-popover" role="region" aria-label="CLIP条件" hidden={tool !== 'clip'}>
        <Prompt key={epochs.clip} label="CLIPプロンプト" value={!clipUploading && state.conditions.clip?.kind === 'text' ? state.conditions.clip.text : ''} disabled={fieldDisabled('clip') || clipUploading} onApply={value => send({ kind:'clip_text', value })} />
        {tool === 'clip' && <div className="clip-media">
          {state.conditions.clip?.kind === 'image' && <img className="clip-preview" alt="CLIP画像" src={state.conditions.clip.preview} />}
          <video ref={camera.video} muted playsInline hidden={!camera.active} onLoadedData={camera.loaded} />
          <input ref={fileInput} type="file" aria-label="画像ファイル" accept="image/jpeg,image/png,image/webp" disabled={fieldDisabled('clip') || clipUploading} hidden
            onChange={event => { const file = event.currentTarget.files?.[0]; if (file) void uploadClip(file); event.currentTarget.value = ''; }} />
          {camera.error && <p role="alert">{camera.error}</p>}
        </div>}
      </div>
      {tool === 'hint' && <div className="tool-popover" role="region" aria-label="手描きヒント設定">
        <div className="brush-width"><span id="hint-width"><Icon name={hintTool === 'draw' ? 'brush' : 'eraser'} /></span>
          <input type="range" aria-label={hintTool === 'draw' ? '線の太さ' : '消しゴムの太さ'} min={.1} max={6} step={.1} value={brushWidth} disabled={fieldDisabled('hint')} onChange={event => setBrushWidth(event.target.valueAsNumber)} />
          <output>{brushWidth.toFixed(1)}</output>
        </div>
      </div>}
      <div className="tool-popover" role="region" aria-label="更新設定" hidden={tool !== 'timing'}>
        <ControlSlider setting="transition_seconds" value={state.controls.transition_seconds} disabled={disabled} onApply={value => send({ kind:'transition_seconds', value })} />
        <ControlSlider setting="noise_mix" value={state.controls.noise_mix} disabled={disabled} onApply={value => send({ kind:'noise_mix', value })} />
      </div>
    </>}
    {(studio.error || state?.error) && <p className="error" role="alert">{studio.error || state?.error}</p>}
    </div>
    {help && <HelpOverlay items={helpItems} trigger="global-help" onClose={() => setHelp(false)} />}
  </main>;
}
