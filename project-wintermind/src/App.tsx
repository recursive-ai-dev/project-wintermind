import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  WintermindModel,
  DataPipeline,
  TrainStep,
  ModelConfig,
  DEFAULT_CONFIG,
} from './core/model';
import { buildBootstrapVocab, encode, BPEVocab, tokenizeForDisplay } from './core/bpe';
import { GSARModule } from './core/gsar';
import { LayerParamInfo } from './core/rnn';

// ── Training corpus ────────────────────────────────────────────────────────────
const TRAINING_CORPUS = `
the quick brown fox jumps over the lazy dog
winter is cold and the snow falls softly on the ground
neural networks learn by adjusting weights through backpropagation
language models predict the next token given prior context
attention mechanisms allow models to focus on relevant context
the loss function measures how far predictions are from truth
gradient descent minimizes loss by following the negative gradient
tokenization breaks text into subword units for efficient processing
embeddings map tokens into dense continuous vector representations
the model learns to generalize patterns from training data
recurrent networks maintain hidden state across sequence positions
transformer architectures use self-attention for context modeling
byte pair encoding creates efficient subword vocabularies
zero redundancy optimizer shards memory across parallel processes
nesterov momentum uses a look-ahead gradient for faster convergence
hessian curvature provides second-order information about the loss surface
kernel fusion combines operations to reduce memory bandwidth overhead
symbolic reasoning identifies concept patterns without redundant computation
self-explanatory perception monitors prediction confidence in real time
the winter wind blows cold through frozen snow covered trees
ice and frost cover the ground in the bitter winter morning
the neural model processes sequential data with learned representations
language understanding requires context and pattern recognition
training deep models requires careful optimization and regularization
the gradient flows backward through the computational graph
mixed precision training uses fp16 and fp32 for efficiency
distributed training splits the model across multiple compute nodes
`.trim();

// ── Colour helpers ─────────────────────────────────────────────────────────────

function statusDot(s: string): string {
  if (s === 'training') return 'bg-emerald-400';
  if (s === 'building') return 'bg-yellow-400';
  if (s === 'done')     return 'bg-sky-400';
  return 'bg-slate-600';
}

function lossColor(loss: number): string {
  if (loss < 3)  return 'text-emerald-400';
  if (loss < 5)  return 'text-yellow-400';
  return 'text-red-400';
}

// ── App ────────────────────────────────────────────────────────────────────────

type Tab = 'train' | 'generate' | 'diagnostics' | 'gsar' | 'sep';

export function App() {
  const [vocab,         setVocab]         = useState<BPEVocab | null>(null);
  const [model,         setModel]         = useState<WintermindModel | null>(null);
  const [status,        setStatus]        = useState<'idle' | 'building' | 'training' | 'done'>('idle');
  const [trainLog,      setTrainLog]      = useState<TrainStep[]>([]);
  const [modelSummary,  setModelSummary]  = useState('');
  const [diagnostics,   setDiagnostics]   = useState<Record<string, unknown>>({});
  const [prompt,        setPrompt]        = useState('the winter snow');
  const [generated,     setGenerated]     = useState('');
  const [gsarText,      setGsarText]      = useState('');
  const [activeTab,     setActiveTab]     = useState<Tab>('train');
  const [config,        setConfig]        = useState<ModelConfig>({ ...DEFAULT_CONFIG });
  const [maxSteps,      setMaxSteps]      = useState(100);
  const [corpusText,    setCorpusText]    = useState(TRAINING_CORPUS);
  const [sepLog,        setSepLog]        = useState<string[]>([]);
  const [paramInfo,     setParamInfo]     = useState<LayerParamInfo[]>([]);
  const [buildError,    setBuildError]    = useState('');

  const intervalRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepRef      = useRef(0);
  const modelRef     = useRef<WintermindModel | null>(null);
  const pipelineRef  = useRef<DataPipeline | null>(null);
  const vocabRef     = useRef<BPEVocab | null>(null);
  const maxStepsRef  = useRef(maxSteps);
  maxStepsRef.current = maxSteps;

  // ── Build ──────────────────────────────────────────────────────────────────

  const buildModel = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStatus('building');
    setTrainLog([]);
    setSepLog([]);
    setGenerated('');
    setBuildError('');
    stepRef.current = 0;

    setTimeout(() => {
      try {
        const v           = buildBootstrapVocab();
        const actualVocab = v.tokenToId.size;
        const cfg: ModelConfig = { ...config, vocabSize: actualVocab };
        const m           = new WintermindModel(cfg, maxStepsRef.current);
        const dp          = new DataPipeline(corpusText, v, 12);

        vocabRef.current    = v;
        modelRef.current    = m;
        pipelineRef.current = dp;

        setVocab(v);
        setModel(m);
        setModelSummary(m.summary());
        setDiagnostics(m.diagnostics());
        setParamInfo(m.parameterInspector());
        setStatus('idle');
      } catch (err) {
        setBuildError(String(err));
        setStatus('idle');
      }
    }, 40);
  }, [config, corpusText]);

  // ── Training loop ──────────────────────────────────────────────────────────

  const startTraining = useCallback(() => {
    if (!modelRef.current || !pipelineRef.current || !vocabRef.current) return;
    setStatus('training');

    intervalRef.current = setInterval(() => {
      const m  = modelRef.current!;
      const dp = pipelineRef.current!;
      const v  = vocabRef.current!;

      if (stepRef.current >= maxStepsRef.current) {
        clearInterval(intervalRef.current!);
        setStatus('done');
        setDiagnostics(m.diagnostics());
        setModelSummary(m.summary());
        setParamInfo(m.parameterInspector());
        return;
      }

      const batch = dp.nextBatch();
      try {
        const result = m.trainStep(batch.inputIds, batch.targetIds, v);
        stepRef.current = result.step;

        setTrainLog(prev => [...prev, result].slice(-300));

        if (result.step % 10 === 0) {
          setDiagnostics(m.diagnostics());
          setParamInfo(m.parameterInspector());
        }

        if (result.step % 12 === 0) {
          const micro = result.sepAnalysis?.microPrediction;
          const full  = result.sepAnalysis?.fullPrediction;
          const line  = [
            `Step ${String(result.step).padStart(4)}`,
            `loss=${result.loss.toFixed(4)}`,
            `ppl=${result.perplexity.toFixed(1).padStart(7)}`,
            micro ? `micro="${micro.tokenStr}" @${(micro.confidence * 100).toFixed(1)}%` : '',
            full  ? `full="${full.tokenStr}"  @${(full.confidence * 100).toFixed(1)}%` : '',
            `Δ=${result.sepDelta >= 0 ? '+' : ''}${(result.sepDelta * 100).toFixed(1)}%`,
            `GSAR=${(result.gsarRatio * 100).toFixed(0)}%`,
          ].filter(Boolean).join('  │  ');
          setSepLog(prev => [...prev.slice(-60), line]);
        }
      } catch (e) {
        console.error('Train step error:', e);
      }
    }, 50);
  }, []);

  const stopTraining = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStatus('done');
    if (modelRef.current) {
      setDiagnostics(modelRef.current.diagnostics());
      setModelSummary(modelRef.current.summary());
      setParamInfo(modelRef.current.parameterInspector());
    }
  }, []);

  // ── Generation ─────────────────────────────────────────────────────────────

  const runGenerate = useCallback(() => {
    if (!model || !vocab) return;
    const promptIds = encode(prompt, vocab);
    const result    = model.generate(promptIds, vocab, 40, 0.85, 10);
    setGenerated(result.text || '(empty — try more training steps)');
  }, [model, vocab, prompt]);

  // ── GSAR ───────────────────────────────────────────────────────────────────

  const runGSAR = useCallback(() => {
    const g      = new GSARModule();
    const tokens = prompt.toLowerCase().split(/\s+/).filter(Boolean);
    if (tokens.length === 0) return;
    const result = g.reason(tokens);

    const lines: string[] = [
      `Input:  "${prompt}"`,
      `Tokens: [${tokens.join(', ')}]  (${tokens.length} total)`,
      '',
      'GSAR Segment Analysis',
      '─'.repeat(64),
    ];
    for (const seg of result.segments) {
      const badge = seg.symbolic ? '✓ SYMBOLIC SKIP' : '○ DENSE PROCESS';
      const pri   = (seg.priority * 100).toFixed(0).padStart(3);
      lines.push(`  ${badge}  pri=${pri}%  [${seg.tokens.join(' ')}]`);
      lines.push(`         → ${seg.conceptId ? `concept: "${seg.label}"` : 'no concept match'}`);
    }
    lines.push('');
    lines.push('Summary');
    lines.push('─'.repeat(64));
    lines.push(`  Total tokens:          ${tokens.length}`);
    lines.push(`  Symbolically skipped:  ${result.skippedTokens}`);
    lines.push(`  Dense processed:       ${result.processedTokens}`);
    lines.push(`  Compute savings:       ${(g.allocationRatio(result) * 100).toFixed(1)}%`);
    lines.push(`  Active concept nodes:  [${result.nodesActivated.join(', ')}]`);
    lines.push('');
    lines.push('Token Priority Map');
    lines.push('─'.repeat(64));
    tokens.forEach((tok, idx) => {
      const pri = result.priorityMap.get(idx) ?? 0;
      const filled = Math.round(pri * 20);
      const bar = '█'.repeat(filled) + '░'.repeat(20 - filled);
      lines.push(`  [${String(idx).padStart(2)}] ${tok.padEnd(16)} ${bar}  ${(pri * 100).toFixed(0)}%`);
    });
    setGsarText(lines.join('\n'));
  }, [prompt]);

  // ── Cleanup ────────────────────────────────────────────────────────────────

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  // ── Derived chart data ─────────────────────────────────────────────────────

  const lossHistory  = useMemo(() => trainLog.map(s => s.loss),                     [trainLog]);
  const pplHistory   = useMemo(() => trainLog.map(s => Math.min(s.perplexity, 500)), [trainLog]);
  const gradHistory  = useMemo(() => trainLog.map(s => s.gradNorm),                 [trainLog]);
  const lrHistory    = useMemo(() => trainLog.map(s => s.lr),                       [trainLog]);
  const latest       = trainLog[trainLog.length - 1] ?? null;
  const progress     = maxSteps > 0 ? Math.min((stepRef.current / maxSteps) * 100, 100) : 0;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-[#07090f] text-slate-100 font-mono flex flex-col select-none">

      {/* ══ Header ══ */}
      <header className="border-b border-slate-800/80 bg-[#0b0f18] px-5 py-2.5 flex items-center justify-between shrink-0 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-400 via-blue-600 to-indigo-700 flex items-center justify-center text-white font-black text-base shadow-lg shadow-cyan-900/40">
            W
          </div>
          <div>
            <h1 className="text-[13px] font-bold text-white tracking-[0.18em] uppercase">
              Project Wintermind
            </h1>
            <p className="text-[9px] text-slate-500 tracking-widest uppercase">
              BPE · LSTM · Transformer · NAG · Hessian · ZeRO · GSAR · SEP · Kernel Fusion · Mixed-Precision
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {latest && (
            <div className="hidden md:flex items-center gap-4 text-[10px] mr-3">
              <Pill label="step"    value={String(latest.step)}                     color="text-slate-300" />
              <Pill label="loss"    value={latest.loss.toFixed(4)}                  color={lossColor(latest.loss)} />
              <Pill label="ppl"     value={latest.perplexity.toFixed(1)}            color="text-purple-400" />
              <Pill label="‖∇‖"     value={latest.gradNorm.toFixed(4)}              color="text-amber-400" />
              <Pill label="GSAR"    value={`${(latest.gsarRatio * 100).toFixed(0)}%`} color="text-blue-400" />
            </div>
          )}
          <span className={`w-2 h-2 rounded-full ${statusDot(status)} ${status === 'training' ? 'animate-pulse' : ''}`} />
          <span className="text-[10px] text-slate-400 uppercase tracking-widest">{status}</span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* ══ Sidebar ══ */}
        <aside className="w-64 border-r border-slate-800/80 bg-[#0b0f18] flex flex-col overflow-y-auto shrink-0">
          <div className="p-4 space-y-5">

            <Section title="Model Architecture">
              <CfgRow label="Embed Dim"    value={config.embedDim}              min={16}  max={128} step={16} disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, embedDim: v, hiddenDim: v }))} />
              <CfgRow label="FF Dim"       value={config.ffDim}                 min={32}  max={256} step={32} disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, ffDim: v }))} />
              <CfgRow label="Blocks"       value={config.numTransformerBlocks}  min={1}   max={4}   step={1}  disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, numTransformerBlocks: v }))} />
            </Section>

            <Section title="Optimizer">
              <CfgRow label="LR (×10⁻⁴)"       value={Math.round(config.lr * 10000)}       min={1}  max={50}  step={1}  disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, lr: v / 10000 }))} />
              <CfgRow label="Momentum (%)"      value={Math.round(config.momentum * 100)}   min={50} max={99}  step={1}  disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, momentum: v / 100 }))} />
              <CfgRow label="Grad Clip (×0.1)"  value={Math.round(config.clipGradNorm * 10)} min={1}  max={50}  step={1}  disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, clipGradNorm: v / 10 }))} />
              <CheckRow label="Hessian Diag"    checked={config.useHessian}    disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, useHessian: v }))} />
              <CheckRow label="FP16 Precision"  checked={config.dtype === 'fp16'} disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, dtype: v ? 'fp16' : 'fp32' }))} />
            </Section>

            <Section title="Memory / Dist.">
              <CfgRow label="ZeRO Stage"  value={config.zeroStage}  min={0}  max={3}  step={1}  disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, zeroStage: v as 0|1|2|3 }))} />
              <CfgRow label="Num Ranks"   value={config.numRanks}   min={1}  max={16} step={1}  disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, numRanks: v }))} />
            </Section>

            <Section title="SEP Config">
              <CfgRow label="Micro Batch"  value={config.microBatchSize} min={2} max={16} step={1} disabled={status === 'training'} onChange={v => setConfig(c => ({ ...c, microBatchSize: v }))} />
              <CfgRow label="Train Steps"  value={maxSteps}              min={20} max={500} step={10} disabled={status === 'training'} onChange={v => setMaxSteps(v)} />
            </Section>

            <Section title="Training Corpus">
              <textarea
                rows={5}
                value={corpusText}
                onChange={e => setCorpusText(e.target.value)}
                disabled={status === 'training'}
                className="w-full text-[10px] bg-[#131720] border border-slate-700/70 rounded p-2 text-slate-400 resize-y leading-relaxed focus:outline-none focus:border-cyan-800"
              />
            </Section>

            <Section title="Pipeline Controls">
              <Btn onClick={buildModel}    disabled={status === 'training'}                                   color="slate">① Build / Rebuild</Btn>
              <Btn onClick={startTraining} disabled={status === 'training' || status === 'building' || !model} color="cyan">② Start Training</Btn>
              <Btn onClick={stopTraining}  disabled={status !== 'training'}                                   color="red">⏹ Stop</Btn>
            </Section>

            {buildError && (
              <div className="text-[10px] text-red-400 bg-red-950/30 rounded p-2 border border-red-900/50 break-all">
                {buildError}
              </div>
            )}

            {/* Progress bar */}
            {(status === 'training' || status === 'done') && maxSteps > 0 && (
              <div>
                <div className="flex justify-between text-[9px] text-slate-500 mb-1">
                  <span>Progress</span>
                  <span>{stepRef.current}/{maxSteps}</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-600 to-blue-500 rounded-full transition-all duration-200"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Live metrics */}
            {latest && (
              <Section title="Live Metrics">
                <Stat label="Step"       value={latest.step} />
                <Stat label="Loss"       value={latest.loss.toFixed(5)} />
                <Stat label="Perplexity" value={latest.perplexity.toFixed(2)} />
                <Stat label="‖∇‖₂"      value={latest.gradNorm.toFixed(5)} />
                <Stat label="‖θ‖₂"      value={latest.paramNorm.toFixed(3)} />
                <Stat label="LR"         value={latest.lr.toExponential(3)} />
                <Stat label="GSAR skip"  value={`${(latest.gsarRatio * 100).toFixed(0)}%`} />
                <Stat label="SEP Δ"      value={`${latest.sepDelta >= 0 ? '+' : ''}${(latest.sepDelta * 100).toFixed(1)}%`} />
                <Stat label="ZeRO Mem"   value={`${(latest.zeroMemFactor * 100).toFixed(0)}%/rank`} />
                <Stat label="Predicted"  value={`"${latest.prediction}"`} />
                <Stat label="Target"     value={`"${latest.targetToken}"`} />
              </Section>
            )}
          </div>
        </aside>

        {/* ══ Main Panel ══ */}
        <main className="flex-1 flex flex-col overflow-hidden">

          {/* Tabs */}
          <nav className="flex border-b border-slate-800/80 bg-[#0b0f18] shrink-0 overflow-x-auto">
            {([
              ['train',       '📉 Training'],
              ['generate',    '🔮 Generate'],
              ['diagnostics', '🔬 Diagnostics'],
              ['gsar',        '🧠 GSAR'],
              ['sep',         '👁 SEP'],
            ] as [Tab, string][]).map(([tab, label]) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-5 py-3 text-[10px] font-semibold uppercase tracking-widest whitespace-nowrap transition border-b-2 ${
                  activeTab === tab
                    ? 'border-cyan-500 text-cyan-400 bg-[#0e1521]'
                    : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>

          <div className="flex-1 overflow-y-auto p-5 space-y-5">

            {/* ══════════════════════════════════════════════════════════════════
                TRAIN TAB
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'train' && (
              <>
                {modelSummary && (
                  <Panel>
                    <pre className="text-[10px] text-cyan-300 whitespace-pre leading-relaxed overflow-x-auto">{modelSummary}</pre>
                  </Panel>
                )}

                {!model && status === 'idle' && !buildError && (
                  <EmptyState>← Configure in the sidebar and click "Build / Rebuild" to initialize the model</EmptyState>
                )}

                {lossHistory.length > 1 && (
                  <div className="grid grid-cols-2 gap-4">
                    <ChartCard title="Cross-Entropy Loss"  color="#22d3ee" data={lossHistory} yLabel="nats" />
                    <ChartCard title="Perplexity"          color="#a78bfa" data={pplHistory}  yLabel="exp(L)" />
                    <ChartCard title="Gradient Norm  ‖∇‖₂" color="#f59e0b" data={gradHistory} yLabel="L2" />
                    <ChartCard title="Learning Rate"       color="#34d399" data={lrHistory}   yLabel="lr" />
                  </div>
                )}

                {latest && latest.topKPredictions.length > 0 && (
                  <Panel>
                    <PanelTitle>Top-K Prediction Distribution — step {latest.step}</PanelTitle>
                    <div className="space-y-1.5 mt-2">
                      {latest.topKPredictions.map((p, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <span className="text-[10px] text-slate-600 w-4 text-right">{i + 1}.</span>
                          <span className="text-[10px] text-slate-300 w-24 truncate font-mono">"{p.str}"</span>
                          <div className="flex-1 bg-slate-800/80 rounded-full h-2 overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all duration-300"
                              style={{
                                width: `${(p.prob * 100).toFixed(1)}%`,
                                background: i === 0
                                  ? 'linear-gradient(90deg,#06b6d4,#3b82f6)'
                                  : 'linear-gradient(90deg,#475569,#334155)',
                              }}
                            />
                          </div>
                          <span className="text-[10px] w-10 text-right tabular-nums" style={{ color: i === 0 ? '#22d3ee' : '#64748b' }}>
                            {(p.prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </Panel>
                )}

                {trainLog.length > 0 && (
                  <Panel>
                    <PanelTitle>Training Log — last {Math.min(trainLog.length, 25)} steps</PanelTitle>
                    <div className="overflow-x-auto mt-2">
                      <table className="w-full text-[10px]">
                        <thead>
                          <tr className="text-slate-600 border-b border-slate-800">
                            {['Step', 'Loss', 'PPL', '‖∇‖', 'LR', 'GSAR%', 'ZeRO%', 'SEP Δ', 'Pred', 'Target'].map(h => (
                              <th key={h} className="px-2.5 py-2 text-left font-semibold">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {trainLog.slice(-25).reverse().map(s => (
                            <tr key={s.step} className="border-b border-slate-900/60 hover:bg-slate-800/20 transition">
                              <td className="px-2.5 py-1.5 text-slate-500">{s.step}</td>
                              <td className={`px-2.5 py-1.5 tabular-nums ${lossColor(s.loss)}`}>{s.loss.toFixed(4)}</td>
                              <td className="px-2.5 py-1.5 text-purple-400 tabular-nums">{s.perplexity.toFixed(1)}</td>
                              <td className="px-2.5 py-1.5 text-amber-400  tabular-nums">{s.gradNorm.toFixed(4)}</td>
                              <td className="px-2.5 py-1.5 text-emerald-400 tabular-nums">{s.lr.toExponential(2)}</td>
                              <td className="px-2.5 py-1.5 text-blue-400">{(s.gsarRatio * 100).toFixed(0)}%</td>
                              <td className="px-2.5 py-1.5 text-indigo-400">{(s.zeroMemFactor * 100).toFixed(0)}%</td>
                              <td className="px-2.5 py-1.5 text-sky-400">{s.sepDelta >= 0 ? '+' : ''}{(s.sepDelta * 100).toFixed(1)}%</td>
                              <td className="px-2.5 py-1.5 text-slate-300 max-w-[70px] truncate">"{s.prediction}"</td>
                              <td className="px-2.5 py-1.5 text-slate-600 max-w-[70px] truncate">"{s.targetToken}"</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Panel>
                )}

                {sepLog.length > 0 && (
                  <Panel>
                    <PanelTitle>SEP Perception Log</PanelTitle>
                    <div className="mt-2 space-y-0.5 max-h-52 overflow-y-auto">
                      {sepLog.map((line, i) => (
                        <div key={i} className="text-[10px] text-slate-500 font-mono leading-relaxed py-0.5 border-b border-slate-900/40 last:border-0">
                          {line}
                        </div>
                      ))}
                    </div>
                  </Panel>
                )}
              </>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                GENERATE TAB
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'generate' && (
              <>
                <Panel>
                  <PanelTitle>Autoregressive Text Generation</PanelTitle>
                  <p className="text-[10px] text-slate-500 mb-4 leading-relaxed">
                    Prompt → BPE encode → EmbeddingLayer → TransformerBlocks → LSTMCell warm-up
                    → top-K nucleus sample (T=0.85, K=10) → BPE decode. Generation stops at &lt;eos&gt; or 40 tokens.
                  </p>
                  <div className="flex gap-2">
                    <input
                      className="flex-1 bg-[#131720] border border-slate-700/70 rounded px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:outline-none focus:border-cyan-700 transition"
                      placeholder="Enter prompt..."
                      value={prompt}
                      onChange={e => setPrompt(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && model && runGenerate()}
                    />
                    <button
                      onClick={runGenerate}
                      disabled={!model}
                      className="px-5 py-2 rounded bg-cyan-900/80 hover:bg-cyan-800 border border-cyan-700/40 text-[11px] font-semibold text-cyan-200 disabled:opacity-30 transition"
                    >
                      Generate →
                    </button>
                  </div>
                  {generated && (
                    <div className="mt-4 bg-[#0d1520] rounded-lg p-4 border border-emerald-800/40">
                      <div className="text-[9px] text-emerald-600 mb-2 uppercase tracking-widest">Generated Output</div>
                      <div className="text-sm text-emerald-300 leading-relaxed whitespace-pre-wrap font-mono">{generated}</div>
                    </div>
                  )}
                  {!model && (
                    <div className="mt-3 text-[10px] text-slate-600">Build and train the model first →</div>
                  )}
                </Panel>

                {vocab && (
                  <Panel>
                    <PanelTitle>BPE Tokenization Inspector</PanelTitle>
                    <div className="flex gap-4 text-[10px] text-slate-500 mb-3">
                      <span>Vocab: <span className="text-white font-semibold">{vocab.tokenToId.size}</span> tokens</span>
                      <span>Merges: <span className="text-white font-semibold">{vocab.merges.length}</span></span>
                    </div>
                    {prompt.trim() && (
                      <>
                        <div className="bg-[#131720] rounded p-3 text-[10px] border border-slate-700/50 space-y-2 mb-3">
                          <div>
                            <span className="text-slate-600">encode() → ids: </span>
                            <span className="text-amber-300 break-all">[{encode(prompt, vocab).join(', ')}]</span>
                          </div>
                          <div>
                            <span className="text-slate-600">tokens: </span>
                            <span className="text-cyan-300 break-all">
                              [{encode(prompt, vocab).map(id => `"${vocab.idToToken.get(id) ?? '?'}"`).join(', ')}]
                            </span>
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-1.5 mb-3">
                          {tokenizeForDisplay(prompt, vocab).map((tok, i) => (
                            <span key={i} className="bg-slate-800 border border-slate-700/60 rounded px-2 py-0.5 text-[10px] text-slate-200">
                              {tok}
                            </span>
                          ))}
                        </div>
                      </>
                    )}
                    <div className="text-[9px] text-slate-600 uppercase tracking-widest mb-2">Top 24 BPE Merges (in training order)</div>
                    <div className="flex flex-wrap gap-1">
                      {vocab.merges.slice(0, 24).map(([a, b], i) => (
                        <span key={i} className="bg-[#131720] border border-slate-700/40 rounded px-2 py-0.5 text-[9px] text-slate-400">
                          "{a}" + "{b}"
                        </span>
                      ))}
                    </div>
                  </Panel>
                )}
              </>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                DIAGNOSTICS TAB
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'diagnostics' && (
              <>
                {trainLog.length > 0 && (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    <MetricCard label="Total Params"    value={String(diagnostics?.paramCount ?? '—')}                       color="text-white" />
                    <MetricCard label="Best Loss"       value={lossHistory.length ? Math.min(...lossHistory).toFixed(5) : '—'} color="text-cyan-400" />
                    <MetricCard label="Best PPL"        value={pplHistory.length  ? Math.min(...pplHistory).toFixed(2)  : '—'} color="text-purple-400" />
                    <MetricCard label="ZeRO Stage"      value={`Stage ${config.zeroStage}`}                                   color="text-cyan-400" />
                    <MetricCard label="Mem / Rank"      value={model ? `${(model.optimizer.memoryReductionFactor() * 100).toFixed(0)}%` : '—'} color="text-emerald-400" />
                    <MetricCard label="Avg Grad Norm"   value={gradHistory.length ? (gradHistory.reduce((a, b) => a + b, 0) / gradHistory.length).toFixed(5) : '—'} color="text-amber-400" />
                  </div>
                )}

                <Panel>
                  <PanelTitle>Architecture Diagram</PanelTitle>
                  <ArchDiagram config={config} />
                </Panel>

                {model && (
                  <Panel>
                    <PanelTitle>ZeRO Memory Analysis</PanelTitle>
                    <ZeroPanel config={config} model={model} />
                  </Panel>
                )}

                {paramInfo.length > 0 && (
                  <Panel>
                    <PanelTitle>Parameter Inspector — {paramInfo.reduce((s, p) => s + p.paramCount, 0).toLocaleString()} total parameters</PanelTitle>
                    <div className="overflow-x-auto mt-2">
                      <table className="w-full text-[10px]">
                        <thead>
                          <tr className="text-slate-600 border-b border-slate-800">
                            {['Layer', 'Shape', 'Params', '‖W‖₂', '‖∇‖₂'].map(h => (
                              <th key={h} className="px-2.5 py-2 text-left font-semibold">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {paramInfo.map((p, i) => (
                            <tr key={i} className="border-b border-slate-900/60 hover:bg-slate-800/20">
                              <td className="px-2.5 py-1.5 text-cyan-400 font-mono">{p.name}</td>
                              <td className="px-2.5 py-1.5 text-slate-400">[{p.shape.join('×')}]</td>
                              <td className="px-2.5 py-1.5 text-slate-300 tabular-nums">{p.paramCount.toLocaleString()}</td>
                              <td className="px-2.5 py-1.5 text-amber-400 tabular-nums">{p.weightNorm.toFixed(4)}</td>
                              <td className="px-2.5 py-1.5 tabular-nums" style={{ color: p.gradNorm > 0.01 ? '#f59e0b' : '#475569' }}>
                                {p.gradNorm.toFixed(5)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Panel>
                )}

                <Panel>
                  <PanelTitle>Kernel Fusion Trace</PanelTitle>
                  <div className="space-y-1 mt-2">
                    {[
                      { op: 'add  (a + b)',             fused: true,  note: 'single-pass kernelFusedBinary — no intermediate allocation' },
                      { op: 'sub  (a − b)',             fused: true,  note: 'single-pass kernelFusedBinary — negated gradient path' },
                      { op: 'mul  (a ⊙ b)',             fused: true,  note: 'single-pass kernelFusedBinary — hadamard product' },
                      { op: 'scale (a · k)',            fused: true,  note: 'single-pass Float32Array map — no alloc' },
                      { op: 'relu  max(0, x)',          fused: true,  note: 'in-place element mask — grad = indicator(x>0)' },
                      { op: 'tanh  tanh(x)',            fused: true,  note: 'grad = (1 − out²) — reuses forward buffer' },
                      { op: 'sigmoid  σ(x)',            fused: true,  note: 'grad = out·(1−out) — reuses forward buffer' },
                      { op: 'softmax  exp/norm',        fused: true,  note: 'numerically stable: max-shift → exp → div in one pass' },
                      { op: 'layernorm mean+var+norm',  fused: true,  note: 'mean, variance, normalize in single row pass per token' },
                      { op: 'matmul  M·K·N',           fused: false, note: 'O(MNK) GEMM — real GPU: cuBLAS WMMA tensor core' },
                      { op: 'LSTM gates (i/f/g/o)',     fused: true,  note: 'all 4 gates fused in single raw[4H] loop' },
                    ].map((row, i) => (
                      <div key={i} className="flex items-center gap-3 py-1 border-b border-slate-900/40 last:border-0">
                        <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${row.fused ? 'bg-emerald-500' : 'bg-amber-500'}`} />
                        <span className="w-44 text-[10px] text-slate-300 font-mono shrink-0">{row.op}</span>
                        <span className="text-[10px] text-slate-600">{row.note}</span>
                      </div>
                    ))}
                  </div>
                </Panel>

                <Panel>
                  <PanelTitle>Full Diagnostics JSON</PanelTitle>
                  <pre className="text-[10px] text-slate-500 whitespace-pre-wrap overflow-x-auto max-h-72 leading-relaxed">
                    {JSON.stringify(diagnostics, null, 2)}
                  </pre>
                </Panel>
              </>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                GSAR TAB
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'gsar' && (
              <>
                <Panel>
                  <PanelTitle>GSAR — General Symbolic Arrays Reasoning</PanelTitle>
                  <p className="text-[10px] text-slate-500 mb-4 leading-relaxed">
                    Each token sequence is analyzed against a self-training registry of concept nodes.
                    Tokens whose concept's priority score exceeds θ=0.5 are routed to symbolic processing,
                    avoiding redundant dense computation. The registry updates its priority scores online
                    with each hit, and co-occurrence patterns can be registered dynamically.
                  </p>
                  <div className="flex gap-2">
                    <input
                      className="flex-1 bg-[#131720] border border-slate-700/70 rounded px-3 py-2 text-sm text-white placeholder:text-slate-600 focus:outline-none focus:border-purple-700 transition"
                      placeholder="Enter text to analyze with GSAR..."
                      value={prompt}
                      onChange={e => setPrompt(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && runGSAR()}
                    />
                    <button
                      onClick={runGSAR}
                      className="px-5 py-2 rounded bg-purple-900/70 hover:bg-purple-800 border border-purple-700/40 text-[11px] font-semibold text-purple-200 transition"
                    >
                      Analyze →
                    </button>
                  </div>
                  {gsarText && (
                    <pre className="mt-4 bg-[#0d1117] rounded-lg p-4 text-[10px] text-slate-300 whitespace-pre leading-relaxed border border-slate-800/60 overflow-x-auto max-h-[28rem]">
                      {gsarText}
                    </pre>
                  )}
                </Panel>

                {model && (
                  <Panel>
                    <PanelTitle>Live Concept Registry — {model.gsar.getConcepts().length} nodes</PanelTitle>
                    <div className="space-y-2 mt-2">
                      {model.gsar.getConcepts().map(concept => (
                        <div key={concept.id} className="bg-[#0e1420] border border-slate-800/60 rounded-lg p-3 flex gap-4 justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="text-[11px] text-cyan-300 font-semibold">{concept.conceptLabel}</div>
                            <div className="text-[9px] text-slate-500 mt-1 truncate">
                              {concept.tokens.slice(0, 12).join(' · ')}{concept.tokens.length > 12 ? ` +${concept.tokens.length - 12}` : ''}
                            </div>
                            <div className="text-[9px] text-slate-700 mt-0.5">
                              {concept.synonymGroups.length} synonym groups  ·  {concept.tokens.length} tokens
                            </div>
                          </div>
                          <div className="text-right shrink-0">
                            <div className="text-[13px] text-amber-400 font-bold tabular-nums">{(concept.priority * 100).toFixed(0)}%</div>
                            <div className="text-[9px] text-slate-600">priority</div>
                            <div className="text-[10px] text-slate-400 mt-1">{concept.hitCount} hits</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Panel>
                )}

                <Panel>
                  <PanelTitle>GSAR Mathematical Specification</PanelTitle>
                  <div className="text-[10px] text-slate-400 space-y-2.5 leading-relaxed">
                    <MathRow label="Priority update"    eq="p(c, t) = min( p₀(c) + α·hits(c,t),  1.0 )" note="α=0.01 for unigrams, 0.02 for bigrams" />
                    <MathRow label="Bigram key"         eq="k(a,b) = a ⊕ U+001F ⊕ b"                     note="unit separator — O(1) hash lookup in ngramIndex" />
                    <MathRow label="Skip condition"     eq="skip(t) = ∃c ∈ C : t ∈ c  ∧  p(c) ≥ θ"      note="θ = 0.5  (priority threshold)" />
                    <MathRow label="Allocation ratio"   eq="α = |skipped| / (|skipped| + |processed|)"   note="fractional compute savings" />
                    <MathRow label="Co-occ. learning"   eq="learn(a,b,c) → ngramIdx[k(a,b)] = c"         note="online bigram concept registration" />
                    <MathRow label="Match priority"     eq="bigram match always supersedes unigram match" note="longer match wins — greedy left-to-right scan" />
                  </div>
                </Panel>
              </>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                SEP TAB
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'sep' && (
              <>
                <Panel>
                  <PanelTitle>SEP — Self-Explanatory Perception</PanelTitle>
                  <p className="text-[10px] text-slate-500 mb-4 leading-relaxed">
                    SEP operates on each forward pass in two phases. <span className="text-slate-300">Micro-prediction</span> aggregates
                    logits from the first N tokens (micro-batch boundary) to form a preliminary hypothesis.
                    <span className="text-slate-300"> Full-prediction</span> uses all logits. The confidence delta between them
                    reveals whether additional context strengthened or weakened the prediction. Spurious
                    correlations are flagged before any output is finalized. Token attributions are
                    computed via logit sensitivity — each token's logit deviation from mean, normalized 0→1.
                  </p>
                </Panel>

                {latest?.sepAnalysis && (
                  <SEPDeepDive analysis={latest.sepAnalysis} step={latest.step} />
                )}

                {!latest && (
                  <EmptyState>Train the model to populate SEP analysis</EmptyState>
                )}

                <Panel>
                  <PanelTitle>SEP Mathematical Specification</PanelTitle>
                  <div className="text-[10px] text-slate-400 space-y-2.5 leading-relaxed">
                    <MathRow label="Micro logits"       eq="L̂_μ = (1/N_μ) Σ_{t=1}^{N_μ} logits_t"       note="N_μ = microBatchSize (default 8)" />
                    <MathRow label="Full logits"        eq="L̂_F = (1/T) Σ_{t=1}^{T} logits_t"            note="T = sequence length" />
                    <MathRow label="Prediction"         eq="ŷ = argmax softmax(L̂ / τ)"                   note="τ = calibrationTemperature (default 1.2)" />
                    <MathRow label="Confidence delta"   eq="Δ = conf_F(ŷ_F) − conf_μ(ŷ_μ)"               note="signed shift — positive = refined upward" />
                    <MathRow label="Attribution score"  eq="a_t = |logits_t[target] − mean(logits_t)|"    note="unnormalized sensitivity at position t" />
                    <MathRow label="Normalization"      eq="ã_t = a_t / max(a_1..T)"                      note="rescale to [0,1]" />
                    <MathRow label="Calibration"        eq="conf_cal = σ( logit(p) / τ )"                 note="Platt scaling — temperature-scaled sigmoid" />
                    <MathRow label="Spurious flag"      eq="flag iff ∃t : ã_t / Σ ã > 0.7"              note="single token dominates attribution weight" />
                  </div>
                </Panel>

                {sepLog.length > 0 && (
                  <Panel>
                    <PanelTitle>SEP Step Log — last {Math.min(sepLog.length, 50)} entries</PanelTitle>
                    <div className="mt-2 max-h-72 overflow-y-auto space-y-0.5">
                      {sepLog.slice(-50).reverse().map((line, i) => (
                        <div key={i} className="text-[9px] text-slate-500 font-mono leading-relaxed py-0.5 border-b border-slate-900/40 last:border-0">
                          {line}
                        </div>
                      ))}
                    </div>
                  </Panel>
                )}
              </>
            )}

          </div>
        </main>
      </div>
    </div>
  );
}

// ── SEP Deep-Dive Panel ────────────────────────────────────────────────────────

function SEPDeepDive({ analysis, step }: { analysis: NonNullable<TrainStep['sepAnalysis']>; step: number }) {
  return (
    <>
      <div className="grid grid-cols-2 gap-4">
        {/* Micro prediction */}
        <Panel>
          <PanelTitle>Micro-Prediction (step {step})</PanelTitle>
          <div className="space-y-2 mt-1">
            <div className="flex justify-between">
              <span className="text-[10px] text-slate-500">Predicted token</span>
              <span className="text-[11px] text-cyan-300 font-semibold">"{analysis.microPrediction.tokenStr}"</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[10px] text-slate-500">Raw confidence</span>
              <span className="text-[11px] text-amber-400 tabular-nums">{(analysis.microPrediction.confidence * 100).toFixed(2)}%</span>
            </div>
            <div className="mt-3 space-y-1">
              <div className="text-[9px] text-slate-600 uppercase tracking-wider mb-1">Top-K</div>
              {analysis.microPrediction.topK.map((t, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-[9px] text-slate-600 w-16 truncate">"{t.str}"</span>
                  <div className="flex-1 bg-slate-800/80 rounded-full h-1.5">
                    <div className="h-full rounded-full bg-cyan-900" style={{ width: `${(t.prob * 100).toFixed(1)}%` }} />
                  </div>
                  <span className="text-[9px] text-slate-500 tabular-nums w-9 text-right">{(t.prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </Panel>

        {/* Full prediction */}
        <Panel>
          <PanelTitle>Full-Context Prediction</PanelTitle>
          <div className="space-y-2 mt-1">
            <div className="flex justify-between">
              <span className="text-[10px] text-slate-500">Predicted token</span>
              <span className="text-[11px] text-emerald-300 font-semibold">"{analysis.fullPrediction.tokenStr}"</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[10px] text-slate-500">Raw confidence</span>
              <span className="text-[11px] text-amber-400 tabular-nums">{(analysis.fullPrediction.confidence * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[10px] text-slate-500">Calibrated (τ-scaled)</span>
              <span className="text-[11px] text-sky-400 tabular-nums">{(analysis.calibratedConfidence * 100).toFixed(2)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[10px] text-slate-500">Confidence Δ</span>
              <span className={`text-[11px] tabular-nums font-semibold ${analysis.delta >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {analysis.delta >= 0 ? '+' : ''}{(analysis.delta * 100).toFixed(2)}%
              </span>
            </div>
            <div className="mt-3 space-y-1">
              <div className="text-[9px] text-slate-600 uppercase tracking-wider mb-1">Top-K</div>
              {analysis.fullPrediction.topK.map((t, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-[9px] text-slate-600 w-16 truncate">"{t.str}"</span>
                  <div className="flex-1 bg-slate-800/80 rounded-full h-1.5">
                    <div className="h-full rounded-full bg-emerald-900" style={{ width: `${(t.prob * 100).toFixed(1)}%` }} />
                  </div>
                  <span className="text-[9px] text-slate-500 tabular-nums w-9 text-right">{(t.prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </Panel>
      </div>

      {/* Token attribution heatmap */}
      {analysis.attributions.length > 0 && (
        <Panel>
          <PanelTitle>Token Attribution Heatmap</PanelTitle>
          <p className="text-[9px] text-slate-600 mb-3">
            Score = |logit[predicted] − mean(logits)| at each token position, normalized to [0,1].
            Direction: positive = supports prediction · negative = opposes · neutral = ambiguous.
          </p>
          <div className="flex flex-wrap gap-1.5 mt-1">
            {analysis.attributions.map((a, i) => {
              const intensity = Math.round(a.score * 9);
              const bg =
                a.direction === 'positive' ? `rgba(34,197,94,${0.1 + a.score * 0.7})`
                : a.direction === 'negative' ? `rgba(239,68,68,${0.1 + a.score * 0.7})`
                : `rgba(100,116,139,${0.1 + a.score * 0.4})`;
              return (
                <div
                  key={i}
                  className="rounded px-2 py-1 text-[10px] font-mono border border-slate-700/30 cursor-default"
                  style={{ background: bg }}
                  title={`"${a.tokenStr}"  score=${a.score.toFixed(3)}  ${a.direction}  intensity=${intensity}`}
                >
                  <span className="text-slate-200">{a.tokenStr}</span>
                  <span className="text-slate-500 text-[8px] ml-1">{(a.score * 100).toFixed(0)}</span>
                </div>
              );
            })}
          </div>
        </Panel>
      )}

      {/* Spurious flags */}
      <Panel>
        <PanelTitle>Spurious Correlation Analysis</PanelTitle>
        {analysis.spuriousFlags.length === 0 ? (
          <div className="flex items-center gap-2 text-[10px] text-emerald-400 mt-1">
            <span className="text-emerald-500">✓</span> No spurious correlations detected in this step
          </div>
        ) : (
          <div className="space-y-2 mt-1">
            {analysis.spuriousFlags.map((flag, i) => (
              <div key={i} className="flex items-start gap-2 bg-amber-950/30 border border-amber-800/40 rounded p-2">
                <span className="text-amber-500 text-[11px] shrink-0">⚠</span>
                <span className="text-[10px] text-amber-300">{flag}</span>
              </div>
            ))}
          </div>
        )}
      </Panel>

      {/* Natural-language explanation */}
      <Panel>
        <PanelTitle>SEP Natural Language Explanation</PanelTitle>
        <pre className="text-[10px] text-slate-400 whitespace-pre-wrap leading-relaxed mt-1 font-mono">
          {analysis.explanation}
        </pre>
      </Panel>
    </>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[9px] font-bold text-slate-600 uppercase tracking-widest mb-2">{title}</div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function Panel({ children }: { children: React.ReactNode }) {
  return (
    <div className="bg-[#0b0f18] border border-slate-800/70 rounded-xl p-4 shadow-sm">
      {children}
    </div>
  );
}

function PanelTitle({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">{children}</div>;
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-center h-28 text-[11px] text-slate-600 text-center px-8">
      {children}
    </div>
  );
}

function Pill({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <span>
      <span className="text-slate-600">{label} </span>
      <span className={`tabular-nums font-semibold ${color}`}>{value}</span>
    </span>
  );
}

function CfgRow({ label, value, onChange, min, max, step, disabled }: {
  label: string; value: number; min: number; max: number; step: number;
  disabled?: boolean; onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-[10px] text-slate-500 flex-1 truncate">{label}</label>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-14 accent-cyan-500 cursor-pointer"
      />
      <span className="text-[10px] text-cyan-300 w-7 text-right tabular-nums">{value}</span>
    </div>
  );
}

function CheckRow({ label, checked, onChange, disabled }: {
  label: string; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="checkbox" checked={checked} disabled={disabled}
        onChange={e => onChange(e.target.checked)}
        className="accent-cyan-500 cursor-pointer"
      />
      <label className="text-[10px] text-slate-500">{label}</label>
    </div>
  );
}

function Btn({ children, onClick, disabled, color }: {
  children: React.ReactNode; onClick: () => void;
  disabled?: boolean; color: 'slate' | 'cyan' | 'red';
}) {
  const cls = {
    slate: 'bg-slate-700/70 hover:bg-slate-700 border-slate-600/40',
    cyan:  'bg-cyan-900/70  hover:bg-cyan-900  border-cyan-700/40',
    red:   'bg-red-900/60   hover:bg-red-900   border-red-700/40',
  }[color];
  return (
    <button
      onClick={onClick} disabled={disabled}
      className={`w-full py-2 rounded border ${cls} text-[10px] font-semibold text-slate-200 disabled:opacity-25 transition`}
    >
      {children}
    </button>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between">
      <span className="text-[10px] text-slate-600">{label}</span>
      <span className="text-[10px] text-slate-300 tabular-nums">{value}</span>
    </div>
  );
}

function MathRow({ label, eq, note }: { label: string; eq: string; note: string }) {
  return (
    <div className="grid grid-cols-[120px_1fr] gap-3 py-1 border-b border-slate-800/40 last:border-0">
      <span className="text-slate-500 text-[9px] uppercase tracking-wider pt-0.5">{label}</span>
      <div>
        <div className="text-cyan-400 font-mono text-[10px]">{eq}</div>
        <div className="text-slate-600 text-[9px] mt-0.5">{note}</div>
      </div>
    </div>
  );
}

// ── Chart Card ─────────────────────────────────────────────────────────────────

function ChartCard({ title, color, data, yLabel }: {
  title: string; color: string; data: number[]; yLabel: string;
}) {
  const last  = data[data.length - 1];
  const min   = Math.min(...data);
  const max   = Math.max(...data, min + 1e-9);
  const range = max - min || 1;
  const H = 52, W = 400;

  const pts = data.map((v, i) => {
    const x = (i / Math.max(data.length - 1, 1)) * W;
    const y = H - ((v - min) / range) * (H - 6) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  // Min/max labels
  const minY = H - 4;
  const maxY = 6;

  return (
    <div className="bg-[#0b0f18] border border-slate-800/70 rounded-xl p-3 shadow-sm">
      <div className="flex justify-between items-center mb-1">
        <span className="text-[10px] text-slate-400 font-semibold">{title}</span>
        <span className="text-[11px] font-bold tabular-nums" style={{ color }}>
          {typeof last === 'number' ? last.toFixed(5) : '—'}
        </span>
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id={`g_${title.replace(/\s/g,'_')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor={color} stopOpacity="0.22" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {/* Gridlines */}
        <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="#1e293b" strokeWidth="0.5" />
        {data.length > 1 && (
          <>
            <polygon
              points={`0,${H} ${pts} ${W},${H}`}
              fill={`url(#g_${title.replace(/\s/g,'_')})`}
            />
            <polyline
              points={pts}
              fill="none"
              stroke={color}
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </>
        )}
        {/* Axis labels */}
        <text x="2" y={maxY} fontSize="7" fill="#475569">{max.toFixed(2)}</text>
        <text x="2" y={minY} fontSize="7" fill="#475569">{min.toFixed(2)}</text>
        <text x={W - 2} y={H - 2} fontSize="7" fill="#334155" textAnchor="end">{yLabel}</text>
      </svg>
      <div className="flex justify-between text-[8px] text-slate-700 mt-0.5">
        <span>step 1</span>
        <span>step {data.length}</span>
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-[#0b0f18] border border-slate-800/70 rounded-xl p-3 shadow-sm">
      <div className="text-[9px] text-slate-600 uppercase tracking-wider">{label}</div>
      <div className={`text-[17px] font-bold mt-1 tabular-nums leading-none ${color}`}>{value}</div>
    </div>
  );
}

// ── Architecture Diagram ───────────────────────────────────────────────────────

function ArchDiagram({ config }: { config: ModelConfig }) {
  const layers = [
    { name: 'Input Tokens',     detail: `BPE encoded — vocab=${config.vocabSize}`,                        color: 'border-slate-600   text-slate-300' },
    { name: 'EmbeddingLayer',   detail: `[${config.vocabSize}×${config.embedDim}]  Glorot/√D init`,        color: 'border-cyan-800    text-cyan-300' },
    ...Array.from({ length: config.numTransformerBlocks }, (_, i) => ({
      name:   `TransformerBlock ${i + 1}`,
      detail: `Pre-norm: SelfAttn(Q/K/V causal) + LayerNorm + FFN(${config.ffDim}) + Residual`,
      color:  'border-blue-700   text-blue-300',
    })),
    { name: 'LSTMCell',         detail: `i/f/g/o gates  ${config.embedDim}→${config.hiddenDim}  forget_bias=1.0`, color: 'border-purple-700  text-purple-300' },
    { name: 'ProjectionHead',   detail: `[${config.hiddenDim}×${config.vocabSize}]  linear — logits over vocab`,   color: 'border-pink-800    text-pink-300' },
    { name: 'CrossEntropyLoss', detail: 'softmax(logits) − log p(target)  — teacher forcing',              color: 'border-red-800     text-red-300' },
    { name: 'NAG + Hessian',    detail: `μ=${config.momentum}  lr=${config.lr}  Hessian=${config.useHessian ? '✓ finite-diff' : '✗'} `, color: 'border-amber-700   text-amber-300' },
    { name: 'ZeRO Optimizer',   detail: `Stage ${config.zeroStage}  ·  ${config.numRanks} ranks  ·  ${config.dtype.toUpperCase()} precision`, color: 'border-emerald-800 text-emerald-300' },
  ];

  return (
    <div className="space-y-1.5">
      {layers.map((l, i) => (
        <div key={i}>
          <div className={`border ${l.color} rounded-lg px-3 py-2 flex justify-between items-center`}>
            <span className="text-[11px] font-bold">{l.name}</span>
            <span className="text-[10px] text-slate-500 text-right max-w-xs truncate">{l.detail}</span>
          </div>
          {i < layers.length - 1 && (
            <div className="flex justify-center text-slate-700 text-[10px] leading-none py-0.5">↓</div>
          )}
        </div>
      ))}
      <div className="text-[9px] text-slate-700 mt-3 pt-2 border-t border-slate-800/60 flex gap-4 flex-wrap">
        <span>Kernel Fusion ✓</span>
        <span>Autograd Tape ✓</span>
        <span>Cosine LR Schedule ✓</span>
        <span>GSAR Routing ✓</span>
        <span>SEP Analysis ✓</span>
        <span>{config.dtype.toUpperCase()} {config.dtype === 'fp16' ? '(10-bit mantissa sim)' : ''}</span>
      </div>
    </div>
  );
}

// ── ZeRO Memory Panel ──────────────────────────────────────────────────────────

function ZeroPanel({ config, model }: { config: ModelConfig; model: WintermindModel }) {
  const diag        = model.diagnostics();
  const totalParams = (diag.paramCount as number) ?? 0;
  const fp32B       = totalParams * 4;
  const gradB       = totalParams * 4;
  const optB        = totalParams * 8;   // velocity buf (4B) + hessian diag (4B)
  const baseline    = fp32B + gradB + optB;
  const factor      = model.optimizer.memoryReductionFactor();
  const perRank     = baseline * factor;
  const saved       = baseline - perRank;

  return (
    <div className="space-y-3 text-[10px]">
      <div className="text-slate-400 text-[10px]">{model.optimizer.zeroStageDescription()}</div>
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: 'Parameters',     bytes: fp32B, color: 'text-cyan-400' },
          { label: 'Gradients',      bytes: gradB, color: 'text-amber-400' },
          { label: 'Opt States',     bytes: optB,  color: 'text-purple-400' },
          { label: 'Total Baseline', bytes: baseline, color: 'text-white' },
          { label: `Per Rank (${config.numRanks}R)`, bytes: perRank, color: 'text-emerald-400' },
          { label: 'Saved / Rank',   bytes: saved, color: 'text-emerald-300' },
        ].map(({ label, bytes, color }) => (
          <div key={label} className="bg-[#0e1420] rounded-lg p-2.5 border border-slate-800/50">
            <div className="text-slate-600 text-[9px]">{label}</div>
            <div className={`font-bold mt-0.5 ${color}`}>
              {bytes >= 1024 * 1024
                ? `${(bytes / 1024 / 1024).toFixed(2)} MB`
                : `${(bytes / 1024).toFixed(2)} KB`}
            </div>
            <div className="text-slate-700 text-[8px]">{totalParams.toLocaleString()} params</div>
          </div>
        ))}
      </div>
      <div className="bg-emerald-950/30 border border-emerald-800/40 rounded-lg p-3 text-emerald-400">
        ZeRO-{config.zeroStage} saves{' '}
        <strong>{((1 - factor) * 100).toFixed(0)}%</strong> memory per rank vs no sharding
        <span className="text-emerald-700 ml-2 text-[9px]">
          ({(baseline / 1024).toFixed(1)} KB → {(perRank / 1024).toFixed(1)} KB per rank)
        </span>
      </div>
      <div className="text-[9px] text-slate-700 leading-relaxed">
        Baseline = 4B (param, fp32) + 4B (grad, fp32) + 4B (velocity) + 4B (hessian diag) = 16B/param
        &emsp;·&emsp; ZeRO-3 shards all 16B across {config.numRanks} ranks → {(16 / config.numRanks).toFixed(1)}B/param/rank
      </div>
    </div>
  );
}
