import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ─── Theme ───
const C = {
  bg: "#0f172a", card: "#1e293b", cardL: "#334155", border: "#475569",
  text: "#e2e8f0", dim: "#94a3b8", white: "#fff",
  pink: "#f472b6", purple: "#a78bfa", blue: "#3b82f6",
  green: "#10b981", amber: "#f59e0b", red: "#ef4444", cyan: "#22d3ee",
  orange: "#fb923c", lime: "#84cc16", rose: "#fb7185", sky: "#38bdf8",
};

// ─── Stage Definitions ───
interface Stage {
  id: string;
  label: string;
  icon: string;
  color: string;
  sub: string;
  duration: number; // base ms at 1x speed
}

const STAGES: Stage[] = [
  { id: "input",    label: "Input",       icon: "📝", color: C.purple, sub: "raw text",              duration: 2000 },
  { id: "tokenize", label: "Tokenize",    icon: "✂️",  color: C.amber,  sub: "BPE → token IDs",       duration: 2500 },
  { id: "embed",    label: "Embed",       icon: "📊", color: C.amber,  sub: "ID → 512-dim vector",   duration: 2500 },
  { id: "block1",   label: "Block 1",     icon: "⚙️",  color: C.pink,   sub: "norm → attn → FFN",     duration: 4000 },
  { id: "blocks",   label: "Blocks 2-12", icon: "🔁", color: C.purple, sub: "deepening abstractions", duration: 3000 },
  { id: "output",   label: "Output",      icon: "🎯", color: C.green,  sub: "logits → softmax → token", duration: 3500 },
];

// ─── Helpers ───
function fakeVec(seed: number, n = 14): number[] {
  const v: number[] = [];
  for (let i = 0; i < n; i++)
    v.push(parseFloat((Math.sin(seed * (i + 1) * 0.7) * 0.5).toFixed(2)));
  return v;
}

// ─── Shared Primitives ───
function Dim({ children }: { children: React.ReactNode }) {
  return <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6 }}>{children}</div>;
}

function VecRow({ values, color, animate }: { values: number[]; color: string; animate?: boolean }) {
  return (
    <div style={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
      {values.map((v, i) => (
        <motion.div
          key={i}
          initial={animate ? { opacity: 0, scale: 0 } : false}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: animate ? i * 0.04 : 0, duration: 0.2 }}
          style={{
            width: 18, height: 18, borderRadius: 2,
            background: v >= 0 ? `rgba(16,185,129,${Math.abs(v)})` : `rgba(239,68,68,${Math.abs(v)})`,
            border: `1px solid ${color}33`,
          }}
        />
      ))}
      <span style={{ fontSize: 10, color: C.dim, marginLeft: 4 }}>...512</span>
    </div>
  );
}

// ─── Data Packet (traveling orb) ───
function DataPacket({ active, color }: { active: boolean; color: string }) {
  return (
    <AnimatePresence>
      {active && (
        <motion.div
          initial={{ opacity: 0, scale: 0.3 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.3, y: 20 }}
          transition={{ duration: 0.3 }}
          style={{
            width: 12, height: 12, borderRadius: "50%",
            background: `radial-gradient(circle, ${color}, ${color}88)`,
            boxShadow: `0 0 16px ${color}88, 0 0 32px ${color}44`,
            margin: "0 auto",
          }}
        />
      )}
    </AnimatePresence>
  );
}

// ─── Connector Line ───
function Connector({ active, color, packetVisible }: { active: boolean; color: string; packetVisible: boolean }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", height: 40, justifyContent: "center", position: "relative" }}>
      <motion.div
        animate={{
          background: active
            ? `linear-gradient(180deg, ${color}88, ${color})`
            : C.border + "44",
        }}
        transition={{ duration: 0.4 }}
        style={{ width: 2, height: 28, borderRadius: 1 }}
      />
      <div style={{ position: "absolute", top: "50%", transform: "translateY(-50%)" }}>
        <DataPacket active={packetVisible} color={color} />
      </div>
    </div>
  );
}

// ─── Stage Card (collapsed) ───
function StageCardCollapsed({ stage, index, isCompleted, isCurrent, onClick }: {
  stage: Stage; index: number; isCompleted: boolean; isCurrent: boolean; onClick: () => void;
}) {
  return (
    <motion.div
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      animate={{
        borderColor: isCurrent ? stage.color : isCompleted ? stage.color + "66" : C.border + "44",
        background: isCurrent ? stage.color + "15" : C.card,
      }}
      transition={{ duration: 0.3 }}
      style={{
        display: "flex", alignItems: "center", gap: 12,
        padding: "12px 16px", borderRadius: 10,
        border: `1.5px solid ${C.border}44`,
        cursor: "pointer", position: "relative", overflow: "hidden",
      }}
    >
      {isCompleted && !isCurrent && (
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100%" }}
          style={{
            position: "absolute", left: 0, top: 0, height: "100%",
            background: `linear-gradient(90deg, ${stage.color}08, transparent)`,
          }}
        />
      )}
      <div style={{ fontSize: 22 }}>{stage.icon}</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: isCurrent ? stage.color : isCompleted ? C.text : C.dim }}>
          {stage.label}
        </div>
        <div style={{ fontSize: 10, color: C.dim }}>{stage.sub}</div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ fontSize: 10, color: C.dim, fontFamily: "monospace" }}>
          {index + 1}/{STAGES.length}
        </div>
        {isCompleted && !isCurrent && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            style={{
              width: 20, height: 20, borderRadius: "50%",
              background: stage.color + "30", display: "flex",
              alignItems: "center", justifyContent: "center",
              fontSize: 11, color: stage.color,
            }}
          >
            ✓
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

// ═══════════════════════════════════════
// ─── Per-Stage Detail Components ───
// ═══════════════════════════════════════

function InputDetail({ progress }: { progress: number }) {
  const text = "def fib";
  const visibleChars = Math.floor(progress * text.length);
  return (
    <div style={{ textAlign: "center", padding: 20 }}>
      <Dim>raw text input to the model:</Dim>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        style={{
          display: "inline-block", background: C.bg, borderRadius: 12,
          padding: "20px 40px", border: `2px solid ${C.purple}`,
          fontFamily: "'JetBrains Mono', monospace", fontSize: 32, fontWeight: 700,
          color: C.purple, letterSpacing: 3, marginTop: 16,
          boxShadow: `0 0 ${8 + progress * 16}px ${C.purple}44`,
        }}
      >
        {text.split("").map((ch, i) => (
          <motion.span
            key={i}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: i < visibleChars ? 1 : 0.15, y: 0 }}
            transition={{ delay: i * 0.08, duration: 0.15 }}
            style={{ color: i < visibleChars ? C.purple : C.dim }}
          >
            {ch}
          </motion.span>
        ))}
      </motion.div>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: progress > 0.5 ? 1 : 0 }}
        style={{ marginTop: 16 }}
      >
        <Dim>just a string of characters. the model has no idea what this means yet.</Dim>
      </motion.div>
    </div>
  );
}

function TokenizeDetail({ progress }: { progress: number }) {
  const tokens = [
    { text: "def", id: 142, color: C.amber },
    { text: " fib", id: 3891, color: C.blue },
  ];
  const showSplit = progress > 0.2;
  const showIds = progress > 0.5;
  return (
    <div>
      <Dim>BPE tokenizer splits text into subword tokens & maps each to an integer ID.</Dim>
      <div style={{ display: "flex", justifyContent: "center", gap: 24, marginTop: 16, flexWrap: "wrap" }}>
        {tokens.map((tok, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: i === 0 ? 30 : -30 }}
            animate={{ opacity: showSplit ? 1 : 0, x: 0 }}
            transition={{ delay: i * 0.15, duration: 0.4, type: "spring" }}
            style={{
              background: C.bg, borderRadius: 10, padding: "14px 24px",
              border: `1.5px solid ${tok.color}44`, textAlign: "center",
            }}
          >
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 24, color: tok.color, fontWeight: 700 }}>
              {tok.text}
            </div>
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: showIds ? 1 : 0, height: showIds ? "auto" : 0 }}
              transition={{ duration: 0.3 }}
            >
              <div style={{ color: C.dim, fontSize: 14, margin: "6px 0" }}>↓</div>
              <AnimatedCounter target={tok.id} active={showIds} color={tok.color} />
            </motion.div>
          </motion.div>
        ))}
      </div>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: progress > 0.8 ? 1 : 0 }}
        style={{
          marginTop: 16, background: C.cardL, borderRadius: 8,
          padding: 10, fontSize: 11, color: C.dim, textAlign: "center",
        }}
      >
        vocab size: 8,192 tokens. each ID maps to a row in the embedding matrix.
      </motion.div>
    </div>
  );
}

function AnimatedCounter({ target, active, color }: { target: number; active: boolean; color: string }) {
  const [value, setValue] = useState(0);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    if (!active) { setValue(0); return; }
    const start = performance.now();
    const dur = 600;
    function tick(now: number) {
      const t = Math.min((now - start) / dur, 1);
      setValue(Math.floor(t * target));
      if (t < 1) frameRef.current = requestAnimationFrame(tick);
    }
    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [active, target]);

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', monospace", fontSize: 22, fontWeight: 700,
      color, background: color + "15", borderRadius: 6, padding: "4px 14px",
    }}>
      {value}
    </div>
  );
}

function EmbedDetail({ progress }: { progress: number }) {
  const tokens = [
    { id: 142, label: "def", color: C.amber, seed: 142 },
    { id: 3891, label: "fib", color: C.blue, seed: 3891 },
  ];
  const showVecs = progress > 0.3;
  return (
    <div>
      <Dim>each token ID indexes a learned embedding table (8192 x 512). one dense vector per token.</Dim>
      <div style={{ display: "flex", gap: 12, marginTop: 14, flexWrap: "wrap" }}>
        {tokens.map((tok, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.2, duration: 0.4 }}
            style={{
              flex: "1 1 200px", background: C.bg, borderRadius: 8, padding: 14,
              border: `1px solid ${tok.color}33`,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <span style={{ fontFamily: "monospace", color: tok.color, fontWeight: 700, fontSize: 14 }}>{tok.id}</span>
              <span style={{ color: C.dim, fontSize: 11 }}>→</span>
              <span style={{ color: tok.color, fontSize: 12, fontWeight: 600 }}>"{tok.label}" embedding</span>
            </div>
            {showVecs && <VecRow values={fakeVec(tok.seed)} color={tok.color} animate />}
          </motion.div>
        ))}
      </div>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: progress > 0.7 ? 1 : 0 }}
        style={{
          marginTop: 12, background: C.green + "15", border: `1px solid ${C.green}33`,
          borderRadius: 8, padding: 10, fontSize: 11, color: C.dim,
        }}
      >
        <strong style={{ color: C.green }}>params:</strong> 8,192 x 512 = 4,194,304 embedding params. tied with output projection.
      </motion.div>
    </div>
  );
}

function Block1Detail({ progress }: { progress: number }) {
  const subSteps = [
    { label: "RMSNorm", color: C.cyan, desc: "normalize vectors to unit scale. prevents activations from exploding." },
    { label: "Attention + RoPE", color: C.pink, desc: '"fib" attends to "def" (score 0.80) and itself (0.20). RoPE encodes relative position.' },
    { label: "+ Residual", color: C.green, desc: "add pre-attention vector back. skip connection preserves original signal." },
    { label: "RMSNorm", color: C.cyan, desc: "normalize again before feedforward. same operation — stabilize & rescale." },
    { label: "SwiGLU FFN", color: C.amber, desc: "gate = swish(x·W_gate), up = x·W_up, out = (gate ⊙ up)·W_down. 512→1376→512." },
    { label: "+ Residual", color: C.green, desc: "add pre-FFN vector back. 2 residual connections per block = 2 gradient highways." },
  ];
  const activeSubStep = Math.min(Math.floor(progress * subSteps.length), subSteps.length - 1);

  return (
    <div>
      <Dim>block 1 of 12. each block: RMSNorm → Attention → +Residual → RMSNorm → SwiGLU → +Residual</Dim>
      <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 12, maxWidth: 400, margin: "12px auto 0" }}>
        {subSteps.map((step, i) => {
          const isActive = i === activeSubStep;
          const isDone = i < activeSubStep;
          return (
            <motion.div
              key={i}
              animate={{
                borderColor: isActive ? step.color : isDone ? step.color + "44" : C.border + "22",
                background: isActive ? step.color + "18" : C.bg,
              }}
              transition={{ duration: 0.3 }}
              style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "8px 12px", borderRadius: 6,
                border: `1.5px solid ${C.border}22`,
              }}
            >
              <motion.div
                animate={{
                  background: isActive || isDone ? step.color : C.cardL,
                  scale: isActive ? 1.2 : 1,
                }}
                style={{ width: 6, height: 24, borderRadius: 3, flexShrink: 0 }}
              />
              <div style={{ flex: 1 }}>
                <div style={{
                  fontSize: 12, fontWeight: 600, fontFamily: "monospace",
                  color: isActive ? step.color : isDone ? C.text : C.dim,
                }}>
                  {step.label}
                </div>
                <AnimatePresence>
                  {isActive && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      style={{ fontSize: 10, color: C.dim, marginTop: 2, lineHeight: 1.5 }}
                    >
                      {step.desc}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              {isDone && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  style={{ fontSize: 10, color: step.color }}
                >
                  ✓
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>
      {/* Attention heatmap when on step 1 */}
      <AnimatePresence>
        {activeSubStep === 1 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            style={{ marginTop: 12 }}
          >
            <div style={{ display: "flex", justifyContent: "center", gap: 24 }}>
              {[{ t: "def", s: 0.8, c: C.amber }, { t: "fib", s: 0.2, c: C.blue }].map((tok, i) => (
                <div key={i} style={{ textAlign: "center" }}>
                  <div style={{ fontFamily: "monospace", color: tok.c, fontWeight: 700, fontSize: 14 }}>{tok.t}</div>
                  <div style={{
                    width: 60, height: 6, background: C.bg, borderRadius: 3,
                    marginTop: 4, overflow: "hidden",
                  }}>
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${tok.s * 100}%` }}
                      transition={{ duration: 0.6, delay: 0.2 }}
                      style={{ height: "100%", background: C.pink, borderRadius: 3 }}
                    />
                  </div>
                  <div style={{ fontSize: 11, color: C.pink, fontWeight: 600, marginTop: 2 }}>{tok.s}</div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function BlocksDetail({ progress }: { progress: number }) {
  const layers = [
    { range: "1-4", label: "Syntax", desc: "brackets, indentation, keyword patterns", color: C.cyan },
    { range: "5-8", label: "Semantics", desc: "variable types, function signatures, scope", color: C.purple },
    { range: "9-12", label: "High-Level", desc: "algorithm structure, code intent, naming", color: C.pink },
  ];
  const layerProgress = progress * 12;

  return (
    <div>
      <Dim>same block repeats 12x. each layer builds more abstract representations.</Dim>
      <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 14 }}>
        {layers.map((layer, i) => {
          const start = i * 4;
          const end = start + 4;
          const pct = Math.max(0, Math.min(1, (layerProgress - start) / (end - start)));
          const isActive = layerProgress >= start && layerProgress < end;

          return (
            <motion.div
              key={i}
              animate={{
                borderColor: isActive ? layer.color : pct >= 1 ? layer.color + "44" : C.border + "22",
                background: isActive ? layer.color + "10" : C.bg,
              }}
              style={{
                padding: "12px 14px", borderRadius: 8,
                border: `1.5px solid ${C.border}22`,
                borderLeft: `3px solid ${pct > 0 ? layer.color : C.cardL}`,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <span style={{ color: layer.color, fontWeight: 700, fontSize: 13 }}>
                  Blocks {layer.range}: {layer.label}
                </span>
                <span style={{ fontSize: 11, color: C.dim, fontFamily: "monospace" }}>
                  {Math.round(pct * 100)}%
                </span>
              </div>
              <div style={{ fontSize: 11, color: C.dim, marginBottom: 6 }}>{layer.desc}</div>
              <div style={{ height: 6, background: C.cardL, borderRadius: 3, overflow: "hidden" }}>
                <motion.div
                  animate={{ width: `${pct * 100}%` }}
                  transition={{ duration: 0.2 }}
                  style={{
                    height: "100%", borderRadius: 3,
                    background: `linear-gradient(90deg, ${layer.color}66, ${layer.color})`,
                  }}
                />
              </div>
            </motion.div>
          );
        })}
      </div>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: progress > 0.85 ? 1 : 0 }}
        style={{
          marginTop: 12, background: C.cardL, borderRadius: 8,
          padding: 10, fontSize: 11, color: C.dim, textAlign: "center",
        }}
      >
        24 residual connections total (2 per block x 12 blocks)
      </motion.div>
    </div>
  );
}

function OutputDetail({ progress }: { progress: number }) {
  const preds = [
    { token: "onacci", prob: 0.67, color: C.green },
    { token: "_", prob: 0.12, color: C.amber },
    { token: "(", prob: 0.08, color: C.dim },
    { token: "o", prob: 0.04, color: C.dim },
    { token: "er", prob: 0.03, color: C.dim },
  ];
  const showVec = progress > 0.1;
  const showBars = progress > 0.3;
  const showResult = progress > 0.75;

  return (
    <div>
      <Dim>final hidden state → unembedding → logits → softmax → probabilities over 8,192 tokens.</Dim>

      {/* Hidden state vector */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: showVec ? 1 : 0 }}
        style={{ background: C.bg, borderRadius: 8, padding: 12, marginTop: 12 }}
      >
        <div style={{ fontSize: 11, color: C.dim, marginBottom: 6 }}>final hidden state @ position 2:</div>
        <VecRow values={fakeVec(999)} color={C.purple} animate={showVec} />
      </motion.div>

      {/* Formula */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: showBars ? 1 : 0 }}
        style={{ textAlign: "center", fontSize: 12, color: C.purple, fontFamily: "monospace", margin: "10px 0" }}
      >
        hidden x W_embed<sup>T</sup> → logits [8192] → softmax
      </motion.div>

      {/* Probability bars */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: showBars ? 1 : 0 }}
        style={{ background: C.bg, borderRadius: 8, padding: 14 }}
      >
        <div style={{ fontSize: 11, color: C.dim, marginBottom: 10 }}>top predictions for next token after "fib":</div>
        {preds.map((p, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <div style={{ fontSize: 12, color: C.dim, width: 60, textAlign: "right", fontFamily: "monospace" }}>
              "{p.token}"
            </div>
            <div style={{ flex: 1, height: 20, background: C.cardL, borderRadius: 4, overflow: "hidden" }}>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: showBars ? `${(p.prob / 0.7) * 100}%` : 0 }}
                transition={{ delay: i * 0.1 + 0.2, duration: 0.6, ease: "easeOut" }}
                style={{
                  height: "100%", borderRadius: 4,
                  background: `linear-gradient(90deg, ${p.color}88, ${p.color})`,
                }}
              />
            </div>
            <div style={{ fontSize: 11, color: C.dim, width: 36, fontFamily: "monospace" }}>
              {p.prob.toFixed(2)}
            </div>
          </div>
        ))}
      </motion.div>

      {/* Final result */}
      <AnimatePresence>
        {showResult && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ type: "spring", damping: 12 }}
            style={{
              marginTop: 16, textAlign: "center", padding: 20,
              background: C.green + "15", border: `2px solid ${C.green}`,
              borderRadius: 12,
              boxShadow: `0 0 30px ${C.green}22`,
            }}
          >
            <div style={{ fontSize: 11, color: C.dim, marginBottom: 4 }}>argmax → predicted next token:</div>
            <motion.div
              initial={{ scale: 0.5 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", damping: 8, delay: 0.2 }}
              style={{
                fontFamily: "'JetBrains Mono', monospace", fontSize: 32, fontWeight: 700,
                color: C.green,
              }}
            >
              "onacci"
            </motion.div>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              style={{
                marginTop: 10, fontSize: 16, fontWeight: 700, color: C.purple,
                fontFamily: "'JetBrains Mono', monospace",
              }}
            >
              def fib → def fibonacci
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Stage Detail Renderer ───
const DETAIL_RENDERERS = [InputDetail, TokenizeDetail, EmbedDetail, Block1Detail, BlocksDetail, OutputDetail];

// ═══════════════════════════════════════
// ─── Main App ───
// ═══════════════════════════════════════

export default function App() {
  const [activeStage, setActiveStage] = useState(-1); // -1 = idle
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [stageProgress, setStageProgress] = useState(0); // 0..1 within current stage
  const progressRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-play logic
  const advanceStage = useCallback(() => {
    setActiveStage(prev => {
      if (prev >= STAGES.length - 1) {
        setPlaying(false);
        return prev;
      }
      setStageProgress(0);
      return prev + 1;
    });
  }, []);

  // Progress ticker — drives stageProgress from 0→1 during auto-play
  useEffect(() => {
    if (!playing || activeStage < 0 || activeStage >= STAGES.length) return;
    const duration = STAGES[activeStage].duration / speed;
    const interval = 50; // tick every 50ms
    const increment = interval / duration;

    progressRef.current = 0;
    setStageProgress(0);

    timerRef.current = setInterval(() => {
      progressRef.current += increment;
      if (progressRef.current >= 1) {
        progressRef.current = 1;
        setStageProgress(1);
        if (timerRef.current) clearInterval(timerRef.current);
        // Brief pause then advance
        setTimeout(() => {
          if (activeStage < STAGES.length - 1) {
            advanceStage();
          } else {
            setPlaying(false);
          }
        }, 400 / speed);
      } else {
        setStageProgress(progressRef.current);
      }
    }, interval);

    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, activeStage, speed, advanceStage]);

  function handlePlay() {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (activeStage >= STAGES.length - 1) {
      // Reset if at end
      setActiveStage(-1);
      setStageProgress(0);
      setTimeout(() => {
        setActiveStage(0);
        setStageProgress(0);
        setPlaying(true);
      }, 200);
    } else {
      if (activeStage < 0) setActiveStage(0);
      setStageProgress(0);
      setPlaying(true);
    }
  }

  function handleReset() {
    setPlaying(false);
    setActiveStage(-1);
    setStageProgress(0);
  }

  function handleStageClick(i: number) {
    setPlaying(false);
    setActiveStage(i);
    setStageProgress(1);
  }

  const speeds = [0.5, 1, 2];

  const ActiveDetail = activeStage >= 0 && activeStage < STAGES.length ? DETAIL_RENDERERS[activeStage] : null;

  return (
    <div style={{ maxWidth: 620, margin: "0 auto", padding: "20px 16px 60px" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <h1 style={{ fontSize: 20, fontWeight: 800, color: C.white }}>Forward Pass Simulator</h1>
        <p style={{ fontSize: 12, color: C.dim, marginTop: 2 }}>
          tracing <span style={{ color: C.purple, fontFamily: "monospace" }}>"def fib"</span> through a 50M param transformer
        </p>
      </div>

      {/* Controls */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "center",
        gap: 10, marginBottom: 20, flexWrap: "wrap",
      }}>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handlePlay}
          style={{
            background: playing ? C.cardL : C.pink,
            color: C.white, border: "none", borderRadius: 10,
            padding: "10px 24px", fontSize: 14, fontWeight: 700,
            cursor: "pointer", display: "flex", alignItems: "center", gap: 6,
          }}
        >
          {playing ? "⏸ Pause" : activeStage >= STAGES.length - 1 ? "↻ Replay" : "▶ Play"}
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleReset}
          style={{
            background: C.cardL, color: C.dim, border: `1px solid ${C.border}`,
            borderRadius: 10, padding: "10px 16px", fontSize: 13,
            fontWeight: 600, cursor: "pointer",
          }}
        >
          Reset
        </motion.button>
        <div style={{ display: "flex", gap: 2, background: C.cardL, borderRadius: 8, padding: 2 }}>
          {speeds.map(s => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              style={{
                background: speed === s ? C.card : "transparent",
                color: speed === s ? C.white : C.dim,
                border: "none", borderRadius: 6, padding: "6px 10px",
                fontSize: 11, fontWeight: 600, cursor: "pointer",
              }}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* Overall progress bar */}
      <div style={{ display: "flex", gap: 3, marginBottom: 16 }}>
        {STAGES.map((stage, i) => (
          <motion.div
            key={i}
            animate={{
              background: i < activeStage ? stage.color
                : i === activeStage ? `linear-gradient(90deg, ${stage.color} ${stageProgress * 100}%, ${C.cardL} ${stageProgress * 100}%)`
                : C.cardL,
            }}
            style={{ flex: 1, height: 4, borderRadius: 2 }}
          />
        ))}
      </div>

      {/* Pipeline */}
      <div style={{ display: "flex", flexDirection: "column" }}>
        {STAGES.map((stage, i) => {
          const isCurrent = i === activeStage;
          const isCompleted = i < activeStage;
          const showPacket = playing && i === activeStage && stageProgress < 0.1;

          return (
            <div key={stage.id}>
              {i > 0 && (
                <Connector
                  active={isCompleted || isCurrent}
                  color={stage.color}
                  packetVisible={showPacket}
                />
              )}
              <StageCardCollapsed
                stage={stage}
                index={i}
                isCompleted={isCompleted}
                isCurrent={isCurrent}
                onClick={() => handleStageClick(i)}
              />
              {/* Expanded detail panel */}
              <AnimatePresence>
                {isCurrent && ActiveDetail && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.35, ease: "easeInOut" }}
                    style={{ overflow: "hidden" }}
                  >
                    <div style={{
                      background: C.card, borderRadius: "0 0 10px 10px",
                      padding: "16px 18px", borderLeft: `3px solid ${stage.color}`,
                      borderBottom: `1px solid ${stage.color}22`,
                      marginTop: -1,
                    }}>
                      <ActiveDetail progress={stageProgress} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      {/* Completion message */}
      <AnimatePresence>
        {activeStage >= STAGES.length - 1 && stageProgress >= 0.9 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ delay: 0.5 }}
            style={{
              marginTop: 20, textAlign: "center", padding: 16,
              background: C.card, borderRadius: 12,
              border: `1px solid ${C.green}33`,
            }}
          >
            <div style={{ fontSize: 13, color: C.dim }}>
              That's one forward pass! Each of the <strong style={{ color: C.white }}>42 million parameters</strong> contributed to predicting <strong style={{ color: C.green }}>"onacci"</strong> as the next token.
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
