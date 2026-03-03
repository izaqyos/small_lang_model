import { useState, useEffect, useRef } from "react";

const C = {
	bg: "#0f172a", card: "#1e293b", cardL: "#334155", border: "#475569",
	text: "#e2e8f0", dim: "#94a3b8", white: "#fff",
	pink: "#f472b6", purple: "#a78bfa", blue: "#3b82f6",
	green: "#10b981", amber: "#f59e0b", red: "#ef4444", cyan: "#22d3ee",
	orange: "#fb923c", lime: "#84cc16", rose: "#fb7185", sky: "#38bdf8",
};

// ─── Shared Components ───
function Box({ children, color, style }) {
	return <div style={{ background: C.card, borderRadius: 10, padding: "16px 18px", borderLeft: `3px solid ${color || C.border}`, marginBottom: 12, ...style }}>{children}</div>;
}
function Tag({ children, color }) {
	return <span style={{ background: (color || C.purple) + "20", color: color || C.purple, padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 600, fontFamily: "monospace" }}>{children}</span>;
}
function Label({ children, color }) {
	return <div style={{ fontSize: 13, fontWeight: 700, color: color || C.text, marginBottom: 6 }}>{children}</div>;
}
function Dim({ children }) {
	return <div style={{ fontSize: 12, color: C.dim, lineHeight: 1.6 }}>{children}</div>;
}
function Mono({ children, color }) {
	return <span style={{ fontFamily: "monospace", color: color || C.amber, fontSize: 12 }}>{children}</span>;
}
function Arrow() {
	return <div style={{ textAlign: "center", color: C.dim, fontSize: 16, margin: "4px 0" }}>↓</div>;
}
function FlowStep({ icon, title, sub, color, active }) {
	return (
		<div style={{ background: active ? color + "20" : C.bg, border: `1px solid ${active ? color : C.border}`, borderRadius: 8, padding: "10px 14px", textAlign: "center", flex: "1 1 100px", transition: "all 0.3s" }}>
			<div style={{ fontSize: 20, marginBottom: 2 }}>{icon}</div>
			<div style={{ fontSize: 11, fontWeight: 700, color: active ? color : C.text }}>{title}</div>
			{sub && <div style={{ fontSize: 9, color: C.dim, marginTop: 2 }}>{sub}</div>}
		</div>
	);
}
function BarChart({ data, maxVal }) {
	const mx = maxVal || Math.max(...data.map(d => d.value));
	return (
		<div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
			{data.map((d, i) => (
				<div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
					<div style={{ fontSize: 11, color: C.dim, width: 80, textAlign: "right", fontFamily: "monospace" }}>{d.label}</div>
					<div style={{ flex: 1, height: 18, background: C.bg, borderRadius: 4, overflow: "hidden" }}>
						<div style={{ width: `${(d.value / mx) * 100}%`, height: "100%", background: `linear-gradient(90deg, ${(d.color || C.pink)}88, ${d.color || C.pink})`, borderRadius: 4, transition: "width 0.6s ease-out" }} />
					</div>
					<div style={{ fontSize: 10, color: C.dim, width: 60, fontFamily: "monospace" }}>{d.display || d.value}</div>
				</div>
			))}
		</div>
	);
}
function VectorRow({ values, color }) {
	return (
		<div style={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
			{values.map((v, i) => (
				<div key={i} style={{ width: 16, height: 16, borderRadius: 2, background: v >= 0 ? `rgba(16,185,129,${Math.abs(v)})` : `rgba(239,68,68,${Math.abs(v)})`, border: `1px solid ${(color || C.green) + "33"}` }} />
			))}
			<span style={{ fontSize: 10, color: C.dim, marginLeft: 4 }}>...512</span>
		</div>
	);
}
function Tabs({ tabs, active, onChange, color }) {
	return (
		<div style={{ display: "flex", gap: 4, marginBottom: 14, flexWrap: "wrap" }}>
			{tabs.map((t, i) => (
				<button key={i} onClick={() => onChange(i)} style={{ background: active === i ? (color || C.pink) : C.cardL, color: active === i ? C.white : C.dim, border: "none", borderRadius: 8, padding: "7px 14px", fontSize: 12, cursor: "pointer", fontWeight: 600, transition: "all 0.2s" }}>{t}</button>
			))}
		</div>
	);
}

// ─── SECTIONS ───

function BigPicture() {
	const [activeStage, setActiveStage] = useState(0);
	const stages = [
		{ icon: "🧪", title: "Stage 1: Learn", sub: "Train 50M from scratch", color: C.cyan, desc: "Build a tiny Llama model from zero to learn the full pipeline: data prep, tokenizer training, model architecture, training loop, evaluation.", details: ["457K Python functions from code_search_net", "Custom BPE tokenizer (8K vocab)", "12-layer Llama architecture (42M params)", "Train on MacBook w/ MLX framework"] },
		{ icon: "🎓", title: "Stage 2: Distill", sub: "Fine-tune Qwen 0.5B/3B", color: C.purple, desc: "Take a pre-trained model (already trained on 5.5T tokens) and specialize it for Python using knowledge distillation from Claude.", details: ["5K+ synthetic instruction/response pairs", "Claude as teacher model (generates data)", "LoRA fine-tuning (only 0.29% of params)", "3.9 hours on MacBook Pro"] },
		{ icon: "🚀", title: "Stage 3: Ship", sub: "Publish to Ollama", color: C.green, desc: "Convert, quantize, and package the model so anyone can run it with a single command.", details: ["Fuse LoRA adapters into base model", "Convert to GGUF format (llama.cpp)", "Quantize Q4_K_M (4x compression, ~95% quality)", "ollama run yosii/python-expert"] },
	];
	return (
		<div>
			<Dim>we build a Python code expert in 3 stages. click each 2 explore:</Dim>
			<div style={{ display: "flex", gap: 8, margin: "14px 0", flexWrap: "wrap" }}>
				{stages.map((s, i) => (
					<div key={i} onClick={() => setActiveStage(i)} style={{ cursor: "pointer", flex: "1 1 150px" }}>
						<FlowStep icon={s.icon} title={s.title} sub={s.sub} color={s.color} active={activeStage === i} />
					</div>
				))}
			</div>
			<Box color={stages[activeStage].color}>
				<Label color={stages[activeStage].color}>{stages[activeStage].title}</Label>
				<Dim>{stages[activeStage].desc}</Dim>
				<div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6 }}>
					{stages[activeStage].details.map((d, i) => (
						<Tag key={i} color={stages[activeStage].color}>{d}</Tag>
					))}
				</div>
			</Box>
			<Box color={C.amber}>
				<Label color={C.amber}>Why two stages?</Label>
				<Dim>Training a useful 500M model from scratch requires trillions of tokens and massive compute. Instead: learn the pipeline w/ a tiny model, then specialize a pre-trained one w/ a few thousand high-quality examples. This is knowledge distillation.</Dim>
			</Box>
		</div>
	);
}

function LMConcepts() {
	const [tab, setTab] = useState(0);
	return (
		<div>
			<Tabs tabs={["Language Model", "Loss & Perplexity", "LoRA"]} active={tab} onChange={setTab} color={C.cyan} />
			{tab === 0 && (
				<div>
					<Box color={C.cyan}>
						<Label color={C.cyan}>What is a Language Model?</Label>
						<Dim>A language model predicts the next token given previous tokens. That's it. Everything else — code generation, conversations, reasoning — emerges from this simple objective.</Dim>
						<div style={{ background: C.bg, borderRadius: 8, padding: 12, marginTop: 10, fontFamily: "monospace", fontSize: 12, color: C.text, lineHeight: 1.8 }}>
							<span style={{ color: C.dim }}>Input: </span><span style={{ color: C.amber }}>{'"def fibonacci(n):\\n    if n <= 1:\\n        return"'}</span><br />
							<span style={{ color: C.dim }}>Model: </span><span style={{ color: C.pink }}>P(next_token | previous_tokens)</span><br />
							<span style={{ color: C.dim }}>Output: </span><span style={{ color: C.green }}>{'" n"'}</span>
						</div>
					</Box>
				</div>
			)}
			{tab === 1 && (
				<div>
					<Box color={C.red}>
						<Label color={C.red}>Cross-Entropy Loss</Label>
						<Dim>loss = -log(P(correct_token)). If the model assigns P=0.9, loss=0.105 (good). If P=0.01, loss=4.605 (bad).</Dim>
						<div style={{ marginTop: 10 }}>
							<BarChart data={[
								{ label: "P=0.9", value: 0.105, display: "0.105", color: C.green },
								{ label: "P=0.5", value: 0.693, display: "0.693", color: C.amber },
								{ label: "P=0.1", value: 2.303, display: "2.303", color: C.orange },
								{ label: "P=0.01", value: 4.605, display: "4.605", color: C.red },
							]} maxVal={5} />
						</div>
					</Box>
					<Box color={C.amber}>
						<Label color={C.amber}>Perplexity = e^loss</Label>
						<Dim>Represents how many tokens the model is confused between. Our 50M model: started at ~13,000 (random) and dropped to ~80-120 after 4,000 steps.</Dim>
						<div style={{ marginTop: 10 }}>
							<BarChart data={[
								{ label: "Step 0", value: 13661, display: "13,661", color: C.red },
								{ label: "Step 100", value: 3007, display: "3,007", color: C.orange },
								{ label: "Step 500", value: 665, display: "~665", color: C.amber },
								{ label: "Step 2K", value: 181, display: "~181", color: C.lime },
								{ label: "Step 4K", value: 90, display: "~90", color: C.green },
							]} />
						</div>
					</Box>
				</div>
			)}
			{tab === 2 && (
				<div>
					<Box color={C.purple}>
						<Label color={C.purple}>LoRA: Low-Rank Adaptation</Label>
						<Dim>Freeze the original 494M params. Add small trainable matrices (A: down-project, B: up-project). Only train 1.4M params (0.29%).</Dim>
						<div style={{ background: C.bg, borderRadius: 8, padding: 14, marginTop: 10, textAlign: "center" }}>
							<svg viewBox="0 0 360 120" style={{ width: "100%", maxWidth: 400 }}>
								<rect x="10" y="40" width="80" height="40" rx="6" fill={C.cardL} stroke={C.blue} strokeWidth="1.5" />
								<text x="50" y="64" textAnchor="middle" fill={C.blue} fontSize="11" fontWeight="600">W (frozen)</text>
								<text x="50" y="76" textAnchor="middle" fill={C.dim} fontSize="8">512×512</text>
								<line x1="100" y1="60" x2="140" y2="60" stroke={C.dim} strokeWidth="1.5" markerEnd="url(#arr)" />
								<rect x="150" y="10" width="60" height="30" rx="6" fill={C.purple + "30"} stroke={C.purple} strokeWidth="1.5" />
								<text x="180" y="29" textAnchor="middle" fill={C.purple} fontSize="10" fontWeight="600">A (down)</text>
								<text x="180" y="8" textAnchor="middle" fill={C.dim} fontSize="8">32×512</text>
								<rect x="150" y="80" width="60" height="30" rx="6" fill={C.purple + "30"} stroke={C.purple} strokeWidth="1.5" />
								<text x="180" y="99" textAnchor="middle" fill={C.purple} fontSize="10" fontWeight="600">B (up)</text>
								<text x="180" y="77" textAnchor="middle" fill={C.dim} fontSize="8">512×32</text>
								<line x1="180" y1="40" x2="180" y2="80" stroke={C.purple} strokeWidth="1.5" />
								<text x="190" y="65" fill={C.dim} fontSize="9">32-dim</text>
								<circle cx="260" cy="60" r="14" fill={C.green + "30"} stroke={C.green} strokeWidth="1.5" />
								<text x="260" y="64" textAnchor="middle" fill={C.green} fontSize="14" fontWeight="700">+</text>
								<line x1="210" y1="95" x2="246" y2="66" stroke={C.purple} strokeWidth="1.5" />
								<line x1="140" y1="55" x2="150" y2="25" stroke={C.dim} strokeWidth="1" strokeDasharray="3,3" />
								<line x1="100" y1="55" x2="246" y2="55" stroke={C.blue} strokeWidth="1.5" />
								<rect x="290" y="45" width="60" height="30" rx="6" fill={C.green + "20"} stroke={C.green} strokeWidth="1.5" />
								<text x="320" y="64" textAnchor="middle" fill={C.green} fontSize="10" fontWeight="600">output</text>
								<line x1="274" y1="60" x2="290" y2="60" stroke={C.green} strokeWidth="1.5" />
							</svg>
						</div>
						<div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
							{[{ l: "Full params", v: "494M", c: C.blue }, { l: "LoRA params", v: "1.4M", c: C.purple }, { l: "Ratio", v: "0.29%", c: C.green }].map((d, i) => (
								<div key={i} style={{ background: d.c + "15", border: `1px solid ${d.c}33`, borderRadius: 6, padding: "6px 14px", textAlign: "center" }}>
									<div style={{ fontSize: 10, color: C.dim }}>{d.l}</div>
									<div style={{ fontSize: 16, fontWeight: 700, color: d.c }}>{d.v}</div>
								</div>
							))}
						</div>
					</Box>
				</div>
			)}
		</div>
	);
}

function Architecture() {
	const [layer, setLayer] = useState(0);
	const layers = ["Embedding", "RMSNorm", "Attention + RoPE", "SwiGLU FFN", "Residuals", "Output", "Full Pass"];
	const layerColors = [C.amber, C.cyan, C.pink, C.amber, C.green, C.purple, C.rose];

	return (
		<div>
			<Tabs tabs={layers} active={layer} onChange={setLayer} color={layerColors[layer]} />

			{layer === 0 && (
				<Box color={C.amber}>
					<Label color={C.amber}>Embedding Layer — Lookup Table</Label>
					<Dim>Each of 8,192 tokens maps to a learned 512-dim vector. Similar tokens end up near each other in this space.</Dim>
					<div style={{ background: C.bg, borderRadius: 8, padding: 14, marginTop: 10 }}>
						<div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
							{[{ id: "142", tok: "def", c: C.amber }, { id: "3891", tok: "fib", c: C.blue }, { id: "52", tok: "n", c: C.green }].map((t, i) => (
								<div key={i} style={{ textAlign: "center" }}>
									<div style={{ fontFamily: "monospace", fontSize: 16, color: t.c, fontWeight: 700 }}>{t.tok}</div>
									<div style={{ fontSize: 10, color: C.dim }}>ID: {t.id}</div>
									<div style={{ margin: "6px 0" }}>↓</div>
									<VectorRow values={[0.3, -0.5, 0.2, 0.8, -0.1, 0.6, -0.3, 0.4]} color={t.c} />
								</div>
							))}
						</div>
					</div>
					<div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
						<Tag color={C.amber}>8,192 x 512 = 4.2M params</Tag>
						<Tag color={C.amber}>Tied w/ output projection</Tag>
					</div>
				</Box>
			)}

			{layer === 1 && (
				<Box color={C.cyan}>
					<Label color={C.cyan}>RMSNorm — Stabilize Activations</Label>
					<Dim>Divide each element by the root mean square of the vector, then scale by learned gamma. Skips mean subtraction (unlike LayerNorm) for ~10-15% speedup.</Dim>
					<div style={{ background: C.bg, borderRadius: 8, padding: 14, marginTop: 10 }}>
						<div style={{ textAlign: "center", fontFamily: "monospace", fontSize: 13, color: C.text, lineHeight: 2 }}>
							RMSNorm(x)<sub>i</sub> = (x<sub>i</sub> / RMS(x)) * <span style={{ color: C.cyan }}>gamma</span><sub>i</sub>
						</div>
						<div style={{ marginTop: 10, display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
							<div style={{ textAlign: "center" }}>
								<div style={{ fontSize: 10, color: C.dim, marginBottom: 4 }}>LayerNorm</div>
								<div style={{ fontSize: 11, fontFamily: "monospace", color: C.dim }}>(x - mean) / std * gamma + beta</div>
								<div style={{ fontSize: 10, color: C.red, marginTop: 2 }}>4 ops, 2 learned params</div>
							</div>
							<div style={{ textAlign: "center" }}>
								<div style={{ fontSize: 10, color: C.cyan, marginBottom: 4 }}>RMSNorm</div>
								<div style={{ fontSize: 11, fontFamily: "monospace", color: C.cyan }}>x / RMS(x) * gamma</div>
								<div style={{ fontSize: 10, color: C.green, marginTop: 2 }}>2 ops, 1 learned param</div>
							</div>
						</div>
					</div>
					<Tag color={C.cyan}>25 norms x 512 = 12,800 params total</Tag>
				</Box>
			)}

			{layer === 2 && <AttentionSection />}
			{layer === 3 && <SwiGLUSection />}
			{layer === 4 && <ResidualSection />}
			{layer === 5 && <OutputSection />}
			{layer === 6 && <ForwardPass />}
		</div>
	);
}

function AttentionSection() {
	const [sub, setSub] = useState(0);
	return (
		<div>
			<Tabs tabs={["Q / K / V", "Scores & Mask", "Multi-Head", "RoPE"]} active={sub} onChange={setSub} color={C.pink} />
			{sub === 0 && (
				<Box color={C.pink}>
					<Label color={C.pink}>Step 1: Project into Q, K, V</Label>
					<Dim>Each token gets 3 vectors via learned matrix multiplies. Q = what am I looking for? K = what do I contain? V = what info do I carry?</Dim>
					<div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap", justifyContent: "center" }}>
						{[{ l: "Query", s: "Q = x @ W_Q", c: C.amber }, { l: "Key", s: "K = x @ W_K", c: C.blue }, { l: "Value", s: "V = x @ W_V", c: C.green }].map((v, i) => (
							<div key={i} style={{ background: v.c + "15", border: `1px solid ${v.c}33`, borderRadius: 8, padding: "10px 16px", textAlign: "center", flex: "1 1 120px" }}>
								<div style={{ color: v.c, fontWeight: 700, fontSize: 14 }}>{v.l}</div>
								<code style={{ fontSize: 11, color: v.c }}>{v.s}</code>
								<div style={{ fontSize: 9, color: C.dim, marginTop: 4 }}>512 x 512 = 262K params</div>
							</div>
						))}
					</div>
					<div style={{ marginTop: 10, textAlign: "center" }}>
						<div style={{ fontFamily: "monospace", fontSize: 12, color: C.text, background: C.bg, borderRadius: 6, padding: 10, display: "inline-block" }}>
							Attn(Q,K,V) = <span style={{ color: C.pink }}>softmax</span>(<span style={{ color: C.amber }}>Q</span> . <span style={{ color: C.blue }}>K</span><sup>T</sup> / <span style={{ color: C.dim }}>sqrt(d)</span>) . <span style={{ color: C.green }}>V</span>
						</div>
					</div>
				</Box>
			)}
			{sub === 1 && (
				<Box color={C.pink}>
					<Label color={C.pink}>Causal Mask + Attention Scores</Label>
					<Dim>Each token can only attend to itself and previous tokens — never future ones. The mask sets future positions to -infinity before softmax.</Dim>
					<div style={{ background: C.bg, borderRadius: 8, padding: 14, marginTop: 10 }}>
						<div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 3, maxWidth: 250, margin: "0 auto" }}>
							{["", "def", "fib", "(", "n"].map((h, i) => <div key={`h${i}`} style={{ fontSize: 9, color: C.dim, textAlign: "center", fontFamily: "monospace" }}>{h}</div>)}
							{[
								["def", 1, 0, 0, 0], ["fib", 0.8, 0.2, 0, 0], ["(", 0.1, 0.3, 0.6, 0], ["n", 0.05, 0.4, 0.15, 0.4]
							].map((row, ri) => [
								<div key={`r${ri}`} style={{ fontSize: 9, color: C.dim, fontFamily: "monospace", display: "flex", alignItems: "center" }}>{row[0]}</div>,
								...row.slice(1).map((v, ci) => (
									<div key={`${ri}-${ci}`} style={{ width: "100%", aspectRatio: "1", borderRadius: 4, background: v > 0 ? `rgba(244,114,182,${v})` : C.cardL, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: v > 0 ? C.white : C.dim }}>
										{v > 0 ? v.toFixed(1) : "-inf"}
									</div>
								))
							])}
						</div>
						<div style={{ fontSize: 10, color: C.dim, textAlign: "center", marginTop: 8 }}>upper triangle = masked (can't see future tokens)</div>
					</div>
				</Box>
			)}
			{sub === 2 && (
				<Box color={C.pink}>
					<Label color={C.pink}>Multi-Head: 8 Parallel Attention Heads</Label>
					<Dim>512 dims split into 8 heads x 64 dims. Each head specializes in different patterns, then results are concatenated.</Dim>
					<div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 10, justifyContent: "center" }}>
						{["Syntax", "Variables", "Scope", "Types", "Indent", "Flow", "Names", "Patterns"].map((h, i) => (
							<div key={i} style={{ background: C.bg, borderRadius: 6, padding: "6px 10px", textAlign: "center", border: `1px solid ${C.pink}33` }}>
								<div style={{ fontSize: 9, color: C.dim }}>Head {i + 1}</div>
								<div style={{ fontSize: 10, color: C.pink, fontWeight: 600 }}>{h}</div>
								<div style={{ fontSize: 8, color: C.dim }}>64 dims</div>
							</div>
						))}
					</div>
					<div style={{ textAlign: "center", marginTop: 8, fontSize: 11, color: C.dim }}>
						concat all 8 heads (8 x 64 = 512) then project through W_O (512 x 512)
					</div>
					<div style={{ marginTop: 8 }}><Tag color={C.pink}>Total attention params per layer: 1.05M</Tag></div>
				</Box>
			)}
			{sub === 3 && (
				<Box color={C.purple}>
					<Label color={C.purple}>RoPE — Rotary Position Embeddings</Label>
					<Dim>Rotates Q and K vectors based on position. The dot product naturally encodes relative distance between tokens — no extra params needed.</Dim>
					<div style={{ background: C.bg, borderRadius: 10, padding: 16, marginTop: 10 }}>
						<svg viewBox="0 0 300 160" style={{ width: "100%", height: "auto" }}>
							<circle cx="150" cy="85" r="60" fill="none" stroke={C.cardL} strokeWidth="1.5" />
							{["def", "fib", "(", "n", ")", ":"].map((t, i) => {
								const a = (i * 50 - 90) * Math.PI / 180;
								const x = 150 + 60 * Math.cos(a), y = 85 + 60 * Math.sin(a);
								const tx = 150 + 78 * Math.cos(a), ty = 85 + 78 * Math.sin(a);
								return (
									<g key={i}>
										<line x1="150" y1="85" x2={x} y2={y} stroke={C.purple + "44"} strokeWidth="1" />
										<circle cx={x} cy={y} r="4" fill={C.purple} />
										<text x={tx} y={ty} textAnchor="middle" dominantBaseline="middle" fill={C.dim} fontSize="9" fontFamily="monospace">{t}</text>
									</g>
								);
							})}
							<text x="150" y="15" textAnchor="middle" fill={C.purple} fontSize="10" fontWeight="600">rotation angle = position</text>
							<text x="150" y="155" textAnchor="middle" fill={C.dim} fontSize="9">Q.K dot product encodes relative distance</text>
						</svg>
					</div>
					<div style={{ marginTop: 8, fontSize: 11, color: C.dim, lineHeight: 1.6 }}>
						<span style={{ color: C.purple }}>Low-freq dims</span> = long-range position (which function am I in?)<br />
						<span style={{ color: C.purple }}>High-freq dims</span> = local position (which line/token?)
					</div>
				</Box>
			)}
		</div>
	);
}

function SwiGLUSection() {
	return (
		<Box color={C.amber}>
			<Label color={C.amber}>SwiGLU Feed-Forward Network</Label>
			<Dim>After attention mixes tokens, SwiGLU processes each token independently. Uses a gating mechanism: the gate learns to selectively pass or suppress information.</Dim>
			<div style={{ background: C.bg, borderRadius: 8, padding: 14, marginTop: 10, textAlign: "center" }}>
				<div style={{ fontFamily: "monospace", fontSize: 12, color: C.text, lineHeight: 2 }}>
					SwiGLU(x) = (<span style={{ color: C.pink }}>SiLU</span>(x @ <span style={{ color: C.amber }}>W_gate</span>) <span style={{ color: C.dim }}>*</span> x @ <span style={{ color: C.blue }}>W_up</span>) @ <span style={{ color: C.green }}>W_down</span>
				</div>
				<div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
					{[{ l: "W_gate", d: "512→1376", c: C.amber }, { l: "W_up", d: "512→1376", c: C.blue }, { l: "W_down", d: "1376→512", c: C.green }].map((w, i) => (
						<div key={i} style={{ background: w.c + "15", borderRadius: 6, padding: "6px 12px", textAlign: "center" }}>
							<div style={{ fontSize: 11, fontWeight: 600, color: w.c }}>{w.l}</div>
							<div style={{ fontSize: 9, color: C.dim }}>{w.d}</div>
						</div>
					))}
				</div>
			</div>
			<div style={{ marginTop: 10, fontSize: 11, color: C.dim, lineHeight: 1.6 }}>
				<strong style={{ color: C.amber }}>Why 1376 not 2048?</strong> SwiGLU has 3 matrices vs 2. To match param count: (2/3) * 4 * 512 = 1365, rounded to 1376 (multiple of 32 for GPU efficiency).
			</div>
			<Tag color={C.amber}>FFN params per layer: 2.1M</Tag>
		</Box>
	);
}

function ResidualSection() {
	return (
		<Box color={C.green}>
			<Label color={C.green}>Residual (Skip) Connections</Label>
			<Dim>Each sub-layer's output is added to its input, not replacing it. This creates gradient highways that enable training 12+ layers deep.</Dim>
			<div style={{ background: C.bg, borderRadius: 8, padding: 14, marginTop: 10 }}>
				<div style={{ display: "flex", flexDirection: "column", gap: 6, maxWidth: 300, margin: "0 auto" }}>
					{[
						{ l: "x", c: C.text, type: "input" },
						{ l: "RMSNorm", c: C.cyan, type: "op" },
						{ l: "Attention", c: C.pink, type: "op" },
						{ l: "+ residual", c: C.green, type: "add" },
						{ l: "RMSNorm", c: C.cyan, type: "op" },
						{ l: "SwiGLU", c: C.amber, type: "op" },
						{ l: "+ residual", c: C.green, type: "add" },
					].map((s, i) => (
						<div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
							<div style={{ width: 4, height: 28, background: s.type === "add" ? C.green : C.cardL, borderRadius: 2 }} />
							<div style={{ flex: 1, background: s.type === "add" ? C.green + "20" : C.cardL, borderRadius: 6, padding: "5px 12px", fontSize: 12, color: s.c, fontWeight: s.type === "add" ? 700 : 400, fontFamily: "monospace" }}>
								{s.l}
								{s.type === "add" && <span style={{ float: "right", fontSize: 10, color: C.dim }}>skip connection</span>}
							</div>
						</div>
					))}
				</div>
			</div>
			<div style={{ marginTop: 10, fontSize: 11, color: C.dim }}>
				Without residuals: gradients vanish after a few layers. With: each layer adds a small delta, gradients flow directly through +. 24 residual connections total (2 per block x 12 blocks).
			</div>
		</Box>
	);
}

function OutputSection() {
	return (
		<Box color={C.purple}>
			<Label color={C.purple}>Output Projection + Softmax</Label>
			<Dim>Final hidden state (512-dim) is multiplied by the embedding matrix transposed to get logits for all 8,192 tokens. Softmax converts to probabilities.</Dim>
			<div style={{ background: C.bg, borderRadius: 8, padding: 14, marginTop: 10 }}>
				<div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
					{[
						{ l: "hidden [seq_len, 512]", c: C.text },
						{ l: "@ W_embed.T (tied weights)", c: C.purple },
						{ l: "logits [seq_len, 8192]", c: C.amber },
						{ l: "softmax", c: C.pink },
						{ l: "probabilities [seq_len, 8192]", c: C.green },
					].map((s, i) => (
						<div key={i}>
							{i > 0 && <div style={{ textAlign: "center", color: C.dim, fontSize: 12 }}>↓</div>}
							<div style={{ background: C.cardL, borderRadius: 6, padding: "5px 16px", fontSize: 11, color: s.c, fontFamily: "monospace", textAlign: "center" }}>{s.l}</div>
						</div>
					))}
				</div>
			</div>
			<div style={{ marginTop: 10, fontSize: 11, color: C.dim }}>
				<strong style={{ color: C.purple }}>Tied embeddings:</strong> W_output = W_embed.T. The embedding asks "what vector represents this token?" The output asks "which token matches this vector?" Same question, reversed. Saves 4.2M params.
			</div>
		</Box>
	);
}

function ForwardPass() {
	const [step, setStep] = useState(0);
	const steps = [
		{ l: "Tokenize", icon: "✂️", c: C.cyan, desc: '"def fib" -> [142, 3891]' },
		{ l: "Embed", icon: "📊", c: C.amber, desc: "142 -> [0.02, -0.15, ...] (512 floats)" },
		{ l: "Block 1", icon: "⚙️", c: C.pink, desc: '"fib" attends to "def" (0.8) + itself (0.2). Now carries context.' },
		{ l: "Blocks 2-12", icon: "🔁", c: C.purple, desc: "Early: syntax. Mid: semantics. Late: high-level patterns." },
		{ l: "Output", icon: "🎯", c: C.green, desc: 'softmax -> P("onacci")=0.67, P("_")=0.12, P("(")=0.08' },
	];
	return (
		<Box color={C.rose}>
			<Label color={C.rose}>Full Forward Pass: "def fib"</Label>
			<div style={{ display: "flex", gap: 4, marginTop: 8, marginBottom: 12, flexWrap: "wrap" }}>
				{steps.map((s, i) => (
					<div key={i} onClick={() => setStep(i)} style={{ cursor: "pointer", flex: "1 1 60px", background: step === i ? s.c + "20" : C.bg, border: `1px solid ${step === i ? s.c : C.border}`, borderRadius: 6, padding: "6px 4px", textAlign: "center", transition: "all 0.2s" }}>
						<div style={{ fontSize: 14 }}>{s.icon}</div>
						<div style={{ fontSize: 9, color: step === i ? s.c : C.dim, fontWeight: 600 }}>{s.l}</div>
					</div>
				))}
			</div>
			<div style={{ background: C.bg, borderRadius: 8, padding: 14, borderLeft: `3px solid ${steps[step].c}` }}>
				<div style={{ fontSize: 13, fontWeight: 700, color: steps[step].c, marginBottom: 4 }}>{steps[step].icon} {steps[step].l}</div>
				<div style={{ fontSize: 12, color: C.dim, fontFamily: "monospace", lineHeight: 1.6 }}>{steps[step].desc}</div>
			</div>
			{step === 4 && (
				<div style={{ marginTop: 10, textAlign: "center", background: C.green + "15", borderRadius: 8, padding: 12 }}>
					<div style={{ fontFamily: "monospace", fontSize: 20, fontWeight: 700, color: C.green }}>def fib → def fibonacci</div>
				</div>
			)}
		</Box>
	);
}

function Stage1() {
	const [tab, setTab] = useState(0);
	return (
		<div>
			<Tabs tabs={["Dataset", "Tokenizer", "Model Config", "Training Loop", "Evaluation"]} active={tab} onChange={setTab} color={C.cyan} />
			{tab === 0 && (
				<Box color={C.cyan}>
					<Label color={C.cyan}>Step 1: Dataset Preparation</Label>
					<Dim>Download Python functions from code_search_net. Quality filters: skip tiny/huge files, auto-generated code, and duplicates.</Dim>
					<div style={{ marginTop: 10 }}>
						<BarChart data={[
							{ label: "Train", value: 403904, display: "403,904", color: C.cyan },
							{ label: "Val", value: 8243, display: "8,243", color: C.blue },
						]} />
					</div>
					<div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
						<Tag color={C.cyan}>~559MB total</Tag><Tag color={C.cyan}>JSONL format</Tag><Tag color={C.cyan}>code_search_net</Tag>
					</div>
					<div style={{ background: C.bg, borderRadius: 6, padding: 10, marginTop: 10, fontFamily: "monospace", fontSize: 10, color: C.dim, lineHeight: 1.6, overflow: "auto" }}>
						{`{"text": "\\"\\"\\"Return the nth Fibonacci number.\\"\\"\\"\\ndef fibonacci(n):\\n    if n <= 1:\\n        return n"}`}
					</div>
				</Box>
			)}
			{tab === 1 && (
				<Box color={C.amber}>
					<Label color={C.amber}>Step 2: BPE Tokenizer Training</Label>
					<Dim>Start with 256 bytes, repeatedly merge the most frequent adjacent pair until vocab_size=8,192. Python-specific tokens like "def", "return", indentation patterns get their own entries.</Dim>
					<div style={{ background: C.bg, borderRadius: 8, padding: 12, marginTop: 10 }}>
						{[
							{ step: "0", tokens: "[d] [e] [f] [ ] [f] [i] [b]", desc: "individual bytes" },
							{ step: "1", tokens: "[de] [f] [ ] [f] [i] [b]", desc: "merge d+e" },
							{ step: "2", tokens: "[def] [ ] [f] [i] [b]", desc: "merge de+f" },
							{ step: "N", tokens: '[def] [_fib] [onacci]', desc: "final tokens" },
						].map((s, i) => (
							<div key={i} style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 6 }}>
								<div style={{ fontSize: 10, color: C.amber, fontWeight: 700, width: 40 }}>Step {s.step}</div>
								<div style={{ flex: 1, fontFamily: "monospace", fontSize: 11, color: C.text }}>{s.tokens}</div>
								<div style={{ fontSize: 10, color: C.dim }}>{s.desc}</div>
							</div>
						))}
					</div>
				</Box>
			)}
			{tab === 2 && (
				<Box color={C.blue}>
					<Label color={C.blue}>Step 3: Model Architecture (42.1M params)</Label>
					<div style={{ marginTop: 8 }}>
						<BarChart data={[
							{ label: "Embedding", value: 4.2, display: "4.2M", color: C.amber },
							{ label: "Attn x12", value: 12.6, display: "12.6M", color: C.pink },
							{ label: "FFN x12", value: 25.4, display: "25.4M", color: C.amber },
							{ label: "Norms", value: 0.013, display: "12.8K", color: C.cyan },
						]} maxVal={26} />
					</div>
					<div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
						{[
							{ l: "hidden", v: "512" }, { l: "layers", v: "12" }, { l: "heads", v: "8" },
							{ l: "FFN dim", v: "1376" }, { l: "vocab", v: "8192" }, { l: "ctx", v: "2048" },
						].map((d, i) => (
							<div key={i} style={{ background: C.bg, borderRadius: 4, padding: "4px 10px", textAlign: "center" }}>
								<div style={{ fontSize: 9, color: C.dim }}>{d.l}</div>
								<div style={{ fontSize: 13, fontWeight: 700, color: C.blue, fontFamily: "monospace" }}>{d.v}</div>
							</div>
						))}
					</div>
				</Box>
			)}
			{tab === 3 && (
				<Box color={C.green}>
					<Label color={C.green}>Step 4: Training Loop</Label>
					<Dim>Teacher forcing: input = tokens[:-1], target = tokens[1:]. Model predicts each next token. AdamW optimizer with cosine LR schedule.</Dim>
					<div style={{ background: C.bg, borderRadius: 8, padding: 12, marginTop: 10 }}>
						<div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center", marginBottom: 10 }}>
							{[{ l: "batch", v: "8" }, { l: "LR", v: "3e-4" }, { l: "decay", v: "cosine" }, { l: "steps", v: "50,488" }].map((d, i) => (
								<div key={i} style={{ background: C.cardL, borderRadius: 4, padding: "4px 10px", textAlign: "center" }}>
									<div style={{ fontSize: 9, color: C.dim }}>{d.l}</div>
									<div style={{ fontSize: 12, fontWeight: 600, color: C.green, fontFamily: "monospace" }}>{d.v}</div>
								</div>
							))}
						</div>
						<div style={{ fontSize: 11, color: C.dim, textAlign: "center" }}>
							LR: 3e-4 → cosine decay → 3e-6 (1% of initial)
						</div>
					</div>
					<Label color={C.green} style={{ marginTop: 12 }}>Training Progress</Label>
					<BarChart data={[
						{ label: "Step 0", value: 9.52, display: "loss 9.52", color: C.red },
						{ label: "Step 100", value: 8.01, display: "loss 8.01", color: C.orange },
						{ label: "Step 500", value: 6.5, display: "loss 6.5", color: C.amber },
						{ label: "Step 2K", value: 5.2, display: "loss 5.2", color: C.lime },
						{ label: "Step 4K", value: 4.5, display: "loss 4.5", color: C.green },
					]} maxVal={10} />
				</Box>
			)}
			{tab === 4 && (
				<Box color={C.purple}>
					<Label color={C.purple}>Step 5: Evaluation</Label>
					<Dim>Convert to MLX-LM format, test interactively and run benchmarks. A 50M model learns Python syntax and common patterns but won't reliably solve problems. That's expected — this stage is about learning the pipeline.</Dim>
					<div style={{ background: C.bg, borderRadius: 8, padding: 12, marginTop: 10, fontFamily: "monospace", fontSize: 11, color: C.dim, lineHeight: 1.6 }}>
						<span style={{ color: C.green }}>$</span> python src/eval/interactive.py --run "Python-Llama-50M"<br />
						<span style={{ color: C.amber }}>{`>>>`}</span> def fibonacci(n):<br />
						<span style={{ color: C.text }}>&nbsp;&nbsp;# model generates syntactically valid but simple code</span>
					</div>
				</Box>
			)}
		</div>
	);
}

function Stage2() {
	const [tab, setTab] = useState(0);
	return (
		<div>
			<Tabs tabs={["Synthetic Data", "LoRA Fine-tuning", "Merge & Eval"]} active={tab} onChange={setTab} color={C.purple} />
			{tab === 0 && (
				<Box color={C.orange}>
					<Label color={C.orange}>Step 6: Synthetic Data Generation</Label>
					<Dim>Two approaches: transform existing corpus into instruction format (5K examples, free), and Claude-generated novel examples (200, high quality).</Dim>
					<div style={{ marginTop: 10 }}>
						<BarChart data={[
							{ label: "Explain", value: 35, display: "35%", color: C.cyan },
							{ label: "Docstring", value: 25, display: "25%", color: C.blue },
							{ label: "Testing", value: 25, display: "25%", color: C.green },
							{ label: "Type hints", value: 15, display: "15%", color: C.purple },
						]} maxVal={40} />
					</div>
					<div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
						<Tag color={C.orange}>4,940 train</Tag><Tag color={C.orange}>260 val</Tag><Tag color={C.orange}>ChatML format</Tag>
					</div>
				</Box>
			)}
			{tab === 1 && (
				<Box color={C.purple}>
					<Label color={C.purple}>Step 7: LoRA Fine-tuning</Label>
					<Dim>Base: Qwen2.5-Coder-0.5B-Instruct (494M params, 5.5T tokens pre-training). LoRA rank=32, only 1.4M trainable params.</Dim>
					<div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap", justifyContent: "center" }}>
						{[{ l: "rank", v: "32" }, { l: "alpha", v: "64" }, { l: "dropout", v: "0.05" }, { l: "iters", v: "1000" }].map((d, i) => (
							<div key={i} style={{ background: C.bg, borderRadius: 4, padding: "4px 10px", textAlign: "center" }}>
								<div style={{ fontSize: 9, color: C.dim }}>{d.l}</div>
								<div style={{ fontSize: 13, fontWeight: 700, color: C.purple, fontFamily: "monospace" }}>{d.v}</div>
							</div>
						))}
					</div>
					<div style={{ marginTop: 12 }}>
						<Label color={C.purple}>Val Loss Curve</Label>
						<BarChart data={[
							{ label: "Iter 1", value: 0.739, display: "0.739", color: C.amber },
							{ label: "Iter 300", value: 0.719, display: "0.719", color: C.lime },
							{ label: "Iter 600", value: 0.690, display: "0.690 (best)", color: C.green },
							{ label: "Iter 900", value: 0.728, display: "0.728 (overfit)", color: C.red },
							{ label: "Iter 1K", value: 0.707, display: "0.707 (final)", color: C.orange },
						]} maxVal={0.8} />
					</div>
					<div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
						<Tag color={C.green}>Peak mem: 10.6 GB</Tag><Tag color={C.green}>~3.9 hours</Tag><Tag color={C.green}>Best @ iter 600</Tag>
					</div>
				</Box>
			)}
			{tab === 2 && (
				<Box color={C.green}>
					<Label color={C.green}>Step 8: Merge LoRA + Evaluate</Label>
					<Dim>Fuse adapter weights into base model: W_merged = W_original + B @ A. Result is a standalone model identical in behavior.</Dim>
					<div style={{ background: C.bg, borderRadius: 8, padding: 14, marginTop: 10, textAlign: "center" }}>
						<div style={{ fontFamily: "monospace", fontSize: 12, lineHeight: 2, color: C.text }}>
							<span style={{ color: C.dim }}>Before:</span> output = <span style={{ color: C.blue }}>W</span> @ x + (<span style={{ color: C.purple }}>B</span> @ <span style={{ color: C.purple }}>A</span>) @ x<br />
							<span style={{ color: C.dim }}>After:</span>&nbsp; <span style={{ color: C.green }}>W_merged</span> = <span style={{ color: C.blue }}>W</span> + <span style={{ color: C.purple }}>B</span> @ <span style={{ color: C.purple }}>A</span><br />
							<span style={{ color: C.dim }}>Result:</span> output = <span style={{ color: C.green }}>W_merged</span> @ x
						</div>
					</div>
				</Box>
			)}
		</div>
	);
}

function Stage3() {
	const [tab, setTab] = useState(0);
	return (
		<div>
			<Tabs tabs={["Pipeline", "GGUF Convert", "Quantization", "Ollama"]} active={tab} onChange={setTab} color={C.green} />
			{tab === 0 && (
				<Box color={C.green}>
					<Label color={C.green}>Shipping Pipeline</Label>
					<div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 10 }}>
						{[
							{ l: "MLX fine-tuned model", s: "adapter + base, Mac only", c: C.purple },
							{ l: "Fused HuggingFace model", s: "standard format, any tool", c: C.blue },
							{ l: "GGUF f16", s: "portable single file, ~6 GB", c: C.amber },
							{ l: "GGUF Q4_K_M", s: "compressed, ~1.5 GB, ~95% quality", c: C.orange },
							{ l: "ollama run yosii/python-expert", s: "anyone, anywhere", c: C.green },
						].map((s, i) => (
							<div key={i}>
								{i > 0 && <div style={{ textAlign: "center", color: C.dim, fontSize: 12 }}>↓</div>}
								<div style={{ background: C.bg, borderRadius: 6, padding: "8px 14px", borderLeft: `3px solid ${s.c}` }}>
									<span style={{ fontSize: 12, fontWeight: 600, color: s.c }}>{s.l}</span>
									<span style={{ fontSize: 10, color: C.dim, marginLeft: 8 }}>{s.s}</span>
								</div>
							</div>
						))}
					</div>
				</Box>
			)}
			{tab === 1 && (
				<Box color={C.amber}>
					<Label color={C.amber}>GGUF Conversion</Label>
					<Dim>GGUF = single portable binary. No Python, no framework deps. Any C++ program can memory-map directly. Used by llama.cpp, Ollama, LM Studio.</Dim>
					<div style={{ marginTop: 10, display: "flex", gap: 12, flexWrap: "wrap" }}>
						<div style={{ flex: 1, minWidth: 140, background: C.bg, borderRadius: 6, padding: 10 }}>
							<div style={{ fontSize: 10, color: C.dim, marginBottom: 4 }}>HuggingFace (multiple files)</div>
							{["config.json", "tokenizer.json", "model-00001.safetensors", "model-00002.safetensors"].map((f, i) => (
								<div key={i} style={{ fontSize: 10, fontFamily: "monospace", color: C.amber, marginBottom: 2 }}>{f}</div>
							))}
						</div>
						<div style={{ flex: 1, minWidth: 140, background: C.bg, borderRadius: 6, padding: 10 }}>
							<div style={{ fontSize: 10, color: C.dim, marginBottom: 4 }}>GGUF (single file)</div>
							<div style={{ fontSize: 10, fontFamily: "monospace", color: C.green }}>python-expert-f16.gguf</div>
							<div style={{ fontSize: 9, color: C.dim, marginTop: 4 }}>header + tokenizer + all weights</div>
						</div>
					</div>
				</Box>
			)}
			{tab === 2 && (
				<Box color={C.orange}>
					<Label color={C.orange}>Quantization: Q4_K_M</Label>
					<Dim>Reduce from 16 bits to 4 bits per weight. K-quant uses different bit widths for different layers (sensitive layers get more bits).</Dim>
					<div style={{ marginTop: 10 }}>
						<BarChart data={[
							{ label: "F16", value: 6, display: "~6 GB / 100%", color: C.blue },
							{ label: "Q8_0", value: 3, display: "~3 GB / ~99%", color: C.cyan },
							{ label: "Q5_K_M", value: 2, display: "~2 GB / ~97%", color: C.green },
							{ label: "Q4_K_M", value: 1.5, display: "~1.5 GB / ~95%", color: C.amber },
							{ label: "Q3_K_M", value: 1.2, display: "~1.2 GB / ~90%", color: C.red },
						]} maxVal={6.5} />
					</div>
					<div style={{ marginTop: 10, background: C.bg, borderRadius: 6, padding: 10, fontSize: 11, color: C.dim, lineHeight: 1.6 }}>
						<strong style={{ color: C.orange }}>How:</strong> Group 256 weights, share a float16 scale factor. Each weight stored as 4-bit int (-8 to 7). Reconstruct: index * scale.
					</div>
				</Box>
			)}
			{tab === 3 && (
				<Box color={C.green}>
					<Label color={C.green}>Ollama Packaging</Label>
					<Dim>Modelfile defines: which weights to load, ChatML template (must match training format), system prompt, inference params.</Dim>
					<div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap", justifyContent: "center" }}>
						{[{ l: "temp", v: "0.2" }, { l: "top_p", v: "0.9" }, { l: "ctx", v: "4096" }, { l: "format", v: "ChatML" }].map((d, i) => (
							<div key={i} style={{ background: C.bg, borderRadius: 4, padding: "4px 10px", textAlign: "center" }}>
								<div style={{ fontSize: 9, color: C.dim }}>{d.l}</div>
								<div style={{ fontSize: 13, fontWeight: 700, color: C.green, fontFamily: "monospace" }}>{d.v}</div>
							</div>
						))}
					</div>
					<div style={{ marginTop: 10, background: C.bg, borderRadius: 6, padding: 10, fontFamily: "monospace", fontSize: 11, color: C.green, textAlign: "center" }}>
						ollama run yosii/python-expert
					</div>
				</Box>
			)}
		</div>
	);
}

function AppendixSection() {
	const [tab, setTab] = useState(0);
	return (
		<div>
			<Tabs tabs={["Param Formula", "Memory", "File Sizes", "Speed"]} active={tab} onChange={setTab} color={C.sky} />
			{tab === 0 && (
				<Box color={C.sky}>
					<Label color={C.sky}>Parameter Count Formula</Label>
					<div style={{ background: C.bg, borderRadius: 6, padding: 12, fontFamily: "monospace", fontSize: 11, color: C.text, lineHeight: 1.8 }}>
						params = <span style={{ color: C.amber }}>vocab * hidden</span><br />
						&nbsp;&nbsp;+ layers * (<br />
						&nbsp;&nbsp;&nbsp;&nbsp;<span style={{ color: C.pink }}>4 * hidden^2</span> <span style={{ color: C.dim }}>// attention</span><br />
						&nbsp;&nbsp;&nbsp;&nbsp;+ <span style={{ color: C.amber }}>3 * hidden * intermediate</span> <span style={{ color: C.dim }}>// SwiGLU</span><br />
						&nbsp;&nbsp;&nbsp;&nbsp;+ <span style={{ color: C.cyan }}>2 * hidden</span> <span style={{ color: C.dim }}>// norms</span><br />
						&nbsp;&nbsp;)
					</div>
				</Box>
			)}
			{tab === 1 && (
				<Box color={C.sky}>
					<Label color={C.sky}>Training Memory</Label>
					<Dim>~= weights (2B bf16) + gradients (2B) + optimizer state (8B for Adam) + activations</Dim>
					<div style={{ marginTop: 10 }}>
						<BarChart data={[
							{ label: "50M model", value: 2, display: "~2 GB", color: C.cyan },
							{ label: "LoRA 0.5B", value: 10.6, display: "10.6 GB", color: C.purple },
							{ label: "LoRA 3B", value: 12, display: "~12 GB", color: C.pink },
							{ label: "LoRA 0.5B*", value: 28.2, display: "28.2 GB (no grad ckpt!)", color: C.red },
						]} maxVal={30} />
					</div>
				</Box>
			)}
			{tab === 2 && (
				<Box color={C.sky}>
					<Label color={C.sky}>File Sizes</Label>
					<div style={{ marginTop: 6 }}>
						<BarChart data={[
							{ label: "Corpus", value: 548, display: "548 MB", color: C.cyan },
							{ label: "Synth data", value: 10, display: "~10 MB", color: C.orange },
							{ label: "Tokenizer", value: 0.24, display: "240 KB", color: C.amber },
							{ label: "50M model", value: 84, display: "84 MB", color: C.blue },
							{ label: "0.5B model", value: 988, display: "988 MB", color: C.purple },
							{ label: "3B GGUF f16", value: 6000, display: "~6 GB", color: C.pink },
							{ label: "3B Q4_K_M", value: 1500, display: "~1.5 GB", color: C.green },
							{ label: "LoRA adapt", value: 13, display: "13 MB", color: C.lime },
						]} />
					</div>
				</Box>
			)}
			{tab === 3 && (
				<Box color={C.sky}>
					<Label color={C.sky}>Tokens Per Second (M4 Pro)</Label>
					<div style={{ marginTop: 6 }}>
						<BarChart data={[
							{ label: "50M train", value: 500, display: "~500 tok/s", color: C.cyan },
							{ label: "0.5B LoRA", value: 1100, display: "~1,100 tok/s", color: C.purple },
							{ label: "3B LoRA", value: 380, display: "~380 tok/s", color: C.pink },
							{ label: "0.5B infer", value: 75, display: "~75 tok/s", color: C.blue },
							{ label: "3B infer", value: 45, display: "~45 tok/s", color: C.green },
						]} />
					</div>
				</Box>
			)}
		</div>
	);
}

// ─── MAIN APP ───
const SECTIONS = [
	{ id: "big-picture", label: "Big Picture", icon: "🗺️", component: BigPicture },
	{ id: "concepts", label: "Key Concepts", icon: "💡", component: LMConcepts },
	{ id: "architecture", label: "Architecture", icon: "🏗️", component: Architecture },
	{ id: "stage1", label: "Stage 1: Train", icon: "🧪", component: Stage1 },
	{ id: "stage2", label: "Stage 2: Distill", icon: "🎓", component: Stage2 },
	{ id: "stage3", label: "Stage 3: Ship", icon: "🚀", component: Stage3 },
	{ id: "appendix", label: "Appendix", icon: "📎", component: AppendixSection },
];

export default function LearningGuide() {
	const [section, setSection] = useState(0);
	const ActiveSection = SECTIONS[section].component;

	return (
		<div style={{ background: C.bg, color: C.text, fontFamily: "'Inter', -apple-system, sans-serif", minHeight: "100vh" }}>
			{/* Header */}
			<div style={{ background: C.card, borderBottom: `1px solid ${C.border}`, padding: "14px 16px", position: "sticky", top: 0, zIndex: 10 }}>
				<div style={{ maxWidth: 680, margin: "0 auto" }}>
					<h1 style={{ fontSize: 17, fontWeight: 800, margin: 0, color: C.white }}>Python Expert SLM — Learning Guide</h1>
					<p style={{ fontSize: 11, color: C.dim, margin: "2px 0 10px" }}>training from scratch + knowledge distillation + shipping to Ollama</p>
					<div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
						{SECTIONS.map((s, i) => (
							<button key={i} onClick={() => setSection(i)} style={{
								background: section === i ? C.pink : "transparent",
								color: section === i ? C.white : C.dim,
								border: `1px solid ${section === i ? C.pink : C.border}`,
								borderRadius: 6, padding: "5px 10px", fontSize: 11,
								cursor: "pointer", fontWeight: 600, transition: "all 0.2s",
								whiteSpace: "nowrap",
							}}>
								{s.icon} {s.label}
							</button>
						))}
					</div>
				</div>
			</div>

			{/* Content */}
			<div style={{ maxWidth: 680, margin: "0 auto", padding: "16px 16px 40px" }}>
				<ActiveSection />
			</div>

			{/* Nav */}
			<div style={{ maxWidth: 680, margin: "0 auto", padding: "0 16px 24px", display: "flex", justifyContent: "space-between" }}>
				<button onClick={() => setSection(Math.max(0, section - 1))} disabled={section === 0} style={{
					background: section === 0 ? C.cardL : C.card, color: section === 0 ? C.dim : C.text,
					border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 20px", fontSize: 13, cursor: section === 0 ? "default" : "pointer", fontWeight: 600,
				}}>← Prev</button>
				<button onClick={() => setSection(Math.min(SECTIONS.length - 1, section + 1))} disabled={section === SECTIONS.length - 1} style={{
					background: section === SECTIONS.length - 1 ? C.cardL : C.pink,
					color: section === SECTIONS.length - 1 ? C.dim : C.white,
					border: "none", borderRadius: 8, padding: "10px 20px", fontSize: 13, cursor: section === SECTIONS.length - 1 ? "default" : "pointer", fontWeight: 600,
				}}>Next →</button>
			</div>
		</div>
	);
}
