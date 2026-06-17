---
title: "Tools"
description: "Interactive tools for LLM architecture analysis"
---

<div class="tools-listing">

<a href="/tools/llm-calculator/" class="tool-card" target="_blank" rel="noopener">
  <div class="tool-card-icon">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="4" y="2" width="16" height="20" rx="2"/>
      <line x1="8" y1="6" x2="16" y2="6"/>
      <line x1="8" y1="10" x2="10" y2="10"/>
      <line x1="12" y1="10" x2="16" y2="10"/>
      <line x1="8" y1="14" x2="10" y2="14"/>
      <line x1="12" y1="14" x2="16" y2="14"/>
      <line x1="8" y1="18" x2="10" y2="18"/>
      <line x1="12" y1="18" x2="16" y2="18"/>
    </svg>
  </div>
  <div class="tool-card-body">
    <span class="tool-card-name">LLM Architecture Calculator</span>
    <span class="tool-card-desc">从 config.json 到参数量 / FLOPs / KV Cache / 推理显存 — 支持 Full Attention / MSA / MLA / Mamba-2 / SWA / GDN 六种架构</span>
  </div>
  <span class="tool-card-arrow">&rarr;</span>
</a>

</div>

<style>
.tools-listing { display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem; }
.tool-card {
  display: flex; align-items: center; gap: 1.25rem;
  padding: 1.5rem; background: var(--entry);
  border: 1px solid var(--border); border-radius: 12px;
  text-decoration: none; color: inherit;
  transition: transform 0.25s cubic-bezier(0.16,1,0.3,1), box-shadow 0.25s cubic-bezier(0.16,1,0.3,1), border-color 0.25s ease;
  position: relative; overflow: hidden;
}
.tool-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(to right, var(--accent), transparent);
}
.tool-card:hover {
  transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.1);
  border-color: var(--accent);
}
.tool-card-icon {
  width: 56px; height: 56px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: color-mix(in srgb, var(--accent) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
  border-radius: 12px; color: var(--accent);
}
.tool-card-body { flex: 1; display: flex; flex-direction: column; gap: 0.25rem; }
.tool-card-name { font-size: 1.1rem; font-weight: 600; color: var(--primary); }
.tool-card-desc { font-size: 0.85rem; color: var(--secondary); line-height: 1.5; }
.tool-card-arrow { font-size: 1.25rem; color: var(--accent); opacity: 0.5; transition: opacity 0.25s, transform 0.25s; }
.tool-card:hover .tool-card-arrow { opacity: 1; transform: translateX(4px); }
@media (max-width: 640px) {
  .tool-card { padding: 1rem; gap: 0.875rem; }
  .tool-card-icon { width: 44px; height: 44px; }
  .tool-card-name { font-size: 1rem; }
  .tool-card-desc { font-size: 0.8rem; }
}
</style>
