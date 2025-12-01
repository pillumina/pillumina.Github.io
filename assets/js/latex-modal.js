// assets/js/latex-modal.js

// 创建弹窗（保持不变）
const latexModal = document.createElement('div');
latexModal.className = 'latex-modal-overlay';
latexModal.innerHTML = `
  <div class="latex-modal">
    <div class="latex-modal__header">
      <span class="latex-modal__title">LaTeX 源码</span>
      <button class="latex-modal__close" aria-label="关闭">&times;</button>
    </div>
    <div class="latex-modal__body">
      <pre class="latex-modal__code"></pre>
    </div>
    <div class="latex-modal__footer">
      <button class="latex-modal__copy-btn">复制代码</button>
      <span class="latex-modal__copy-hint"></span>
    </div>
  </div>
`;
document.body.appendChild(latexModal);

const codeBlock = latexModal.querySelector('.latex-modal__code');
const copyBtn = latexModal.querySelector('.latex-modal__copy-btn');
const copyHint = latexModal.querySelector('.latex-modal__copy-hint');

async function copyLatexCode(text) {
  try {
    await navigator.clipboard.writeText(text);
    copyHint.textContent = '已复制！';
  } catch {
    const t = document.createElement('textarea');
    t.value = text;
    document.body.appendChild(t);
    t.select();
    document.execCommand('copy');
    document.body.removeChild(t);
    copyHint.textContent = '已复制（降级）';
  }
  copyBtn.disabled = true;
  setTimeout(() => { copyHint.textContent = ''; copyBtn.disabled = false; }, 2000);
}

function showLatexModal(latex) {
  codeBlock.textContent = latex;
  latexModal.classList.add('is-visible');
  document.body.style.overflow = 'hidden';
}

function hideLatexModal() {
  latexModal.classList.remove('is-visible');
  document.body.style.overflow = '';
}

document.addEventListener('click', e => {
  const block = e.target.closest('.katex-display');
  if (block?.dataset.latexCode) {
    showLatexModal(block.dataset.latexCode);
    return;
  }
  if (e.target.matches('.latex-modal__close') || e.target === latexModal) {
    hideLatexModal();
  }
});

copyBtn.addEventListener('click', () => copyLatexCode(codeBlock.textContent));

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && latexModal.classList.contains('is-visible')) {
    hideLatexModal();
  }
});

// ==================== 核心修改：轮询提取 ====================
function waitForKaTeXAndExtract() {
  const root = document.querySelector('.post-content') || document.body;
  
  // 每 100ms 检查一次，最多 5 秒
  const interval = setInterval(() => {
    const renderedBlocks = root.querySelectorAll('.katex-display');
    
    if (renderedBlocks.length > 0) {
      // 找到渲染完成的公式，开始提取
      renderedBlocks.forEach(block => {
        // 方法：从 MathML annotation 提取（最可靠）
        const annotation = block.querySelector('annotation[encoding="application/x-tex"]');
        if (annotation) {
          block.dataset.latexCode = annotation.textContent;
        } else {
          // 备选：尝试反解析 KaTeX DOM（复杂，不推荐）
          console.warn('未找到 annotation', block);
        }
        
        if (block.dataset.latexCode) {
          block.style.cursor = 'pointer';
          block.title = '点击查看源码';
        }
      });
      
      clearInterval(interval); // 提取完成，停止轮询
      console.log(`✅ LaTeX 源码提取完成：${renderedBlocks.length} 个公式`);
    }
  }, 100);

  // 5 秒后自动停止
  setTimeout(() => clearInterval(interval), 5000);
}

// 脚本加载后立即启动轮询（不依赖 DOMContentLoaded）
waitForKaTeXAndExtract();
// ==================== 结束：轮询逻辑 ====================