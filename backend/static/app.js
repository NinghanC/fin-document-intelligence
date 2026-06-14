/* ═══════════════════════════════════════════════════════════════════
   Agent Knowledge Hub — Premium Frontend App
   ═══════════════════════════════════════════════════════════════════ */

const API = '/api';

/* ════════════════════ State ════════════════════ */
const state = {
  uploadedDocs: [],
  asking: false,
};

/* ════════════════════ DOM refs ════════════════════ */
const $ = (s) => document.querySelector(s);

const dom = {
  statusDot: $('#statusDot'),
  statusText: $('#statusText'),
  tabs: document.querySelectorAll('.tab-btn'),
  panels: {
    qa: $('#panel-qa'),
    upload: $('#panel-upload'),
    dashboard: $('#panel-dashboard'),
  },
  chatContainer: $('#chatContainer'),
  qaInput: $('#qaInput'),
  qaSendBtn: $('#qaSendBtn'),
  uploadZone: $('#uploadZone'),
  fileInput: $('#fileInput'),
  uploadProgress: $('#uploadProgress'),
  docList: $('#docList'),
  refreshStatsBtn: $('#refreshStatsBtn'),
};

/* ════════════════════ Health Check ════════════════════ */
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    if (d.status === 'ok') {
      dom.statusDot.classList.remove('offline');
      dom.statusText.textContent = d.service || 'System healthy';
    }
  } catch {
    dom.statusDot.classList.add('offline');
    dom.statusText.textContent = 'Service offline';
  }
}
checkHealth();
setInterval(checkHealth, 15000);

/* ════════════════════ Tab Switching ════════════════════ */
dom.tabs.forEach(btn => {
  btn.addEventListener('click', () => {
    dom.tabs.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    Object.values(dom.panels).forEach(p => p.classList.add('hidden'));
    dom.panels[tab].classList.remove('hidden');
    // Re-trigger fade animation
    dom.panels[tab].classList.remove('animate-fade-in-up');
    void dom.panels[tab].offsetWidth;
    dom.panels[tab].classList.add('animate-fade-in-up');
    if (tab === 'dashboard') loadStats();
  });
});

/* ════════════════════ Q&A ════════════════════ */
dom.qaSendBtn.addEventListener('click', sendQuestion);
dom.qaInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
});

async function sendQuestion() {
  const question = dom.qaInput.value.trim();
  if (!question || state.asking) return;

  state.asking = true;
  dom.qaSendBtn.disabled = true;
  dom.qaSendBtn.innerHTML = '<span class="spinner"></span>';
  dom.qaInput.value = '';

  appendMessage('user', question);

  // Show typing indicator
  const typingMsg = showTypingIndicator();

  try {
    const r = await fetch(`${API}/qa/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();

    // Remove typing indicator
    typingMsg.remove();

    appendMessage('agent', data.answer, {
      intent: data.intent,
      confidence: data.confidence,
      sources: data.sources,
      reasoning: data.reasoning_steps,
    });
  } catch (err) {
    typingMsg.remove();
    appendMessage('agent', `Sorry, the request failed: ${err.message}`, { error: true });
  } finally {
    state.asking = false;
    dom.qaSendBtn.disabled = false;
    dom.qaSendBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
  }
}

function showTypingIndicator() {
  const div = document.createElement('div');
  div.className = 'chat-msg agent';
  div.innerHTML = `
    <div class="chat-avatar chat-avatar--agent">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>
    </div>
    <div class="chat-body">
      <div class="chat-bubble chat-bubble--agent">
        <div class="typing-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  `;
  dom.chatContainer.appendChild(div);
  dom.chatContainer.scrollTop = dom.chatContainer.scrollHeight;
  return div;
}

function appendMessage(role, content, meta) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;

  // Avatar
  const avatar = document.createElement('div');
  const avatarClass = role === 'user' ? 'chat-avatar--user' : 'chat-avatar--agent';
  avatar.className = `chat-avatar ${avatarClass}`;
  if (role === 'user') {
    avatar.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
  } else {
    avatar.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>';
  }

  // Body wrapper
  const body = document.createElement('div');
  body.className = 'chat-body';

  // Bubble
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble chat-bubble--${role}`;

  if (meta && meta.error) {
    bubble.innerHTML = `<p style="color:var(--red-500)">${escapeHtml(content)}</p>`;
  } else {
    // Convert newlines to <br>, support simple markdown-like bold
    const lines = escapeHtml(content).split('\n');
    bubble.innerHTML = lines.map(line => `<p>${line || '&nbsp;'}</p>`).join('');
  }

  body.appendChild(bubble);

  // Meta tags
  if (meta) {
    const metaRow = document.createElement('div');
    metaRow.className = 'chat-meta';

    if (meta.intent) {
      const intentTag = document.createElement('span');
      intentTag.className = 'chat-tag chat-tag--intent';
      intentTag.textContent = meta.intent;
      metaRow.appendChild(intentTag);
    }

    if (meta.confidence > 0) {
      const confTag = document.createElement('span');
      confTag.className = 'chat-tag chat-tag--confidence';
      confTag.textContent = `Confidence ${(meta.confidence * 100).toFixed(0)}%`;
      metaRow.appendChild(confTag);
    }

    if (metaRow.children.length > 0) {
      body.appendChild(metaRow);
    }

    // Sources
    if (meta.sources && meta.sources.length > 0) {
      const src = document.createElement('div');
      src.className = 'chat-sources';
      src.innerHTML = '<strong>Sources</strong> &nbsp;' +
        meta.sources.map((s, i) =>
          `[${i + 1}] ${escapeHtml(s.source)} (${(s.score * 100).toFixed(0)}%)`
        ).join(' &nbsp;·&nbsp; ');
      body.appendChild(src);
    }

    // Reasoning steps
    if (meta.reasoning && meta.reasoning.length > 0) {
      const steps = document.createElement('div');
      steps.className = 'reasoning-steps';
      meta.reasoning.forEach(step => {
        const span = document.createElement('span');
        span.className = 'reasoning-step';
        span.textContent = step;
        steps.appendChild(span);
      });
      body.appendChild(steps);
    }
  }

  div.appendChild(avatar);
  div.appendChild(body);
  dom.chatContainer.appendChild(div);

  // Smooth scroll to bottom
  dom.chatContainer.scrollTo({
    top: dom.chatContainer.scrollHeight,
    behavior: 'smooth',
  });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/* ════════════════════ Upload ════════════════════ */
dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
dom.fileInput.addEventListener('change', () => uploadFiles(dom.fileInput.files));

dom.uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  dom.uploadZone.classList.add('drag-over');
});
dom.uploadZone.addEventListener('dragleave', () => {
  dom.uploadZone.classList.remove('drag-over');
});
dom.uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  dom.uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

async function uploadFiles(fileList) {
  const files = Array.from(fileList);
  if (!files.length) return;

  const progress = dom.uploadProgress;
  progress.classList.add('show');
  progress.classList.remove('error');

  for (const file of files) {
    progress.textContent = `Processing: ${file.name} ...`;
    try {
      const form = new FormData();
      form.append('file', file);
      const r = await fetch(`${API}/ingest/upload`, { method: 'POST', body: form });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      addDocToList(file.name, data);
      progress.textContent = `Done: ${file.name} (${data.chunks_count} text chunks, ${data.entities_count} entities, ${data.relations_count} relations)`;
    } catch (err) {
      progress.textContent = `Failed: ${file.name} — ${err.message}`;
      progress.classList.add('error');
    }
  }

  setTimeout(() => progress.classList.remove('show'), 6000);
  dom.fileInput.value = '';
}

function addDocToList(name, data) {
  const item = document.createElement('div');
  item.className = 'doc-item';
  item.innerHTML = `
    <span class="doc-icon">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--emerald-500)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
    </span>
    <span class="doc-name">${escapeHtml(name)}</span>
    <span class="doc-meta">${data.chunks_count} chunks · ${data.entities_count} entities · ${data.relations_count} relations</span>
  `;
  dom.docList.prepend(item);
  state.uploadedDocs.push({ name, data });
}

/* ════════════════════ Dashboard ════════════════════ */
async function loadStats() {
  // Reset to loading state
  ['statVectors', 'statEntities', 'statRelations', 'statBackend'].forEach(id => {
    $(`#${id}`).textContent = '...';
  });

  try {
    const r = await fetch(`${API}/admin/stats`);
    if (!r.ok) throw new Error('Failed');
    const d = await r.json();

    animateValue($('#statVectors'), d.vector_store?.total_vectors ?? 0);
    animateValue($('#statEntities'), d.knowledge_graph?.total_entities ?? 0);
    animateValue($('#statRelations'), d.knowledge_graph?.total_relations ?? 0);

    const backendEl = $('#statBackend');
    backendEl.textContent = d.vector_store?.backend ?? '--';
    backendEl.style.fontSize = '18px';
    backendEl.style.fontWeight = '700';
  } catch {
    ['statVectors', 'statEntities', 'statRelations', 'statBackend'].forEach(id => {
      $(`#${id}`).textContent = 'ERR';
    });
  }
}

function animateValue(el, target) {
  const isNum = typeof target === 'number';
  if (!isNum) {
    el.textContent = target || '--';
    return;
  }
  const duration = 600;
  const start = performance.now();
  const from = 0;

  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // ease-out
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(from + (target - from) * eased);
    el.textContent = current.toLocaleString();
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

dom.refreshStatsBtn.addEventListener('click', loadStats);

/* ════════════════════ Keyboard Shortcuts ════════════════════ */
document.addEventListener('keydown', e => {
  // Ctrl+K / Cmd+K: focus QA input
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    // Switch to QA tab if not active
    const qaTab = document.querySelector('[data-tab="qa"]');
    if (!qaTab.classList.contains('active')) {
      qaTab.click();
    }
    dom.qaInput.focus();
  }

  // Escape: blur input
  if (e.key === 'Escape' && document.activeElement === dom.qaInput) {
    dom.qaInput.blur();
  }
});
