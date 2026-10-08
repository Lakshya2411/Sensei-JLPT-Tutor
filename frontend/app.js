const API = (window.SENSEI_API || "http://localhost:8000").replace(/\/$/, "");
const $ = (sel) => document.querySelector(sel);

// ---------- helpers ----------
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const hasKanji = (s) => /[㐀-鿿々]/.test(s);

async function post(path, body) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join(", ") : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

async function run(form, out, fn) {
  const btn = form.querySelector('button[type="submit"]');
  btn.disabled = true;
  out.innerHTML = '<p class="loading">Thinking… (the free tier can take a few seconds)</p>';
  try {
    out.innerHTML = await fn();
  } catch (e) {
    out.innerHTML = `<p class="error">${esc(e.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
}

function metaLine(answer) {
  if (!answer) return "";
  const parts = [];
  if (answer.provider) parts.push(`answered by ${answer.provider}`);
  if (answer.cached) parts.push("from cache");
  return parts.length ? `<p class="meta">${esc(parts.join(" · "))}</p>` : "";
}

// ---------- rendering ----------
function renderTokens(tokens) {
  if (!tokens?.length) return "";
  const items = tokens
    .map((t) => {
      const rt = hasKanji(t.surface) && t.reading && t.reading !== t.surface ? `<rt>${esc(t.reading)}</rt>` : "<rt>&nbsp;</rt>";
      const gloss = t.gloss?.[0] || t.pos;
      const cls = esc(t.pos.replace(/\s+/g, "-"));
      return `<span class="tok ${cls}" title="${esc(t.pos)} · base: ${esc(t.base)}"><ruby>${esc(t.surface)}${rt}</ruby><small>${esc(gloss)}</small></span>`;
    })
    .join("");
  return `<div class="card"><h2>Word by word</h2><div class="tokens" lang="ja">${items}</div></div>`;
}

function renderSources(sources) {
  if (!sources?.length) return "";
  const items = sources
    .map(
      (s) => `<details class="source">
        <summary>${esc(s.pattern)} <span class="badge">${esc(s.level)}</span></summary>
        <p><b>${esc(s.meaning)}</b></p>
        <p>${esc(s.formation)}</p>
        <p>${esc(s.explanation)}</p>
        ${(s.examples || []).slice(0, 3).map((e) => `<p class="ex" lang="ja">${esc(e.ja)}<span>${esc(e.en)}</span></p>`).join("")}
      </details>`
    )
    .join("");
  return `<div class="card"><h2>Grammar notes used (retrieved)</h2>${items}</div>`;
}

// ---------- explain ----------
$("#explain-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = $("#explain-input").value.trim();
  run(ev.target, $("#explain-out"), async () => {
    const d = await post("/explain", { text });
    const a = d.answer || {};
    const byId = Object.fromEntries((d.sources || []).map((s) => [s.id, s]));
    const grammar = (a.grammar || [])
      .map((g) => {
        const src = g.id ? byId[g.id] : null;
        const level = src ? `<span class="badge">${esc(src.level)}</span>` : "";
        const note = g.id ? "" : '<div class="uncited">Not in the grammar notes: treat with care</div>';
        return `<div class="grammar-item"><b lang="ja">${esc(g.pattern || src?.pattern)}</b>${level}<div>${esc(g.how_it_is_used_here)}</div>${note}</div>`;
      })
      .join("");
    return `
      <div class="card">
        <h2>Translation</h2>
        <p class="translation">${esc(a.translation)}</p>
        ${a.literal ? `<p class="literal">Literally: ${esc(a.literal)}</p>` : ""}
        ${metaLine(a)}
      </div>
      ${renderTokens(d.tokens)}
      ${grammar ? `<div class="card"><h2>Grammar in this sentence</h2>${grammar}</div>` : ""}
      ${a.tip ? `<div class="card"><h2>Tip</h2><p>${esc(a.tip)}</p></div>` : ""}
      ${renderSources(d.sources)}`;
  });
});

// ---------- practice ----------
async function loadGrammar() {
  const sel = $("#practice-select");
  try {
    const res = await fetch(API + "/grammar");
    const items = await res.json();
    const groups = {};
    items.forEach((g) => (groups[g.level] ||= []).push(g));
    sel.innerHTML = Object.keys(groups)
      .sort()
      .reverse()
      .map((lvl) => `<optgroup label="${esc(lvl)}">${groups[lvl].map((g) => `<option value="${esc(g.id)}">${esc(g.pattern)} · ${esc(g.meaning)}</option>`).join("")}</optgroup>`)
      .join("");
  } catch {
    sel.innerHTML = "<option disabled>Couldn't load grammar list</option>";
  }
}

$("#practice-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const grammar_id = $("#practice-select").value;
  const n = Number($("#practice-n").value);
  run(ev.target, $("#practice-out"), async () => {
    const d = await post("/practice", { grammar_id, n });
    const qs = (d.answer.questions || [])
      .map((q, i) => {
        const opts = (q.options || []).length
          ? `<div class="options" data-answer="${esc(q.answer)}">${q.options.map((o) => `<button type="button" lang="ja">${esc(o)}</button>`).join("")}</div>`
          : "";
        return `<div class="q">
          <p class="prompt" lang="ja">${i + 1}. ${esc(q.question)}</p>
          ${opts}
          <details><summary>Show answer</summary><p class="jp">${esc(q.answer)}</p><p>${esc(q.explanation)}</p></details>
        </div>`;
      })
      .join("");
    return `<div class="card"><h2>${esc(d.grammar.pattern)} · ${esc(d.grammar.level)}</h2>${qs}${metaLine(d.answer)}</div>`;
  });
});

$("#practice-out").addEventListener("click", (ev) => {
  const btn = ev.target.closest(".options button");
  if (!btn) return;
  const box = btn.parentElement;
  const right = btn.textContent.trim() === box.dataset.answer.trim();
  btn.classList.add(right ? "right" : "wrong");
  if (right) box.closest(".q").querySelector("details").open = true;
});

// ---------- correct ----------
$("#correct-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = $("#correct-input").value.trim();
  run(ev.target, $("#correct-out"), async () => {
    const d = await post("/correct", { text });
    const a = d.answer || {};
    const errors = (a.errors || [])
      .map(
        (e) => `<div class="grammar-item">
          <div class="fix" lang="ja"><del>${esc(e.original)}</del> → <ins>${esc(e.fix)}</ins></div>
          <div>${esc(e.explanation)}</div>
        </div>`
      )
      .join("");
    const verdict = a.is_correct
      ? '<p class="verdict good">Looks correct! 正解</p>'
      : '<p class="verdict bad">Some fixes needed</p>';
    return `
      <div class="card">
        ${verdict}
        <p class="fix" lang="ja">${esc(a.corrected)}</p>
        <p class="literal">${esc(a.translation)}</p>
        ${errors}
        ${a.more_natural ? `<p><b>More natural:</b> <span class="jp">${esc(a.more_natural)}</span></p>` : ""}
        ${metaLine(a)}
      </div>
      ${renderSources((d.sources || []).filter((s) => (a.errors || []).some((e) => e.grammar_id === s.id)))}`;
  });
});

// ---------- tabs, samples, status ----------
document.querySelectorAll(".tabs button").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((t) => t.setAttribute("aria-selected", t === tab));
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== tab.dataset.tab));
  })
);

document.querySelectorAll(".samples").forEach((box) =>
  box.addEventListener("click", (ev) => {
    if (ev.target.tagName !== "BUTTON") return;
    const input = document.getElementById(box.dataset.target);
    input.value = ev.target.textContent;
    input.form.requestSubmit();
  })
);

(async function checkStatus() {
  const el = $("#status");
  try {
    const h = await (await fetch(API + "/health")).json();
    el.textContent = h.llm_providers.length ? `online · ${h.grammar_points} grammar points` : "online · no API key";
    el.className = "status " + (h.llm_providers.length ? "ok" : "bad");
  } catch {
    el.textContent = "backend offline";
    el.className = "status bad";
  }
  loadGrammar();
})();
