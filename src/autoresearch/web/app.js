/* AutoResearch console. Vanilla JS, no build step. Every server string is escaped. */
(() => {
  "use strict";

  // ---------------------------------------------------------------- helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "n/a" : Number(v).toFixed(d));
  const pct = (v, d = 0) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "n/a" : `${(100 * Number(v)).toFixed(d)}%`);
  const short = (h, n = 12) => (h ? String(h).slice(0, n) : "n/a");
  const icon = (id, cls = "icon") => `<svg class="${cls}"><use href="#i-${id}"></use></svg>`;
  const view = () => $("#view");

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
    const body = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
    if (!res.ok || body.ok === false) throw new Error(body.error || `HTTP ${res.status}`);
    return body;
  }

  function toast(msg, kind = "") {
    const t = $("#toast");
    t.textContent = msg;
    t.className = `toast show ${kind}`;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => (t.className = "toast"), 3200);
  }

  function tierTone(cert) {
    if (!cert) return "muted";
    if (cert.status === "refused") return "violet";
    if (cert.verdict === "affirmed") return cert.tier === "confirmed" ? "ok" : "ok";
    if (cert.tier === "artifact_risk") return "bad";
    return "warn";
  }
  const chip = (text, tone = "muted") => `<span class="chip chip-${tone}"><span class="dot"></span>${esc(text)}</span>`;
  const tierChip = (c) => chip(c?.status === "refused" ? "refused" : c?.tier || c?.status || "unknown", tierTone(c));

  function head(eyebrow, title, sub, actions = "") {
    return `<div class="page-head"><div><div class="eyebrow">${esc(eyebrow)}</div><h1 class="page-title">${esc(title)}</h1>${
      sub ? `<p class="page-sub">${esc(sub)}</p>` : ""
    }</div><div class="page-actions">${actions}</div></div>`;
  }
  const panel = (title, body, meta = "") =>
    `<section class="panel"><div class="panel-head"><h3>${esc(title)}</h3><span class="meta">${esc(meta)}</span></div><div class="panel-body panel-pad">${body}</div></section>`;
  const metric = (label, value, detail = "", tone = "") =>
    `<div class="metric ${tone}"><div class="metric-label">${esc(label)}</div><div class="metric-value">${value}</div><div class="metric-detail">${esc(detail)}</div></div>`;
  function table(cols, rows, empty = "Nothing here yet.") {
    if (!rows.length) return `<div class="table-empty muted">${esc(empty)}</div>`;
    const th = cols.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("");
    const tr = rows
      .map((r, i) => `<tr ${r.__attrs || ""}>${cols.map((c) => `<td class="${c.num ? "num" : ""} ${c.cls || ""}">${c.render ? c.render(r, i) : esc(r[c.key])}</td>`).join("")}</tr>`)
      .join("");
    return `<div class="table-wrap"><table class="data"><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
  }
  const kv = (pairs) => `<dl class="kv">${pairs.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
  const code = (text) => `<pre class="code">${esc(text)}</pre>`;
  const skeleton = () => `<div class="skeleton" style="height:180px"></div>`;

  // Tiny Markdown renderer for README / certificate faces: escape first, then format.
  function markdown(md) {
    const lines = String(md || "").split(/\r?\n/);
    let html = "", inCode = false, inTable = false, inList = false;
    const inline = (s) =>
      esc(s)
        .replace(/`([^`]+)`/g, '<code class="code-inline">$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) => (/^(https?:|#|\/|[\w.-]+\.md$)/.test(u) ? `<a href="${u}">${t}</a>` : t));
    const close = () => {
      if (inTable) { html += "</tbody></table>"; inTable = false; }
      if (inList) { html += "</ul>"; inList = false; }
    };
    for (const line of lines) {
      if (line.startsWith("```")) { close(); html += inCode ? "</pre>" : '<pre class="code">'; inCode = !inCode; continue; }
      if (inCode) { html += esc(line) + "\n"; continue; }
      if (/^\|.*\|$/.test(line.trim())) {
        if (/^\|[\s:|-]+\|$/.test(line.trim())) continue;
        const cells = line.trim().slice(1, -1).split("|").map((c) => inline(c.trim()));
        if (!inTable) { close(); html += `<table><tbody>`; inTable = true; }
        html += `<tr>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
        continue;
      }
      if (/^\s*[-*] /.test(line)) { if (!inList) { close(); html += "<ul>"; inList = true; } html += `<li>${inline(line.replace(/^\s*[-*] /, ""))}</li>`; continue; }
      close();
      const h = line.match(/^(#{1,4}) (.*)/);
      if (h) { const lvl = Math.min(4, h[1].length + 1); html += `<h${lvl}>${inline(h[2])}</h${lvl}>`; continue; }
      if (line.startsWith("> ")) { html += `<p class="claim-text">${inline(line.slice(2))}</p>`; continue; }
      if (line.trim()) html += `<p>${inline(line)}</p>`;
    }
    close();
    if (inCode) html += "</pre>";
    return html;
  }

  // ---------------------------------------------------------------- state & routing
  const state = { status: null, selectedRun: null, live: null, source: null };
  const PAGES = {
    overview: ["OVERVIEW", "INSTRUMENT", renderOverview],
    power: ["POWER LAB", "PRICE A SEARCH", renderPower],
    runs: ["CERTIFICATES", "CLAIMS & REFUSALS", renderRuns],
    atlas: ["ATLAS", "IMMUNOTHERAPY DOMAIN", renderAtlas],
    contract: ["CONTRACT", "THE QUESTION", renderContract],
    calibration: ["CALIBRATION", "OPERATING CHARACTERISTICS", renderCalibration],
    live: ["LIVE RUN", "WATCH IT DECLINE", renderLive],
    docs: ["DOCS", "README", renderDocs],
  };

  function route() {
    const [page, arg] = (location.hash.replace(/^#/, "") || "overview").split("/");
    const entry = PAGES[page] || PAGES.overview;
    $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === (PAGES[page] ? page : "overview")));
    $("#crumb-section").textContent = entry[0];
    $("#crumb-detail").textContent = entry[1];
    view().innerHTML = skeleton();
    entry[2](arg ? decodeURIComponent(arg) : null).catch((err) => {
      view().innerHTML = panel("Could not load this page", `<p class="muted">${esc(err.message)}</p>`);
    });
    window.scrollTo(0, 0);
  }

  async function refreshStatus() {
    try {
      state.status = await api("/api/status");
      const s = state.status;
      const reg = s.registry || {};
      const vault = s.vault || {};
      $("#status-cluster").innerHTML = [
        chip(s.initialized ? (s.demo?.domain || "substrate") : "no substrate", s.initialized ? "ok" : "warn"),
        chip(`registry ${reg.ok ? "chain ok" : "check"}`, reg.ok ? "ok" : "bad"),
        chip(`vault ${vault.ok ? "chain ok" : "check"}`, vault.ok ? "ok" : "bad"),
      ].join("");
      $("#rail-hash").textContent = short(s.demo?.contract_hash, 10);
    } catch (err) {
      $("#status-cluster").innerHTML = chip("server unreachable", "bad");
    }
    return state.status;
  }

  // ---------------------------------------------------------------- overview
  async function renderOverview() {
    const s = (await refreshStatus()) || {};
    const latest = s.latest;
    const cal = s.calibration || {};
    const reg = s.registry || {};
    const rows = cal.rows || [];
    const nullRow = rows.find((r) => Number(r.effect) === 0);
    const claim = latest
      ? `<div class="verdict ${tierTone(latest)}"><div class="verdict-icon">${icon(latest.verdict === "affirmed" ? "check" : "shield")}</div><div class="verdict-main"><div class="verdict-kicker">latest · ${esc(latest.run_id)}</div><div class="verdict-title">${esc(
          latest.status === "refused" ? "Refused, with a prescription" : latest.verdict === "affirmed" ? `Affirmed at tier ${latest.tier}` : `Not affirmed (${latest.tier})`
        )}</div><div class="verdict-text">${esc(latest.claim)}</div></div><a class="btn btn-ghost btn-sm" href="#runs/${encodeURIComponent(latest.dir)}">Open ${icon("arrow")}</a></div>`
      : `<div class="verdict warn"><div class="verdict-main"><div class="verdict-title">No runs yet</div><div class="verdict-text">Run <code class="code-inline">autoresearch demo</code>, or start one under Live Run.</div></div></div>`;
    view().innerHTML = `
      <section class="hero"><div class="hero-grid"><div>
        <div class="eyebrow">discovery with a receipt</div>
        <h1>Prices the search.<br><em>Certifies or refuses.</em></h1>
        <p>Every claim carries the size of the search that produced it, a selection-aware null, one sealed confirmation, an adversary report and a signature. A refusal names its reason and the sample size that would fix it.</p>
        <div class="hero-actions btn-row"><a class="btn btn-primary" href="#live">${icon("play")} Watch a live run</a><a class="btn btn-ghost" href="#power">${icon("power")} Price a search</a></div>
      </div><div class="receipt">
        <div class="receipt-head">REGISTRY</div>
        <div class="receipt-row"><span>contracts registered (denominator)</span><b>${esc(reg.denominator ?? 0)}</b></div>
        <div class="receipt-row"><span>procedures registered before search</span><b>${esc(reg.procedures_registered ?? 0)}</b></div>
        <div class="receipt-row"><span>certificates issued</span><b>${esc(reg.certificates_issued ?? 0)}</b></div>
        <div class="receipt-row"><span>published</span><b>${esc(reg.published ?? 0)}</b></div>
        <div class="receipt-row"><span>operator decisions logged</span><b>${esc(reg.operator_decisions ?? 0)}</b></div>
        <div class="receipt-total receipt-row"><span>ledger head</span><b class="mono">${esc(short(reg.ledger_head, 16))}</b></div>
      </div></div></section>
      <div class="section">${claim}</div>
      <div class="section metric-grid">
        ${metric("certificates", esc(s.runs_count ?? 0), "claims and refusals on disk")}
        ${metric("false-positive rate", nullRow ? pct(nullRow.fpr_decision) : "n/a", nullRow ? `${nullRow.n} pure-null replicates` : "no calibration table", "accent")}
        ${metric("substrate", esc(s.demo?.n_discovery ?? "n/a"), `discovery rows; ${s.demo?.n_confirmation ?? "?"} sealed`)}
        ${metric("sealed budget left", esc(s.vault?.sealed?.budget_remaining ?? "n/a"), `of ${s.vault?.sealed?.budget_total ?? "?"} unsealing queries`, "violet")}
      </div>
      <div class="section split">
        ${panel("Limits stated up front", `<ul class="muted">${Object.values(s.limits || {}).map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`)}
        ${panel("Substrate", kv([
          ["domain", esc(s.demo?.substrate || "none")],
          ["contract", `<span class="mono">${esc(short(s.demo?.contract_hash, 24))}</span>`],
          ["project", `<span class="mono">${esc(s.project || "")}</span>`],
        ]))}
      </div>`;
  }

  // ---------------------------------------------------------------- power lab
  async function renderPower() {
    const fields = [
      ["confirmation_n", "Sealed rows"], ["confirmation_groups", "Sealed groups"], ["k_candidates", "Candidates K"],
      ["effect", "Effect that matters"], ["alpha", "Alpha"], ["target_power", "Target power"],
    ];
    view().innerHTML = head("power oracle", "What can this data show?", "Runs before any hypothesis is proposed. Search is free; confirmation is scarce. This prices both.")
      + `<div class="split"><section class="panel"><div class="panel-head"><h3>Inputs</h3><span class="meta">defaults from the contract</span></div><div class="panel-body panel-pad"><form id="power-form" class="form-grid">${fields
        .map(([k, l]) => `<div class="field"><label for="pf-${k}">${esc(l)}</label><input type="number" step="any" id="pf-${k}" name="${k}"></div>`)
        .join("")}<div class="wide btn-row"><button class="btn btn-primary" type="submit">${icon("power")} Assess</button></div></form></div></section><div id="power-out">${skeleton()}</div></div>`;
    const form = $("#power-form");
    const run = async (params) => {
      $("#power-out").innerHTML = skeleton();
      const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v !== null && v !== undefined));
      const res = await api(`/api/power?${qs}`);
      if (!Object.keys(params).length) for (const [k, v] of Object.entries(res.defaults || {})) { const el = form.elements[k]; if (el && v !== null && v !== undefined) el.value = v; }
      $("#power-out").innerHTML = powerResult(res.assessment);
    };
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      run(Object.fromEntries(new FormData(form))).catch((err) => toast(err.message, "bad"));
    });
    await run({});
  }

  function powerResult(a) {
    const tone = a.verdict === "ANSWERABLE" ? "ok" : a.verdict === "NOT_ANSWERABLE" ? "bad" : "warn";
    const dial = table(
      [
        { label: "alpha", key: "alpha", num: true },
        { label: "MDE (search)", num: true, render: (r) => num(r.mde_search) },
        { label: "power @ effect", num: true, render: (r) => pct(r.power_at_effect) },
        { label: "wrong actions / 100 nulls", num: true, render: (r) => num(r.wrong_actions_per_100_null_decisions, 1) },
        { label: "missed effects / 100", num: true, render: (r) => num(r.missed_real_effects_per_100, 1) },
      ],
      a.operating_points || []
    );
    const designs = (a.designs || [])
      .map((d) => `<div class="check-item"><div class="check-icon">${icon("arrow")}</div><div style="flex:1"><div class="check-name">${esc(d.name)} <span class="mono tiny">· power ${pct(d.power_at_current_n)}${d.required_n ? ` · n for target power ${esc(Math.round(d.required_n))}` : ""}</span></div><div class="tiny muted">Establishes: ${esc(d.establishes || "")}</div>${d.does_not_establish ? `<div class="tiny dim">Does not establish: ${esc(d.does_not_establish)}</div>` : ""}</div></div>`)
      .join("");
    return `<div class="stack">
      <div class="verdict ${tone}"><div class="verdict-icon">${icon(tone === "ok" ? "check" : "x")}</div><div class="verdict-main"><div class="verdict-kicker">verdict</div><div class="verdict-title">${esc(a.verdict)}</div><div class="verdict-text">${esc(a.prescription)}</div></div></div>
      <div class="metric-grid mt16">
        ${metric("effect that matters", num(a.effect_that_matters))}
        ${metric("MDE, one candidate", num(a.mde_plain), "pre-chosen")}
        ${metric(`MDE, best of K=${a.k_candidates}`, num(a.mde_search), "the number nobody computes", tone === "ok" ? "accent" : "warn")}
        ${metric("max K answerable", esc(a.max_k_answerable), `rows needed: ${a.required_n_search ?? "n/a"}`)}
      </div>
      <div class="mt16">${panel("Operating-point dial", dial, "the operating point belongs to the decision")}</div>
      <div class="mt16">${panel("Design advisor", designs || '<p class="muted">No alternatives.</p>', "ranked")}</div>
    </div>`;
  }

  // ---------------------------------------------------------------- certificates
  async function renderRuns(selected) {
    const { runs } = await api("/api/runs");
    const pick = selected || state.selectedRun || runs[0]?.dir;
    view().innerHTML = head("claims and refusals", "Certificates", "A negative certificate is a first-class product with identical structure.", `<a class="btn btn-ghost btn-sm" href="#runs/__reviewer">${icon("shield")} Reviewer: check a certificate</a>`)
      + `<div class="runs-layout"><section class="panel">${table(
        [
          { label: "run", render: (r) => `<span class="mono">${esc(r.run_id)}</span>` },
          { label: "tier", render: (r) => tierChip(r) },
          { label: "delta", num: true, render: (r) => num(r.delta) },
          { label: "claim", render: (r) => `<span class="muted tiny">${esc(String(r.claim || "").slice(0, 90))}${String(r.claim || "").length > 90 ? "…" : ""}</span>` },
        ],
        runs.map((r) => ({ ...r, __attrs: `class="row-link run-row ${r.dir === pick ? "selected" : ""}" data-dir="${esc(r.dir)}"` })),
        "No certificates yet. Run the demo."
      )}</section><section class="panel run-detail" id="run-detail">${skeleton()}</section></div>`;
    $$(".run-row").forEach((tr) => tr.addEventListener("click", () => (location.hash = `#runs/${encodeURIComponent(tr.dataset.dir)}`)));
    if (pick === "__reviewer") return renderReviewer();
    if (pick) { state.selectedRun = pick; await renderRunDetail(pick); }
    else $("#run-detail").innerHTML = `<div class="panel-pad muted">Nothing selected.</div>`;
  }

  async function renderRunDetail(dir) {
    const d = await api(`/api/runs/${encodeURIComponent(dir)}`);
    const c = d.certificate || {};
    const ev = c.evidence || {};
    const sp = c.search_price || {};
    const sn = sp.selection_null || {};
    const ic = c.interpretability_cost || {};
    const signs = ev.sign_commitments;
    const cells = [
      ["delta", num(ev.delta)], ["95% CI", `${num(ev.ci_low)} to ${num(ev.ci_high)}`], ["confirmation p", num(ev.p_one_sided, 4)],
      ["units / groups", `${esc(ev.n_confirmation ?? "n/a")} / ${esc(ev.n_groups_confirmation ?? "n/a")}`],
    ];
    const tabs = ["certificate", "adversary", "gate", "markdown", "events", "files"];
    $("#run-detail").innerHTML = `
      <div class="detail-section"><div class="claim-kicker">${tierChip(c)} ${chip(c.claim_type || "", "muted")} ${c.expires ? chip(`expires ${c.expires}`, "muted") : ""}</div>
        <div class="run-claim">${esc(c.claim)}</div>
        <div class="evidence-grid">${cells.map(([k, v]) => `<div class="evidence-cell"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`).join("")}</div>
        <div class="btn-row mt16"><button class="btn btn-primary btn-sm" id="btn-verify">${icon("shield")} Verify signature & manifest</button><a class="btn btn-ghost btn-sm" href="/api/file/${encodeURIComponent(d.dir)}/certificate.json" download>${icon("download")} certificate.json</a></div>
        <div id="verify-out" class="verify-result"></div></div>
      ${c.status === "refused" && !sp.proposed ? `<div class="detail-section"><h4>Search price</h4><p class="muted tiny">No search was run: the question was refused before any hypothesis was proposed, so nothing was priced and no alpha was spent on a search.</p>${signs ? `<p class="tiny muted">Pre-committed signs: ${esc(signs.agree)} of ${esc(signs.n_signs)} held (one-sided binomial p=${num(signs.p_one_sided, 4)}).</p>` : ""}</div>` : `<div class="detail-section"><h4>Search price</h4><div class="receipt">
        <div class="receipt-row"><span>proposed candidates</span><b>${esc(sp.proposed ?? "n/a")}</b></div>
        <div class="receipt-row"><span>effective independent</span><b>${esc(sp.effective_independent ?? "n/a")}</b></div>
        <div class="receipt-row"><span>selection-null p (${esc(sn.n_permutations ?? 0)} perms, ${esc(sn.computed_on || "")})</span><b>${num(sn.p_value, 4)}</b></div>
        <div class="receipt-row"><span>null max-stat p95 / observed</span><b>${num(sn.max_stat_p95)} / ${num(sn.observed_max_stat)}</b></div>
        <div class="receipt-row"><span>sealed queries</span><b>${esc(sp.sealed_queries ?? "n/a")}</b></div>
        <div class="receipt-total receipt-row"><span>alpha spent / allocated</span><b>${num(sp.alpha_spent)} / ${num(sp.alpha_allocated)}</b></div>
      </div>
      ${ic.gap_vs_unconstrained !== undefined && ic.gap_vs_unconstrained !== null ? `<p class="tiny muted mt8">Interpretability bill: unconstrained model ${num(ic.unconstrained_cv_delta)} vs rule ${num(ic.rule_cv_delta)} (gap ${num(ic.gap_vs_unconstrained)}).</p>` : ""}
      ${signs ? `<p class="tiny muted mt8">Pre-committed signs: ${esc(signs.agree)} of ${esc(signs.n_signs)} held (one-sided binomial p=${num(signs.p_one_sided, 4)}).</p>` : ""}
      </div>`}
      <div class="detail-section"><div class="tabs">${tabs.map((t, i) => `<button class="tab ${i ? "" : "active"}" data-tab="${t}">${t}</button>`).join("")}</div><div id="tab-body" class="mt16"></div></div>`;
    const bodies = {
      certificate: () => code(JSON.stringify(c, null, 2)),
      adversary: () => table(
        [
          { label: "attack", render: (a) => `<span class="mono">${esc(a.attack)}</span>` },
          { label: "result", render: (a) => chip(a.result, a.result === "pass" ? "ok" : a.result === "fail" ? "bad" : "warn") },
          { label: "critical", render: (a) => (a.critical ? "yes" : "") },
          { label: "p", num: true, render: (a) => num(a.p_value, 4) },
          { label: "note", render: (a) => `<span class="muted tiny">${esc(a.note || "")}</span>` },
        ],
        d.adversary || [], "No adversary report (refused before the search)."
      ),
      gate: () => `<div class="check-list">${((d.gate || {}).checks || [])
        .map((k) => `<div class="check-item ${k.ok ? "check-ok" : "check-bad"}"><div class="check-icon">${icon(k.ok ? "check" : "x")}</div><div class="check-name">${esc(k.name)}</div><div class="check-detail">${esc(JSON.stringify(k.detail))}</div></div>`)
        .join("") || '<p class="muted">No gate decision (refused before the gate).</p>'}</div>`,
      markdown: () => `<div class="docs-body">${markdown(d.markdown)}</div>`,
      events: () => `<div class="console">${(d.events || []).map(logLine).join("")}</div>`,
      files: () => table(
        [{ label: "file", render: (f) => `<a class="dl" href="/api/file/${encodeURIComponent(d.dir)}/${encodeURIComponent(f.name)}">${esc(f.name)}</a>` }, { label: "bytes", num: true, key: "bytes" }, { label: "sha256", cls: "mono", render: (f) => esc(short(f.sha256, 20)) }],
        d.files || []
      ),
    };
    const show = (t) => { $$(".tab", $("#run-detail")).forEach((b) => b.classList.toggle("active", b.dataset.tab === t)); $("#tab-body").innerHTML = bodies[t](); };
    $$(".tab", $("#run-detail")).forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
    show("certificate");
    $("#btn-verify").addEventListener("click", async () => {
      try {
        const r = await api("/api/verify", { method: "POST", body: JSON.stringify({ run_id: d.dir }) });
        $("#verify-out").innerHTML = verifyBadge(r);
      } catch (err) { toast(err.message, "bad"); }
    });
  }

  function verifyBadge(r) {
    const cert = r.certificate || {};
    const man = r.manifest;
    const ok = cert.valid && (!man || man.ok);
    return `<div class="verdict ${ok ? "ok" : "bad"}"><div class="verdict-icon">${icon(ok ? "check" : "x")}</div><div class="verdict-main"><div class="verdict-title">${ok ? "Verifies" : "Does not verify"}</div><div class="verdict-text">hash + signature: ${cert.valid ? "ok" : esc((cert.reasons || []).join(", ") || "failed")}${
      man ? ` · manifest: ${man.ok ? `${man.checked} files ok` : `${man.mismatches.length} mismatch(es): ${esc(man.mismatches.map((m) => m.file).join(", "))}`}` : ""
    }${r.contract_registered === false ? " · contract not in this registry" : ""}</div></div></div>`;
  }

  function renderReviewer() {
    $("#run-detail").innerHTML = `<div class="detail-section"><h4>Reviewer view</h4><p class="muted tiny">Paste a certificate JSON. Its hash and signature are checked against this installation's key; nothing is trusted from the issuer.</p>
      <textarea id="rev-text" class="code" style="width:100%;min-height:260px" spellcheck="false"></textarea>
      <div class="btn-row mt8"><button class="btn btn-primary btn-sm" id="rev-go">${icon("shield")} Check</button></div><div id="rev-out" class="verify-result"></div>
      <p class="tiny muted mt16">To reproduce from the certificate alone: <code class="code-inline">autoresearch reproduce --run &lt;run_id&gt; --fresh</code></p></div>`;
    $("#rev-go").addEventListener("click", async () => {
      let cert;
      try { cert = JSON.parse($("#rev-text").value); } catch (err) { toast("Not valid JSON", "bad"); return; }
      try { $("#rev-out").innerHTML = verifyBadge(await api("/api/verify", { method: "POST", body: JSON.stringify({ certificate: cert }) })); }
      catch (err) { toast(err.message, "bad"); }
    });
  }

  // ---------------------------------------------------------------- atlas
  async function renderAtlas() {
    const a = await api("/api/atlas");
    const ds = a.catalog?.datasets || [];
    const mods = a.modalities?.modalities || [];
    const opps = a.opportunities?.opportunities || [];
    view().innerHTML = head("application domain", "Personalised immunotherapy", a.catalog?.disclaimer || "")
      + `<div class="section"><div class="section-head"><h2>Ranked opportunities</h2><span class="hint">${esc(a.opportunities?.note || "")}</span></div><div class="stack">${opps
        .map((o) => `<div class="opp mt8"><div class="opp-rank">#${esc(o.rank)}</div><div><h3>${esc(o.name)}</h3><p>${esc(o.why || o.market || "")}</p><p class="tiny">${esc(o.unit_economics || "")}</p></div><div class="opp-score">${num(o.risk_adjusted_score, 2)}</div></div>`)
        .join("")}</div></div>`
      + `<div class="section">${panel("Datasets", table(
        [
          { label: "dataset", render: (d) => (d.url && /^https?:/.test(d.url) ? `<a href="${esc(d.url)}" target="_blank" rel="noopener noreferrer">${esc(d.name)}</a>` : esc(d.name)) },
          { label: "disease", key: "disease" },
          { label: "labels", render: (d) => chip(d.label_status, String(d.label_status).startsWith("label") ? "ok" : "muted") },
          { label: "access", key: "access_tier" },
          { label: "role", key: "role" },
          { label: "contract readiness", key: "contract_readiness" },
        ], ds), `${ds.length} sources`)}</div>`
      + `<div class="section">${panel("Modalities and unit economics", table(
        [
          { label: "modality", key: "modality" }, { label: "asset", key: "asset" }, { label: "status", key: "status" },
          { label: "ORR", num: true, render: (m) => (m.orr_pct === null || m.orr_pct === undefined ? "n/a" : `${esc(m.orr_pct)}%`) },
          { label: "turnaround (d)", num: true, key: "turnaround_days" },
          { label: "gross margin", num: true, render: (m) => (m.gross_margin === null || m.gross_margin === undefined ? "n/a" : pct(m.gross_margin, 1)) },
        ], mods), a.modalities?.as_of || "")}</div>`;
  }

  // ---------------------------------------------------------------- contract
  async function renderContract() {
    const c = await api("/api/contract");
    const v = c.validation || {};
    view().innerHTML = head("the only place domain knowledge lives", "Dataset contract", `Contract hash ${c.contract_hash}. Changing it invalidates every claim made under the old hash.`)
      + `<div class="split"><div class="stack">
        ${panel("Decision", kv(Object.entries(c.decision || {}).map(([k, val]) => [k, esc(val)])))}
        <div class="mt16">${panel("Features (directions are pre-committed)", table([{ label: "feature", key: "name" }, { label: "direction", key: "direction" }, { label: "transform", key: "transform" }], c.features || []))}</div>
        <div class="mt16">${panel("Complexity budget", kv(Object.entries(c.budget || {}).filter(([, val]) => typeof val !== "object").map(([k, val]) => [k, esc(val)])), "computed before the search")}</div>
      </div><div class="stack">
        ${panel("Validation", `<div class="check-list">${(v.checks || []).map((k) => `<div class="check-item ${k.ok ? "check-ok" : "check-bad"}"><div class="check-icon">${icon(k.ok ? "check" : "x")}</div><div class="check-name">${esc(k.name)}</div><div class="check-detail">${esc(JSON.stringify(k.detail ?? ""))}</div></div>`).join("")}</div>${(v.warnings || []).map((w) => `<p class="tiny muted">${esc(w)}</p>`).join("")}`, v.ok ? "valid" : "invalid")}
        <div class="mt16">${panel("Canonical YAML", code(c.yaml))}</div>
      </div></div>`;
  }

  // ---------------------------------------------------------------- calibration
  async function renderCalibration() {
    const { table: t, feedback_leak: leak } = await api("/api/calibration");
    if (!t || !t.rows) { view().innerHTML = head("operating characteristics", "No table yet", "Run `autoresearch calibrate`."); return; }
    const adv = t.adversary || {};
    const rows = t.rows;
    const bars = rows.map((r) => `<div class="stat-line mt8"><span class="mono">effect ${num(r.effect, 2)}</span><div class="bar-track" style="flex:1"><div class="bar-fill" style="width:${Math.round(100 * (r.detect || 0))}%"></div></div><span class="mono">${pct(r.detect)}</span></div>`).join("");
    view().innerHTML = head("this table is the product", "Operating characteristics", "Effects of known size planted repeatedly; pure nulls run to measure false positives. Without this, 'we promoted nothing' is indistinguishable from a broken gate.")
      + `<div class="split">${panel("Detection rate by planted effect", bars, `${t.parameters?.reps_per_effect ?? "?"} replicates per effect`)}${panel("Gate and adversary", kv([
        ["false positives", `${pct(t.fpr_decision)} of pure-null replicates promoted`],
        ["adversary false alarm", adv.honest_promotions ? `${adv.killed}/${adv.honest_promotions} honest promotions killed (95% CI ${pct(adv.false_alarm_ci_low)}-${pct(adv.false_alarm_ci_high)})` : "not measured"],
        ["generated", esc(t.generated_at)], ["substrate", esc(t.domain)],
      ]))}</div>`
      + `<div class="section">${panel("Table", table(
        [
          { label: "true effect", num: true, render: (r) => num(r.effect) }, { label: "n", num: true, key: "n" },
          { label: "promoted", num: true, render: (r) => pct(r.detect) }, { label: "95% CI", num: true, render: (r) => `${pct(r.detect_ci_low)}-${pct(r.detect_ci_high)}` },
          { label: "confirmed", num: true, render: (r) => pct(r.detect_confirmed) }, { label: "without adversary", num: true, render: (r) => pct(r.detect_without_adversary) },
          { label: "adversary kills", num: true, render: (r) => esc(r.adversary_kills ?? 0) }, { label: "FPR", num: true, render: (r) => pct(r.fpr_decision) },
        ], rows))}</div>`
      + (leak ? `<div class="section">${panel("Feedback-leak test (spec §5)", table(
        [{ label: "channel", key: "ch" }, { label: "replicates", num: true, key: "n" }, { label: "false positives", num: true, key: "false_positives" }, { label: "FPR", num: true, render: (r) => pct(r.fpr) }, { label: "95% CI", num: true, render: (r) => `${pct(r.fpr_ci_low)}-${pct(r.fpr_ci_high)}` }],
        Object.entries(leak.summary || {}).map(([ch, s]) => ({ ch: ch === "leaky" ? "leaky (scores on gate data)" : "guarded (categorical only)", ...s }))
      ), "same proposer, pure-null data")}</div>` : "")
      + `<div class="section">${panel("Limitations", `<ul class="muted">${(t.limitations || []).map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`)}</div>`;
  }

  // ---------------------------------------------------------------- live run
  function logLine(e) {
    const s = String(e.status || "");
    const tone = /ok|issued|registered|committed|selected|pass|confirmed|answerable/i.test(s) ? "ok" : /refus|fail|error|exhaust|bad/i.test(s) ? "bad" : /warn|not_affirmed|exploratory|provisional/i.test(s) ? "warn" : "info";
    const ts = String(e.ts || "").slice(11, 19);
    return `<div class="log-line ${tone}"><span class="seq">${esc(e.seq)}</span><span class="ts">${esc(ts)}</span><span class="stage-name">${esc(e.stage)}·${esc(s)}</span><span class="msg">${esc(e.message)}</span></div>`;
  }

  async function renderLive() {
    const params = [["patients", 300, "Patients in the sandbox substrate"], ["confirmation_groups", 100, "Sealed groups"], ["candidates", 72, "Candidate budget"], ["generations", 2, "Proposer generations"], ["permutations", 19, "Selection-null permutations"], ["seed", 20261114, "Generator seed"]];
    view().innerHTML = head("watch a machine decline to claim things", "Live run", "Builds a sandboxed synthetic substrate, seals its confirmation set, and streams every proposal, keep, rejection and refusal.")
      + `<div class="live-layout"><section class="panel"><div class="panel-head"><h3>Run</h3></div><div class="panel-body panel-pad"><form id="live-form" class="stack">${params
        .map(([k, v, l]) => `<div class="field"><label for="lf-${k}">${esc(l)}</label><input type="number" id="lf-${k}" name="${k}" value="${v}"></div>`)
        .join("")}<button class="btn btn-primary" type="submit">${icon("play")} Start</button></form><div id="live-result" class="mt16"></div></div></section>
        <section class="panel"><div class="panel-head"><h3 class="live-status"><span class="pulse" id="live-pulse"></span><span id="live-label">idle</span></h3><span class="meta" id="live-count"></span></div><div class="console" id="live-console"><div class="live-empty">No run yet.</div></div></section></div>`;
    $("#live-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const body = Object.fromEntries(Object.entries(Object.fromEntries(new FormData(e.target))).map(([k, v]) => [k, Number(v)]));
      try {
        const r = await api("/api/live/start", { method: "POST", body: JSON.stringify(body) });
        if (r.busy) toast("A live run is already in progress; attaching to it.");
        attachLive(r.job.live_id);
      } catch (err) { toast(err.message, "bad"); }
    });
    const { active } = await api("/api/live/current");
    if (active?.live_id) attachLive(active.live_id);
  }

  function attachLive(liveId) {
    if (state.source) state.source.close();
    const box = $("#live-console");
    if (!box) return;
    box.innerHTML = "";
    let count = 0;
    $("#live-pulse").className = "pulse on";
    $("#live-label").textContent = `running · ${liveId}`;
    const src = new EventSource(`/api/live/${encodeURIComponent(liveId)}/stream`);
    state.source = src;
    src.addEventListener("stage", (msg) => {
      const e = JSON.parse(msg.data);
      const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
      box.insertAdjacentHTML("beforeend", logLine(e));
      if (atBottom) box.scrollTop = box.scrollHeight;
      $("#live-count").textContent = `${++count} events`;
    });
    src.addEventListener("done", (msg) => {
      src.close();
      const job = JSON.parse(msg.data || "{}");
      const ok = job.status === "done";
      $("#live-pulse").className = `pulse ${ok ? "" : "bad"}`;
      $("#live-label").textContent = `${job.status || "finished"} · ${liveId}`;
      const res = job.result;
      $("#live-result").innerHTML = res
        ? `<div class="verdict ${tierTone(res)}"><div class="verdict-main"><div class="verdict-kicker">${esc(res.run_id)}</div><div class="verdict-title">${esc(res.status === "refused" ? "Refused" : `${res.tier} / ${res.verdict}`)}</div><div class="verdict-text">${esc(res.claim)}</div></div></div>`
        : `<p class="muted">${esc(job.error || "No result.")}</p>`;
    });
    src.onerror = () => { if (src.readyState === EventSource.CLOSED) $("#live-pulse").className = "pulse"; };
  }

  // ---------------------------------------------------------------- docs
  async function renderDocs() {
    const res = await fetch("/README.md");
    const md = res.ok ? await res.text() : "README not found.";
    view().innerHTML = `<div class="docs-body">${markdown(md)}</div>`;
  }

  // ---------------------------------------------------------------- boot
  $$(".nav-item").forEach((b) => b.addEventListener("click", () => (location.hash = `#${b.dataset.page}`)));
  $("#btn-refresh").addEventListener("click", () => { refreshStatus(); route(); toast("Refreshed"); });
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const keys = Object.keys(PAGES);
    const i = Number(e.key) - 1;
    if (i >= 0 && i < keys.length) location.hash = `#${keys[i]}`;
  });
  window.addEventListener("hashchange", () => { if (state.source && !location.hash.startsWith("#live")) { state.source.close(); state.source = null; } route(); });
  refreshStatus().then(route);
})();
