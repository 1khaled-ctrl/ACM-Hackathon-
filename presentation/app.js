/* ============================================================
   GoPilot x GoTrain — presentation prototype interactions
   Everything here is simulated demo data. No backend calls.
   ============================================================ */

(function () {
  'use strict';

  // ---------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  };
  const nowTime = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  // ===========================================================
  // 1. SCREEN 1 — GoPilot chat demo
  // ===========================================================
  const chatIntro = $('#chatIntro');
  const chatThread = $('#chatThread');
  const chatInputPlaceholder = $('#chatInputPlaceholder');
  const startDemoBtn = $('#startDemoBtn');

  let demoStep = 0; // 0 = not started

  function addUserMsg(text) {
    const m = el('div', 'msg msg-user', text);
    chatThread.appendChild(m);
    chatThread.scrollTop = chatThread.scrollHeight;
  }

  function addBotMsg(html) {
    const m = el('div', 'msg msg-bot', html);
    chatThread.appendChild(m);
    chatThread.scrollTop = chatThread.scrollHeight;
    return m;
  }

  function startChatDemo() {
    if (demoStep > 0) return;
    demoStep = 1;
    chatIntro.style.display = 'none';
    chatThread.style.display = 'flex';

    addUserMsg("I want to detect solar panels in my satellite images, but the existing model isn't performing well in my region.");

    setTimeout(() => {
      const bot = addBotMsg(
        "Your model may need adaptation to your region's imagery. Would you like GoTrain to diagnose your dataset and train a customized model?"
      );
      const btn = el('button', 'train-cta-btn', '&#129514; Train Custom Model');
      btn.id = 'trainCustomBtn';
      bot.appendChild(document.createElement('br'));
      bot.appendChild(btn);
      btn.addEventListener('click', openSetupPanel);
      chatInputPlaceholder.textContent = 'Click "Train Custom Model" to continue the demo';
    }, 700);
  }

  function openSetupPanel() {
    $('#trainCustomBtn')?.remove();
    const panel = el('div', 'setup-panel');
    panel.innerHTML = `
      <h4>GOTRAIN &middot; TRAINING SETUP</h4>
      <div class="setup-row"><span>Dataset</span><b>solar-panels-region-7 (customer upload)</b></div>
      <div class="setup-row"><span>Task</span><b>Semantic segmentation</b></div>
      <div class="setup-row"><span>Base model</span><b>pv4ger-seg v2 (incompatible region)</b></div>
      <div class="setup-row"><span>Diagnostics</span><b>GeoGuard + DataIQ (auto)</b></div>
    `;
    const startBtn = el('button', 'start-training-btn', 'Start Training &#8594;');
    panel.appendChild(startBtn);
    chatThread.appendChild(panel);
    chatThread.scrollTop = chatThread.scrollHeight;
    chatInputPlaceholder.textContent = 'Click "Start Training" to open Model Studio';
    startBtn.addEventListener('click', () => openOverlay('model-studio'));
  }

  startDemoBtn.addEventListener('click', startChatDemo);
  $('#chatSendBtn').addEventListener('click', startChatDemo);

  function resetChatDemo() {
    demoStep = 0;
    chatThread.innerHTML = '';
    chatThread.style.display = 'none';
    chatIntro.style.display = '';
    chatInputPlaceholder.textContent = 'Click the highlighted action to continue the demo';
  }
  $('#restartDemoBtn').addEventListener('click', resetChatDemo);

  // ===========================================================
  // 2. Overlay open/close (Model Studio / Engineer Console)
  // ===========================================================
  function openOverlay(id) {
    document.getElementById(id).classList.add('open');
    document.body.style.overflow = 'hidden';
    if (id === 'model-studio' && !studioBuilt) buildStudio();
    if (id === 'engineer-console' && !engBuilt) buildEngineerConsole();
  }
  function closeOverlay(id) {
    document.getElementById(id).classList.remove('open');
    document.body.style.overflow = '';
  }
  $$('[data-close]').forEach((btn) => btn.addEventListener('click', () => closeOverlay(btn.dataset.close)));
  $('#navModelStudio').addEventListener('click', () => openOverlay('model-studio'));
  $('#navEngConsole').addEventListener('click', () => openOverlay('engineer-console'));
  $('#useInGoPilotBtn').addEventListener('click', () => closeOverlay('model-studio'));
  $('#navArchitecture').addEventListener('click', () => document.getElementById('architecture').scrollIntoView({ behavior: 'smooth' }));
  $('#navBusiness').addEventListener('click', () => document.getElementById('business').scrollIntoView({ behavior: 'smooth' }));
  $('#scrollCue').addEventListener('click', () => document.getElementById('story-top').scrollIntoView({ behavior: 'smooth' }));

  // ===========================================================
  // 3. MODEL STUDIO — mirrors GoTrain NEXUS panels
  //    (GeoGuard/DataIQ diagnostics, training chart, run
  //    comparison, StressLab, lineage, model passport)
  // ===========================================================
  let studioBuilt = false;

  function sparkline(points, color, h = 120) {
    const w = 400;
    const max = Math.max(...points);
    const min = Math.min(...points);
    const range = max - min || 1;
    const stepX = w / (points.length - 1);
    const path = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${(i * stepX).toFixed(1)} ${(h - ((p - min) / range) * (h - 14) - 7).toFixed(1)}`)
      .join(' ');
    return `<svg class="chart-mini" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <polyline points="${points.map((p, i) => `${(i * stepX).toFixed(1)},${(h - ((p - min) / range) * (h - 14) - 7).toFixed(1)}`).join(' ')}"
        fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/>
    </svg>`;
  }

  function buildStudio() {
    studioBuilt = true;
    const body = $('#studioBody');

    const trainLoss = [1.21, 0.98, 0.81, 0.69, 0.58, 0.49, 0.43, 0.38, 0.34, 0.31, 0.29, 0.27];
    const valMetric = [0.41, 0.52, 0.6, 0.66, 0.71, 0.74, 0.77, 0.79, 0.8, 0.81, 0.815, 0.818];

    body.innerHTML = `
      <div class="studio-grid">

        <!-- Dataset diagnostics: GeoGuard -->
        <div class="g-6 nx-panel">
          <div class="nx-panel-title">GeoGuard &middot; dataset diagnostics <span class="nx-badge nx-badge--amber">TRAIN split, 300 samples</span></div>
          <div class="panel-pad">
            <div class="diag-row"><span class="diag-dot diag-dot--ok"></span><span class="diag-text">Modality detected: <b>optical, RGB+NIR, 4 bands</b></span></div>
            <div class="diag-row"><span class="diag-dot diag-dot--warn"></span><span class="diag-text"><b>Band statistics shifted</b> vs. base model's training distribution (region 7 has higher NIR reflectance)</span></div>
            <div class="diag-row"><span class="diag-dot diag-dot--warn"></span><span class="diag-text">Class balance: solar panel pixels are <b>3.2&times; sparser</b> than base training set</span></div>
            <div class="diag-row"><span class="diag-dot diag-dot--ok"></span><span class="diag-text">No corrupt tiles or duplicate scenes found</span></div>
            <div class="explain-box"><b>What GeoGuard does:</b> scans up to 300 training samples before any download or run &mdash; band stats, SAR/optical modality, and class balance &mdash; so the agent knows what it's working with before it spends compute.</div>
          </div>
        </div>

        <!-- DataIQ -->
        <div class="g-6 nx-panel">
          <div class="nx-panel-title">DataIQ &middot; structural scan <span class="nx-badge nx-badge--green">PASSED</span></div>
          <div class="panel-pad">
            <div class="diag-row"><span class="diag-dot diag-dot--ok"></span><span class="diag-text"><b>0</b> corrupted tiles, <b>0</b> duplicate scenes (auto-quarantine ready)</span></div>
            <div class="diag-row"><span class="diag-dot diag-dot--ok"></span><span class="diag-text">Label kind: <b>segmentation</b> &middot; class count read from dataset registry</span></div>
            <div class="diag-row"><span class="diag-dot diag-dot--warn"></span><span class="diag-text">Loss-scan recommendation: <b>CE + Dice + Focal</b> (sparse positive class)</span></div>
            <div class="mini-label" style="margin-top:10px;">Preflight gate</div>
            <div class="diag-row" style="border-bottom:none; padding-top:4px;"><span class="diag-dot diag-dot--ok"></span><span class="diag-text">Config validated &middot; model compatibility confirmed for <b>pv4ger-seg</b> architecture family</span></div>
          </div>
        </div>

        <!-- Training progress -->
        <div class="g-8 nx-panel">
          <div class="nx-panel-title">Training progress &middot; run #4 (current best) <span class="nx-badge nx-badge--blue">RUNNING_AUTONOMOUSLY</span></div>
          <div class="panel-pad">
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">
              <div>
                <div class="mini-label">Train loss</div>
                ${sparkline(trainLoss, '#e3634f')}
              </div>
              <div>
                <div class="mini-label">Validation mIoU</div>
                ${sparkline(valMetric, '#4fc98a')}
              </div>
            </div>
            <div class="row-between" style="margin-top:10px; font-size:11px; color:var(--nx-text-faint);">
              <span>epoch 12 / 16</span><span>elapsed 6m 40s</span><span>ETA ~2m 10s</span>
            </div>
          </div>
        </div>

        <!-- Agent decision timeline -->
        <div class="g-4 nx-panel">
          <div class="nx-panel-title">Agent decision timeline</div>
          <div class="panel-pad">
            <div class="timeline">
              <div class="tl-item"><div class="tl-time">14:02:08</div><div class="tl-text">Ran <b>GeoGuard</b> &mdash; flagged band-stat shift &amp; class sparsity</div></div>
              <div class="tl-item"><div class="tl-time">14:03:41</div><div class="tl-text">Hypothesis: <b>loss re-weighting</b> will recover recall on sparse class</div></div>
              <div class="tl-item"><div class="tl-time">14:07:55</div><div class="tl-text">Run #2 &mdash; BCE baseline &mdash; <b>underfitting</b> (outcome: rejected)</div></div>
              <div class="tl-item"><div class="tl-time">14:14:20</div><div class="tl-text">Run #3 &mdash; CE+Dice, class weights &mdash; improved, still noisy edges</div></div>
              <div class="tl-item tl-best"><div class="tl-time">14:21:03</div><div class="tl-text">Run #4 &mdash; <b>CE+Dice+Focal</b> &mdash; best so far, <b>0.818 mIoU</b></div></div>
            </div>
          </div>
        </div>

        <!-- Run comparison -->
        <div class="g-12 nx-panel">
          <div class="nx-panel-title">Experiment comparison <span class="nx-badge">4 runs</span></div>
          <div class="panel-pad" style="overflow-x:auto;">
            <table class="run-table">
              <thead><tr><th>Run</th><th>Loss fn</th><th>Backbone</th><th>Val mIoU</th><th>Outcome</th><th>Hypothesis graded</th></tr></thead>
              <tbody>
                <tr><td>#1</td><td>CE</td><td>pv4ger-seg base</td><td>0.512</td><td>baseline</td><td>&mdash;</td></tr>
                <tr><td>#2</td><td>BCE</td><td>pv4ger-seg base</td><td>0.558</td><td>underfitting</td><td>rejected</td></tr>
                <tr><td>#3</td><td>CE + Dice</td><td>pv4ger-seg base</td><td>0.741</td><td>noisy edges</td><td>partially confirmed</td></tr>
                <tr class="row-best"><td class="run-best">#4</td><td>CE + Dice + Focal</td><td>pv4ger-seg base</td><td class="run-best">0.818</td><td class="run-best">best</td><td class="run-best">confirmed</td></tr>
              </tbody>
            </table>
            <div class="explain-box"><b>What the agent changed and why:</b> the base loss function treated every pixel equally, but solar-panel pixels are rare in region 7's imagery. Adding Dice and Focal terms weights the loss toward the sparse positive class instead of the easy negative background &mdash; which is what moved mIoU from 0.51 to 0.82 across runs #1&ndash;#4.</div>
          </div>
        </div>

        <!-- StressLab -->
        <div class="g-6 nx-panel">
          <div class="nx-panel-title">StressLab &middot; robustness <span class="nx-badge nx-badge--green">PASSED</span></div>
          <div class="panel-pad">
            <div class="diag-row"><span class="diag-dot diag-dot--ok"></span><span class="diag-text">Band dropout (1 of 4 bands missing): <b>-4.1%</b> mIoU</span></div>
            <div class="diag-row"><span class="diag-dot diag-dot--ok"></span><span class="diag-text">Gaussian sensor noise: <b>-2.6%</b> mIoU</span></div>
            <div class="diag-row"><span class="diag-dot diag-dot--warn"></span><span class="diag-text">Geometric perturbation (rotation/scale): <b>-9.8%</b> mIoU</span></div>
            <div class="explain-box"><b>What StressLab does:</b> re-evaluates the saved checkpoint on the validation split under synthetic band dropout, noise, and geometric perturbation &mdash; so robustness is checked beyond a single clean-data accuracy number.</div>
          </div>
        </div>

        <!-- Model versions & passport -->
        <div class="g-6 nx-panel">
          <div class="nx-panel-title">Model versions &amp; passport</div>
          <div class="panel-pad">
            <div class="version-chip-row" id="studioChipRow" style="margin-bottom:12px;">
              <div class="version-chip"><span class="vtag">v1</span><span>baseline</span></div>
              <div class="version-chip"><span class="vtag">v2</span><span>approved checkpoint</span></div>
              <div class="version-chip active"><span class="vtag">v3</span><span>active &middot; run #4</span></div>
            </div>
            <dl class="passport-grid">
              <dt>Architecture</dt><dd>pv4ger-seg (U-Net, ResNet34)</dd>
              <dt>Loss</dt><dd>CE + Dice + Focal</dd>
              <dt>Val mIoU</dt><dd>0.818</dd>
              <dt>StressLab score</dt><dd>passed (3/3 checks)</dd>
              <dt>Trained on</dt><dd>solar-panels-region-7</dd>
              <dt>Status</dt><dd>active &middot; next change (ResNet50) awaiting engineer approval</dd>
            </dl>
          </div>
        </div>

      </div>
    `;
  }

  // ===========================================================
  // 4. ENGINEER CONSOLE — pause/resume/manual/approve/reject/
  //    rollback + audit history. Reject & Rollback are interactive.
  // ===========================================================
  let engBuilt = false;
  let engState = 'RUNNING_AUTONOMOUSLY';

  const ENG_LOG_SEED = [
    { t: '14:02:08', text: 'control.json &rarr; RUNNING_AUTONOMOUSLY (session start)' },
    { t: '14:03:41', text: 'agent recorded hypothesis for run #2 (loss re-weighting)' },
    { t: '14:14:20', text: 'run #3 completed &middot; recorded to ledger.jsonl' },
    { t: '14:21:03', text: 'run #4 completed &middot; best_run_id updated' },
    { t: '14:22:10', text: 'agent proposed backbone change &middot; awaiting engineer approval' },
  ];

  function engStateLabel(s) {
    return {
      RUNNING_AUTONOMOUSLY: 'Running autonomously',
      PAUSE_REQUESTED: 'Pause requested',
      PAUSED_FOR_ENGINEER: 'Paused for engineer',
      MANUAL_CONTROL: 'Manual control (engineer)',
      RESUME_REQUESTED: 'Resume requested',
      ROLLBACK_REQUESTED: 'Rollback requested',
      ROLLBACK_VERIFYING: 'Verifying rollback integrity',
    }[s] || s;
  }

  function buildEngineerConsole() {
    engBuilt = true;
    const body = $('#engBody');
    body.innerHTML = `
      <div class="eng-grid">

        <div class="g-12" id="engBanner"></div>

        <div class="g-4 nx-panel">
          <div class="nx-panel-title">Agent control</div>
          <div class="panel-pad eng-control-row" style="flex-direction:column;">
            <div class="eng-control-row">
              <button class="eng-btn" id="btnPause"><span class="eng-ic">&#9208;&#65039;</span>Pause</button>
              <button class="eng-btn" id="btnResume"><span class="eng-ic">&#9654;&#65039;</span>Resume</button>
            </div>
            <div class="eng-control-row">
              <button class="eng-btn" id="btnManual"><span class="eng-ic">&#128272;</span>Manual takeover</button>
            </div>
          </div>
        </div>

        <div class="g-8 nx-panel">
          <div class="nx-panel-title">Pending approval &middot; proposed architecture change</div>
          <div class="panel-pad">
            <div class="proposal-card">
              <h4>Backbone change proposed by agent</h4>
              <p>Agent requests switching backbone from <b>ResNet34</b> &rarr; <b>ResNet50</b> for run #5, citing potential +2&ndash;3% mIoU from increased capacity. This exceeds the agent's autonomous-change permissions and requires engineer sign-off.</p>
              <div class="eng-control-row">
                <button class="eng-btn eng-btn--approve" id="btnApprove"><span class="eng-ic">&#9989;</span>Approve</button>
                <button class="eng-btn eng-btn--reject" id="btnReject"><span class="eng-ic">&#10060;</span>Reject</button>
              </div>
              <div class="feedback-msg" id="rejectMsg">
                <span class="step">&#10060; Change rejected.</span>
                "Change rejected. Approved checkpoint preserved. Agent continuing within existing permissions."
              </div>
            </div>
          </div>
        </div>

        <div class="g-6 nx-panel">
          <div class="nx-panel-title">Rollback to approved checkpoint</div>
          <div class="panel-pad">
            <div class="version-chip-row" id="rollbackChipRow" style="margin-bottom:12px;">
              <div class="version-chip active"><span class="vtag">v3</span><span>active &middot; run #4</span></div>
              <div class="version-chip"><span class="vtag">v2</span><span>approved checkpoint</span></div>
              <div class="version-chip"><span class="vtag">v1</span><span>baseline</span></div>
            </div>
            <button class="eng-btn eng-btn--rollback" id="btnRollback" style="width:100%;"><span class="eng-ic">&#8617;&#65039;</span>Rollback to v2 (approved)</button>
            <div class="feedback-msg" id="rollbackMsg"></div>
          </div>
        </div>

        <div class="g-6 nx-panel">
          <div class="nx-panel-title">Audit history <span class="nx-badge">events.jsonl</span></div>
          <div class="panel-pad eng-log" id="engLog"></div>
        </div>

      </div>
    `;

    renderEngBanner();
    renderEngLog();

    $('#btnPause').addEventListener('click', () => transitionEng('PAUSE_REQUESTED', 'PAUSED_FOR_ENGINEER', 'engineer requested pause'));
    $('#btnResume').addEventListener('click', () => transitionEng('RESUME_REQUESTED', 'RUNNING_AUTONOMOUSLY', 'engineer requested resume'));
    $('#btnManual').addEventListener('click', () => transitionEng('MANUAL_CONTROL', 'MANUAL_CONTROL', 'engineer took manual control'));
    $('#btnApprove').addEventListener('click', onApprove);
    $('#btnReject').addEventListener('click', onReject);
    $('#btnRollback').addEventListener('click', onRollback);
  }

  function renderEngBanner() {
    $('#engBanner').innerHTML = `
      <div class="eng-state-banner">
        <span class="state-dot"></span>
        Agent state: <b>${engStateLabel(engState)}</b>
        <span style="margin-left:auto; color:var(--nx-text-faint);">session m-pv4ger-seg &middot; started 14:02:08</span>
      </div>
    `;
  }

  function appendEngLog(text) {
    const log = $('#engLog');
    const line = el('div', 'eng-log-line flash', `<span class="eng-log-time">${nowTime()}</span>${text}`);
    log.insertBefore(line, log.firstChild);
  }

  function renderEngLog() {
    const log = $('#engLog');
    log.innerHTML = '';
    ENG_LOG_SEED.slice().reverse().forEach((e) => {
      log.appendChild(el('div', 'eng-log-line', `<span class="eng-log-time">${e.t}</span>${e.text}`));
    });
  }

  function transitionEng(transitional, settled, logText) {
    engState = transitional;
    renderEngBanner();
    appendEngLog(logText + ` &rarr; <b>${transitional}</b>`);
    setTimeout(() => {
      engState = settled;
      renderEngBanner();
      if (settled !== transitional) appendEngLog(`state settled &rarr; <b>${settled}</b>`);
    }, 650);
  }

  function onApprove() {
    appendEngLog('engineer <b>approved</b> backbone change (ResNet34 &rarr; ResNet50)');
    $('#btnApprove').disabled = true;
    $('#btnReject').disabled = true;
    $('#btnApprove').style.opacity = '0.5';
    $('#btnReject').style.opacity = '0.5';
  }

  function onReject() {
    const msg = $('#rejectMsg');
    msg.classList.add('show');
    $('#btnApprove').disabled = true;
    $('#btnReject').disabled = true;
    $('#btnApprove').style.opacity = '0.5';
    $('#btnReject').style.opacity = '0.5';
    appendEngLog('engineer <b>rejected</b> backbone change &middot; approved checkpoint v2 preserved');
  }

  function onRollback() {
    const msg = $('#rollbackMsg');
    msg.className = 'feedback-msg rollback show';
    msg.innerHTML = `<span class="step">&#9203; verifying checkpoint integrity&hellip;</span>`;
    appendEngLog('rollback requested &rarr; target v2 &middot; state &rarr; <b>ROLLBACK_REQUESTED</b>');
    engState = 'ROLLBACK_REQUESTED';
    renderEngBanner();

    setTimeout(() => {
      msg.innerHTML = `<span class="step">&#9203; verifying checkpoint integrity&hellip; OK</span><span class="step">&#128190; loading model weights v2&hellip;</span>`;
      engState = 'ROLLBACK_VERIFYING';
      renderEngBanner();
      appendEngLog('checkpoint hash verified &middot; loading v2 weights &middot; state &rarr; <b>ROLLBACK_VERIFYING</b>');
    }, 700);

    setTimeout(() => {
      msg.innerHTML = `<span class="step">&#9203; verifying checkpoint integrity&hellip; OK</span><span class="step">&#128190; loading model weights v2&hellip; OK</span><b>Rollback complete. v2 restored and active. Agent resumed within approved permissions.</b>`;
      engState = 'RUNNING_AUTONOMOUSLY';
      renderEngBanner();
      appendEngLog('rollback complete &middot; <b>v2 active</b> &middot; state &rarr; <b>RUNNING_AUTONOMOUSLY</b>');
      markV2Active();
    }, 1700);
  }

  function markV2Active() {
    const rollbackRow = $('#rollbackChipRow');
    if (rollbackRow) {
      rollbackRow.innerHTML = `
        <div class="version-chip"><span class="vtag">v3</span><span>superseded &middot; run #4</span></div>
        <div class="version-chip active"><span class="vtag">v2</span><span>active &middot; restored</span></div>
        <div class="version-chip"><span class="vtag">v1</span><span>baseline</span></div>
      `;
    }
    const studioRow = $('#studioChipRow');
    if (studioRow) {
      studioRow.innerHTML = `
        <div class="version-chip"><span class="vtag">v1</span><span>baseline</span></div>
        <div class="version-chip active"><span class="vtag">v2</span><span>active &middot; restored</span></div>
        <div class="version-chip"><span class="vtag">v3</span><span>superseded &middot; run #4</span></div>
      `;
    }
    const btn = $('#btnRollback');
    if (btn) { btn.disabled = true; btn.style.opacity = '0.5'; btn.innerHTML = '<span class="eng-ic">&#9989;</span>v2 is active'; }
  }

  // ===========================================================
  // 5. ARCHITECTURE DIAGRAM — adapted from GoTrain Architecture.html
  //    Clickable nodes reveal plain-language explanations.
  // ===========================================================
  const ARCH_NODES = [
    {
      id: 'gopilot', x: 20, y: 40, w: 230, h: 90, variant: 'new', title: 'GoPilot chat',
      lines: ['Customer-facing product', 'Detects "model isn\'t working" in chat'],
      explain: 'GoPilot is the existing RASID product customers already use to ask questions about satellite imagery. GoTrain plugs in only when GoPilot notices a model limitation it can\'t fix on its own.'
    },
    {
      id: 'mcp', x: 20, y: 150, w: 230, h: 70, variant: 'proposed', title: 'Proposed MCP integration', badge: 'PROPOSED',
      lines: ['GoPilot &harr; GoTrain bridge', 'Not yet built'],
      explain: 'A Model Context Protocol (MCP) integration is the proposed bridge that would let GoPilot hand off a diagnosis request to GoTrain and receive status updates back, without the two systems being tightly coupled.'
    },
    {
      id: 'strands', x: 300, y: 40, w: 230, h: 90, variant: 'highlight', title: 'Strands Agent', badge: 'CORE',
      lines: ['Claude Sonnet on Amazon Bedrock', 'Reasons over tool results only'],
      explain: 'The core agent loop. It runs Claude Sonnet via Amazon Bedrock, calling a fixed set of tools (diagnostics, training, comparison) and reasoning over their results &mdash; it never touches raw training data directly.'
    },
    {
      id: 'hooks', x: 300, y: 150, w: 230, h: 70, variant: 'new', title: 'Limits &amp; logging hooks', badge: 'NEW',
      lines: ['Checks control state first', 'Then run/time/call budgets'],
      explain: 'Before any tool call executes, a hook checks whether an engineer has paused or blocked the agent, then enforces budget limits (max runs, max tool calls, time budget) and logs the call for audit.'
    },
    {
      id: 'geoguard', x: 580, y: 40, w: 220, h: 70, variant: 'new', title: 'GeoGuard', badge: 'UPD',
      lines: ['Band stats, modality, class balance'],
      explain: 'Scans up to 300 training samples before any download &mdash; detects band statistics, SAR vs. optical modality, and class balance issues, so the agent understands the dataset before spending compute.'
    },
    {
      id: 'dataiq', x: 580, y: 120, w: 220, h: 70, variant: 'new', title: 'DataIQ', badge: 'UPD',
      lines: ['Structural scan: corruption, duplicates'],
      explain: 'Runs a structural scan for corrupted tiles and duplicate scenes with auto-quarantine, and recommends loss-function settings based on the dataset\'s actual label structure.'
    },
    {
      id: 'preflight', x: 580, y: 200, w: 220, h: 68, variant: 'new', title: 'Preflight &amp; compatibility gates', badge: 'NEW',
      lines: ['Validates config before any run'],
      explain: 'Validates the training configuration and confirms model/architecture compatibility before a run starts. A rejected config costs nothing against the run budget.'
    },
    {
      id: 'training', x: 580, y: 278, w: 220, h: 82, variant: 'highlight', title: 'Training &amp; experimentation',
      lines: ['Classification &middot; segmentation', 'Hypothesis logged per run'],
      explain: 'The agent launches training subprocesses, iterating on hyperparameters and architecture choices. Every experiment records a hypothesis and an outcome graded in code &mdash; not just a black-box sweep.'
    },
    {
      id: 'stresslab', x: 580, y: 370, w: 220, h: 70, variant: 'new', title: 'StressLab validation', badge: 'UPD',
      lines: ['Band dropout, noise, geometric'],
      explain: 'Re-evaluates the saved checkpoint on the validation split under synthetic band dropout, sensor noise, and geometric perturbation &mdash; robustness beyond a single clean-data accuracy number.'
    },
    {
      id: 'registry', x: 860, y: 270, w: 230, h: 80, variant: 'default', title: 'Model registry &amp; rollback',
      lines: ['Versioned checkpoints vN', 'Verified rollback to prior version'],
      explain: 'Every submitted model is registered as a new version. Rollback verifies checkpoint integrity and model-loading before restoring a prior approved version as active.'
    },
    {
      id: 'engineer', x: 860, y: 40, w: 230, h: 180, variant: 'new', title: 'Engineer oversight', badge: 'NEW/UPD',
      lines: ['Pause &middot; Resume &middot; manual takeover', 'Approve / reject architecture changes', 'Rollback &middot; full audit trail'],
      explain: 'The engineer console gives a human continuous authority: pause or resume the agent, take manual control, approve or reject proposed architecture changes, roll back to any approved checkpoint, and read a full audit log.'
    },
    {
      id: 'test', x: 580, y: 450, w: 510, h: 70, variant: 'test', title: 'Human-only locked test evaluation',
      lines: ['The ONE path to the held-out test split &mdash; triggered by a human, never by the agent'],
      explain: 'The held-out GEO-Bench test split is locked and reachable only through a human-triggered evaluation script, run once per dataset. The agent, its tools, GeoGuard, DataIQ, StressLab, and training never open it.'
    },
  ];

  const VARIANT_FILL = {
    default: '#161b23', new: '#161b23', highlight: '#1a2430', proposed: '#161b23', test: '#2a1716',
  };
  const VARIANT_STROKE = {
    default: '#2a3240', new: '#e3a63f', highlight: '#5b9bd9', proposed: '#5b9bd9', test: '#e3634f',
  };

  function renderArchDiagram() {
    const mount = $('#archDiagramMount');
    const W = 1120, H = 540;
    let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block; min-width:900px;" role="img" aria-label="GoTrain architecture diagram">`;

    // connectors (drawn first, under nodes)
    const connectors = [
      ['gopilot', 'mcp', false],
      ['mcp', 'strands', true],
      ['strands', 'hooks', false],
      ['hooks', 'geoguard', false],
      ['hooks', 'dataiq', false],
      ['geoguard', 'preflight', false],
      ['dataiq', 'preflight', false],
      ['preflight', 'training', false],
      ['training', 'stresslab', false],
      ['stresslab', 'registry', false],
      ['registry', 'engineer', false],
      ['engineer', 'hooks', false],
    ];
    const byId = Object.fromEntries(ARCH_NODES.map((n) => [n.id, n]));
    connectors.forEach(([a, b, dashed]) => {
      const A = byId[a], B = byId[b];
      const x1 = A.x + A.w, y1 = A.y + A.h / 2;
      const x2 = B.x, y2 = B.y + B.h / 2;
      const midX = (x1 + x2) / 2;
      svg += `<path d="M ${x1},${y1} C ${midX},${y1} ${midX},${y2} ${x2},${y2}" fill="none" stroke="#3d6a93" stroke-width="1.4" ${dashed ? 'stroke-dasharray="4 4"' : ''} marker-end="url(#archArrow)"/>`;
    });

    svg += `<defs><marker id="archArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#3d6a93"/></marker></defs>`;

    // IBM Plex Mono at 600 weight measures ~0.60em per glyph.
    const CHAR_W = 0.60;
    function wrapTitle(title, fontSize, maxWidth) {
      const words = title.replace(/&amp;/g, '&').split(' ');
      const lines = [];
      let cur = '';
      words.forEach((w) => {
        const trial = cur ? `${cur} ${w}` : w;
        if (trial.length * fontSize * CHAR_W > maxWidth && cur) {
          lines.push(cur);
          cur = w;
        } else {
          cur = trial;
        }
      });
      if (cur) lines.push(cur);
      return lines.slice(0, 2).map((l) => l.replace(/&/g, '&amp;'));
    }

    ARCH_NODES.forEach((n) => {
      const fill = VARIANT_FILL[n.variant];
      const stroke = VARIANT_STROKE[n.variant];
      const dash = n.variant === 'proposed' ? 'stroke-dasharray="5 4"' : '';
      const badgeW = n.badge ? n.badge.length * 6.2 + 12 : 0;
      const titleAvailW = n.w - 28 - (badgeW ? badgeW + 8 : 0);
      const titleFontSize = 12.5;
      const titleLines = wrapTitle(n.title, titleFontSize, titleAvailW);
      // single-line title baseline matches the old fixed y+24; a 2nd title line pushes everything below down by one row
      const titleBlockH = titleLines.length > 1 ? 34 : 24;
      const descStartY = n.y + titleBlockH + 14;
      svg += `<g class="arch-node" data-id="${n.id}" tabindex="0" style="font-family:'IBM Plex Mono',monospace;">
        <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1.4" ${dash}/>
        ${titleLines.map((tl, i) => `<text x="${n.x + 14}" y="${n.y + 20 + i * 15}" font-size="${titleFontSize}" font-weight="600" fill="#e7ecf3">${tl}</text>`).join('')}
        ${n.badge ? `<rect x="${n.x + n.w - badgeW - 14}" y="${n.y + 8}" width="${badgeW}" height="14" rx="3" fill="#2c2416" stroke="${stroke}" stroke-width="0.8"/>
        <text x="${n.x + n.w - badgeW - 8}" y="${n.y + 18}" font-size="8" font-weight="700" fill="${stroke}">${n.badge}</text>` : ''}
        ${n.lines.map((l, i) => `<text x="${n.x + 14}" y="${descStartY + i * 15}" font-size="9.8" fill="#9aa7b8">${l}</text>`).join('')}
      </g>`;
    });

    svg += `</svg>`;
    mount.innerHTML = svg;

    $$('.arch-node', mount).forEach((node) => {
      node.addEventListener('click', () => selectArchNode(node.dataset.id));
      node.addEventListener('keypress', (e) => { if (e.key === 'Enter') selectArchNode(node.dataset.id); });
    });
  }

  function selectArchNode(id) {
    const n = ARCH_NODES.find((x) => x.id === id);
    if (!n) return;
    $$('.arch-node').forEach((g) => g.classList.toggle('selected', g.dataset.id === id));
    $('#archExplain').innerHTML = `<h4>${n.title.replace(/&amp;/g, '&')}</h4><div>${n.explain}</div>`;
  }

  renderArchDiagram();

  // ===========================================================
  // 6. RESET DEMO — restores both screens to initial state
  // ===========================================================
  function resetWholeDemo() {
    resetChatDemo();
    closeOverlay('model-studio');
    closeOverlay('engineer-console');
    studioBuilt = false;
    engBuilt = false;
    engState = 'RUNNING_AUTONOMOUSLY';
    $('#studioBody').innerHTML = '';
    $('#engBody').innerHTML = '';
    $$('.arch-node').forEach((g) => g.classList.remove('selected'));
    $('#archExplain').innerHTML = '<div class="arch-explain-placeholder">Click a component above to see what it does.</div>';
    document.getElementById('demo-screen').scrollIntoView({ behavior: 'smooth' });
  }
  $('#resetDemoBtn').addEventListener('click', resetWholeDemo);
  $('#newChatBtn').addEventListener('click', resetChatDemo);

})();
