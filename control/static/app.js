'use strict';
// Control page: polls /api/status every 2 s and draws the panel from it. Switches only move when
// the server reports the new state; between a click and that, the matching light blinks amber.

const $ = (id) => document.getElementById(id);
const POLL_MS = 2000;
const SAFE_AFTER_MS = 20000;   // after the page stops answering during a power-off
const GUARD_CLOSE_MS = 15000;

let st = null;
let pending = null;            // {op, at}: clicked, server hasn't shown it yet
let flash = null;              // {text, kind, until}: an action's answer, shown for a while
let lostAt = null;             // first failed poll since the last good one
let offSeen = false;           // the server reported that it ran the power-off command
let dryRunDismissed = false;
let guardTimer = null;
let pollTimer = null;

function cls(el, color, blink) {
  el.classList.remove('g', 'a', 'r', 'blink');
  if (color) el.classList.add(color);
  if (blink) el.classList.add('blink');
}

function ann(id, color, value, blink) {
  const el = $(id);
  cls(el, color, blink);
  el.querySelector('.ann-val').textContent = value;
  const name = el.querySelector('.ann-name').textContent;
  const word = { g: 'OK', a: 'caution', r: 'warning' }[color] || 'off';
  el.setAttribute('aria-label', `${name}: ${value} (${word})`);
}

function fmtUptime(s) {
  if (s == null) return '—';
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  const hh = String(h).padStart(2, '0'), mm = String(m).padStart(2, '0'), ss = String(s % 60).padStart(2, '0');
  return d ? `${d}d ${hh}:${mm}` : `${hh}:${mm}:${ss}`;
}

function gb(bytes) { return (bytes / 1e9).toFixed(1) + ' GB'; }

function say(text, kind) {
  if (flash && Date.now() < flash.until) ({ text, kind } = flash);
  const m = $('msg');
  m.textContent = text || '\u00a0';
  m.classList.toggle('err', kind === 'err');
  m.classList.toggle('warn', kind === 'warn');
}

function shuttingDown() {
  const busy = st && st.busy;
  return !!st && (st.power !== 'on' || (!!busy && (busy.op === 'shutdown' || busy.then_poweroff)));
}

function pendingOp() {
  if (pending && Date.now() - pending.at > 8000) pending = null;
  return pending && pending.op;
}

function render() {
  const r = st.robot;
  const busy = st.busy;
  const p = pendingOp();
  if (busy || (p === 'start' && r.state === 'running') || (p === 'stop' && r.state === 'stopped')) pending = null;

  $('page-url').textContent = st.page_url || location.origin + '/';
  $('link-lamp').className = 'lamp-dot g';
  $('link-lamp').setAttribute('aria-label', 'Connection to the Jetson: OK');

  // ROBOT PWR
  const pwr = $('pwr');
  const moving = r.state === 'starting' || r.state === 'stopping';
  pwr.dataset.pos = moving ? 'mid' : (r.state === 'running' ? 'up' : 'down');
  pwr.setAttribute('aria-checked', String(r.state === 'running' || r.state === 'starting'));
  pwr.setAttribute('aria-busy', String(moving || !!p));
  const off = st.power !== 'on';
  pwr.disabled = off || !!busy || !!p;
  const stateWord = { running: 'Running', stopped: 'Stopped', starting: 'Starting', stopping: 'Stopping, saving map' }[r.state];
  $('pwr-state').textContent = stateWord + (r.external ? ' (launch not started from this page)' : '');
  cls($('l-run'), r.state === 'running' ? 'g' : null);
  cls($('l-xfer'), (r.state === 'starting' || p === 'start') ? 'a' : null, r.state === 'starting' || p === 'start');
  cls($('l-save'), (r.state === 'stopping' || p === 'stop') ? 'a' : null, r.state === 'stopping' || p === 'stop');

  // COMMS CHECK
  const rb = st.roomba;
  const checking = !!rb.checking || p === 'comms';
  $('comms').disabled = checking || off || !!busy;
  $('comms').classList.toggle('pressed', checking);
  const linkColor = { responding: 'g', connected: 'a', checking: 'a', not_responding: 'r', missing: 'r' }[rb.link];
  cls($('l-link'), checking ? 'a' : linkColor, checking);
  const res = rb.result;
  let commsText;
  if (checking) commsText = `COMMS: CHECKING${rb.checking && rb.checking.via === 'ros' ? ' VIA ROS (DRIVER RUNNING)' : ' ON THE SERIAL PORT'}…`;
  else if (res) commsText = `COMMS (${res.via.toUpperCase()}, ${res.ago_s}s AGO): ${res.message}`;
  else commsText = 'COMMS: PRESS TEST TO CHECK THE ROOMBA LINK';
  const cm = $('comms-msg');
  cm.textContent = commsText;
  cm.classList.toggle('err', !checking && !!res && res.link !== 'responding');
  $('comms-state').textContent = commsText;

  // JETSON SHUTDOWN
  const sd = $('sd');
  const shutting = (busy && busy.op === 'shutdown') || st.power === 'powering_off' || (busy && busy.then_poweroff);
  sd.dataset.pos = st.power === 'off' ? 'down' : (shutting ? 'mid' : 'up');
  sd.setAttribute('aria-checked', String(st.power === 'on' && !shutting));
  cls($('l-pwr'), st.power === 'off' ? 'r' : (shutting || p === 'shutdown' ? 'a' : 'g'), shutting || p === 'shutdown');
  if (off || shutting) closeGuard();
  $('guard').disabled = off || !!shutting;

  // Message line: what's happening, else the last result
  if (busy) say(`${busy.message} (${busy.elapsed_s} s)`, 'warn');
  else if (st.last) say(st.last.message, st.last.ok ? null : 'err');
  else say('');

  // Annunciators
  const recentFault = st.last && !st.last.ok && st.last.ago_s < 600 && st.last.op !== 'shutdown';
  if (r.state === 'running') ann('a-stack', 'g', r.external ? 'RUN (EXT)' : 'RUN');
  else if (r.state === 'starting') ann('a-stack', 'a', 'STARTING', true);
  else if (r.state === 'stopping') ann('a-stack', 'a', 'SAVING MAP', true);
  else if (recentFault) ann('a-stack', 'r', 'FAULT');
  else ann('a-stack', null, 'OFF');

  const linkVal = { responding: 'RESPONDING', connected: 'CONNECTED', checking: 'CHECKING', not_responding: 'NOT RESPONDING', missing: 'NO CABLE' }[rb.link];
  ann('a-link', checking ? 'a' : linkColor, checking ? 'CHECKING' : linkVal, checking);
  ann('a-cam', st.camera ? 'g' : 'r', st.camera ? 'PRESENT' : 'MISSING');
  const t = st.cpu_temp_c;
  ann('a-temp', t == null ? 'r' : t >= 90 ? 'r' : t >= 75 ? 'a' : 'g', t == null ? 'N/A' : `${t.toFixed(0)}\u00b0C`);
  const w = st.wifi;
  ann('a-wifi', !w ? 'r' : w.level_dbm < -78 ? 'r' : w.level_dbm < -68 ? 'a' : 'g', w ? `${w.level_dbm.toFixed(0)} dBm` : 'NONE');
  const d = st.disk;
  const freeFrac = d ? d.free_bytes / d.total_bytes : 0;
  ann('a-disk', !d ? 'r' : (d.free_bytes < 1e9 || freeFrac < .02) ? 'r' : (d.free_bytes < 5e9 || freeFrac < .08) ? 'a' : 'g', d ? gb(d.free_bytes) + ' FREE' : 'N/A');

  // HUD line
  const parts = [
    `STACK ${r.state.toUpperCase()}${r.pid ? ' PID ' + r.pid : ''}`,
    `UP ${fmtUptime(st.uptime_s)}`,
    `CPU ${t == null ? '—' : t.toFixed(0) + '\u00b0C'}`,
    `WIFI ${w ? w.level_dbm.toFixed(0) + ' dBm' : '—'}`,
    `DISK ${d ? gb(d.free_bytes) : '—'}`,
    `LINK ${linkVal}`,
  ];
  $('hud').textContent = 'SYS \u25b8 ' + parts.join(' \u00b7 ');

  // Robot log
  const log = $('log');
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.replaceChildren();
  const lines = st.log.lines.length ? st.log.lines : ['NO DATA'];
  for (const line of lines) {
    const div = document.createElement('div');
    if (/\b(ERROR|FATAL)\b|\[ERROR\]|died|exception/i.test(line)) div.className = 'err';
    else if (/\bWARN(ING)?\b|lost/i.test(line)) div.className = 'warn';
    div.textContent = line;
    log.appendChild(div);
  }
  if (atBottom) log.scrollTop = log.scrollHeight;
  $('log-age').textContent = st.log.age_s == null ? '' : `UPD ${st.log.age_s}s`;

  // Robot page
  const frame = $('frame');
  const open = $('open');
  const url = st.robot_url;
  if (url) open.href = url;
  open.setAttribute('aria-disabled', String(!(r.web_up && url)));
  if (r.web_up && url) {
    if (frame.dataset.src !== url) { frame.src = url; frame.dataset.src = url; }
    frame.hidden = false;
    $('nosignal').hidden = true;
  } else {
    if (frame.dataset.src) { frame.removeAttribute('src'); delete frame.dataset.src; }
    frame.hidden = true;
    $('nosignal').hidden = false;
  }

  renderOverlay();
}

function overlay(tag, title, text, kind, closable) {
  const box = document.querySelector('.ov-box');
  $('ov-tag').textContent = tag;
  $('ov-title').textContent = title;
  $('ov-text').textContent = text;
  box.classList.toggle('safe', kind === 'safe');
  box.classList.toggle('warn', kind === 'warn');
  $('ov-close').hidden = !closable;
  $('overlay').hidden = false;
}

function renderOverlay() {
  const busy = st && st.busy;
  if (!shuttingDown()) {
    offSeen = false;
    dryRunDismissed = false;
    $('overlay').hidden = true;
    return;
  }
  if (st.power === 'off' && !st.dry_run) offSeen = true;
  if (lostAt) {
    // Only a power-off the server confirmed counts: a Wi-Fi drop while the robot is still
    // stopping must not read as "safe".
    const left = Math.ceil((SAFE_AFTER_MS - (Date.now() - lostAt)) / 1000);
    if (!offSeen) overlay('SHUTDOWN SEQUENCE', 'LINK LOST', "The Jetson stopped answering before it reported the power-off. Don't remove power until this page says so. Retrying…", 'warn');
    else if (left > 0) overlay('SHUTDOWN SEQUENCE', 'POWERING OFF…', `Don't remove power yet (${left} s).`, 'warn');
    else overlay('SHUTDOWN COMPLETE', 'SAFE TO REMOVE POWER', 'The Jetson has shut down. To start it again, apply power: it boots and this page comes back by itself.', 'safe');
    return;
  }
  if (st.power === 'off' && st.dry_run) {
    if (!dryRunDismissed) overlay('DRY RUN', 'SHUTDOWN SEQUENCE COMPLETE', `${st.last ? st.last.message : ''} A real shutdown would now say it is safe to remove power.`, 'safe', true);
    return;
  }
  if (st.power === 'off') overlay('SHUTDOWN SEQUENCE', 'POWERING OFF…', "Don't remove power yet: waiting for the Jetson to go down.", 'warn');
  else if (st.power === 'powering_off') overlay('SHUTDOWN SEQUENCE', 'POWERING OFF…', "Don't remove power yet.", 'warn');
  else overlay('SHUTDOWN SEQUENCE', 'STOPPING ROBOT', `${busy ? busy.message + ` (${busy.elapsed_s} s)` : ''} Don't remove power yet.`, 'warn');
}

function renderOffline() {
  $('link-lamp').className = 'lamp-dot r';
  $('link-lamp').setAttribute('aria-label', 'Connection to the Jetson: lost');
  if (shuttingDown()) { renderOverlay(); return; }
  $('hud').textContent = 'SYS \u25b8 LINK LOST: THE JETSON IS NOT ANSWERING. RETRYING\u2026';
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    const res = await fetch('/api/status', { cache: 'no-store', credentials: 'same-origin' });
    if (res.status === 401) { location.href = '/login'; return; }
    if (!res.ok) throw new Error(res.status);
    st = await res.json();
    lostAt = null;
    render();
  } catch (e) {
    if (!lostAt) lostAt = Date.now();
    renderOffline();
  }
  const down = shuttingDown();
  pollTimer = setTimeout(poll, down ? 1000 : (document.hidden ? 8000 : POLL_MS));
}

async function act(op) {
  pending = { op, at: Date.now() };
  flash = null;
  if (st) render();
  try {
    const res = await fetch('/api/' + op, { method: 'POST', credentials: 'same-origin' });
    if (res.status === 401) { location.href = '/login'; return; }
    const body = await res.json().catch(() => ({ ok: false, message: `HTTP ${res.status}` }));
    if (!body.ok) pending = null;
    flash = { text: body.message, kind: body.ok ? 'warn' : 'err', until: Date.now() + (body.ok ? 2500 : 8000) };
  } catch (e) {
    pending = null;
    flash = { text: 'No answer from the Jetson.', kind: 'err', until: Date.now() + 8000 };
  }
  say('');
  poll();
}

function closeGuard() {
  clearTimeout(guardTimer);
  $('guarded').classList.remove('open');
  $('guard').setAttribute('aria-expanded', 'false');
  $('sd').disabled = true;
  $('sd-state').textContent = 'Guard closed';
}

function openGuard() {
  $('guarded').classList.add('open');
  $('guard').setAttribute('aria-expanded', 'true');
  $('sd').disabled = false;
  $('sd-state').textContent = 'Guard lifted: flip the switch to shut down';
  clearTimeout(guardTimer);
  guardTimer = setTimeout(closeGuard, GUARD_CLOSE_MS);
}

function confirmShutdown() {
  const dlg = $('confirm');
  if (typeof dlg.showModal !== 'function') {
    if (window.confirm('Shut down the Jetson? The robot stops first (saving the map), then the Jetson powers off.')) act('shutdown');
    closeGuard();
    return;
  }
  dlg.returnValue = '';
  dlg.showModal();
}

document.addEventListener('DOMContentLoaded', () => {
  $('pwr').addEventListener('click', () => {
    if (!st) return;
    if (st.robot.state === 'stopped') act('start');
    else if (st.robot.state === 'running') act('stop');
  });
  $('comms').addEventListener('click', () => act('comms'));
  $('guard').addEventListener('click', () => {
    if ($('guarded').classList.contains('open')) closeGuard(); else openGuard();
  });
  $('sd').addEventListener('click', () => {
    if (!$('guarded').classList.contains('open')) return;
    $('sd').dataset.pos = 'mid';
    confirmShutdown();
  });
  $('confirm').addEventListener('close', () => {
    const ok = $('confirm').returnValue === 'ok';
    closeGuard();
    if (ok) act('shutdown'); else if (st) render();
  });
  $('ov-close').addEventListener('click', () => { dryRunDismissed = true; $('overlay').hidden = true; });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
  poll();
});
