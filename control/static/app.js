'use strict';
// Control page: polls /api/status every 2 s and draws the panel from it. Switches only move when
// the server reports the new state; between a click and that, the matching light blinks amber.

const $ = (id) => document.getElementById(id);
const POLL_MS = 2000;
const SILENT_MS = 10000;       // the Jetson answering nothing for this long during a power-off
const REFUSED_MS = 1500;       // a request refused faster than this: the Jetson is still on the network
const GUARD_CLOSE_MS = 15000;

let st = null;
let pending = null;            // {op, at}: clicked, server hasn't shown it yet
let flash = null;              // {text, kind, until}: an action's answer, shown for a while
let lostAt = null;             // first failed poll since the last good one
let offSeen = false;           // the server said it is powering off (before running the command)
let silentSince = null;        // since when requests time out instead of being refused
let dryRunDismissed = false;
let lastTonePress = 0;
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

// VACUUM ARMED on its way on (an ON command or the request for it) or off (an OFF, which wins).
function vacTransition() {
  const vac = st.vacuum;
  const p = pendingOp();
  const disarming = vac.pending === 'disarm' || p === 'vacuum/disarm';
  return { arming: !disarming && (vac.pending === 'arm' || p === 'vacuum/arm'), disarming };
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
  $('pwr-state').textContent = stateWord + (r.leftover ? ' (roomba_driver without its launch: switch off to clean up)'
    : (r.external ? ' (launch not started from this page)' : ''));
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

  // SOUND
  const muted = !!(st.sound && st.sound.muted);
  if (p && p.startsWith('sound/') && muted === (p === 'sound/mute')) pending = null;
  const sndMoving = !!pendingOp() && pendingOp().startsWith('sound/');
  const snd = $('snd');
  snd.dataset.pos = sndMoving ? 'mid' : (muted ? 'down' : 'up');
  snd.setAttribute('aria-checked', String(!muted));
  snd.setAttribute('aria-busy', String(sndMoving));
  snd.disabled = sndMoving;
  cls($('l-snd'), muted || sndMoving ? 'a' : null, sndMoving);
  $('snd-state').textContent = muted ? 'Muted' : 'On';

  // JETSON SHUTDOWN
  const sd = $('sd');
  const shutting = (busy && busy.op === 'shutdown') || st.power === 'powering_off' || (busy && busy.then_poweroff);
  sd.dataset.pos = st.power === 'off' ? 'down' : (shutting ? 'mid' : 'up');
  sd.setAttribute('aria-checked', String(st.power === 'on' && !shutting));
  cls($('l-pwr'), st.power === 'off' ? 'r' : (shutting || p === 'shutdown' ? 'a' : 'g'), shutting || p === 'shutdown');
  if (off || shutting) closeGuard('sd');
  $('guard').disabled = off || !!shutting;

  // VACUUM ARMED (commanded state: the driver doesn't report the motors)
  const vac = st.vacuum;
  const ros = st.ros;
  if (p === 'vacuum/arm' && (vac.pending || vac.armed)) pending = null;
  if (p === 'vacuum/disarm' && (vac.pending === 'disarm' || (!vac.armed && !vac.pending))) pending = null;
  const { arming, disarming } = vacTransition();
  const vacMoving = arming || disarming;
  const vacUnknown = !!vac.unknown && !vacMoving;
  // Switching off is always possible, also while arming: the OFF goes out right after the ON.
  const canOff = vac.armed || arming;
  const vs = $('vac');
  vs.dataset.pos = vacMoving || vacUnknown ? 'mid' : (vac.armed ? 'up' : 'down');
  vs.setAttribute('aria-checked', String(vac.armed));
  vs.setAttribute('aria-busy', String(vacMoving));
  const vacOpen = $('vac-guarded').classList.contains('open');
  vs.disabled = !vacOpen || disarming || (!canOff && (vacMoving || !ros.available));
  if (!canOff && !ros.available && !vacMoving) closeGuard('vac');
  if (canOff && !vacOpen) openGuard('vac', false);
  $('vac-guard').disabled = disarming || (!canOff && (vacMoving || !ros.available));
  $('vac-guard').title = !canOff && !ros.available ? ros.reason : '';
  cls($('l-vac'), vac.armed ? 'g' : (vacMoving || vacUnknown ? 'a' : null), vacMoving || vacUnknown);
  let vacText;
  if (disarming) vacText = 'VACUUM: DISARMING… (THE OFF GOES OUT RIGHT AFTER ANY COMMAND ON ITS WAY)';
  else if (arming) vacText = 'VACUUM: ARMING… FLIP DOWN OR CLOSE THE GUARD TO SWITCH OFF.';
  else if (vacUnknown) vacText = 'VACUUM: STATE UNKNOWN: RETRYING OFF. HAZARDS ARE STILL WATCHED.';
  else if (vac.armed) vacText = 'VACUUM: ARMED (VACUUM, MAIN AND SIDE BRUSH ON). CLOSE THE GUARD OR FLIP DOWN TO SWITCH OFF.';
  else if (!ros.available) vacText = `VACUUM: UNAVAILABLE: ${ros.reason}`;
  else vacText = 'VACUUM: OFF. LIFT THE GUARD AND FLIP UP TO ARM.';
  if (vac.note && vac.note.ago_s < 30 && !vacMoving && (!vac.armed || !vac.note.ok)) vacText += ` (${vac.note.message})`;
  const vm = $('vac-msg');
  vm.textContent = vacText;
  vm.classList.toggle('err', !!vac.note && !vac.note.ok && vac.note.ago_s < 30);
  vm.classList.toggle('warn', !vac.armed && !ros.available);
  $('vac-state').textContent = disarming ? 'Disarming' : arming ? 'Arming: flip down to switch off'
    : vac.armed ? 'Armed' : (ros.available ? 'Off' : 'Off. Unavailable: ' + ros.reason);

  // SOUND: tone buttons
  const tones = st.tones;
  const debounced = lastTonePress && Date.now() - lastTonePress < 1000;
  document.querySelectorAll('.tone').forEach((b) => {
    const sending = tones.pending === b.dataset.tone || (pendingOp() === 'tone/' + b.dataset.tone);
    b.classList.toggle('sending', sending);
    b.classList.toggle('busy', !!tones.pending);
    b.disabled = !ros.available || !!tones.pending || debounced;
    b.title = ros.available ? '' : ros.reason;
  });
  if (p && p.startsWith('tone/') && tones.pending) pending = null;
  let toneText;
  if (!ros.available) toneText = `SOUNDS UNAVAILABLE: ${ros.reason}`;
  else if (tones.pending) toneText = `PLAYING ${tones.pending.toUpperCase()}…`;
  else if (tones.last && tones.last.ago_s < 60) toneText = `${tones.last.key.toUpperCase()}: ${tones.last.ok ? 'OK' : 'FAILED'}: ${tones.last.message}`;
  else toneText = 'PRESS A BUTTON TO PLAY IT ON THE ROOMBA (AUTO SOUND MUTE DOES NOT SILENCE THESE)';
  const tm = $('tone-msg');
  tm.textContent = toneText;
  tm.classList.toggle('warn', !ros.available);
  tm.classList.toggle('err', ros.available && !tones.pending && !!tones.last && !tones.last.ok && tones.last.ago_s < 60);

  // BATTERY (pack number and learned capacity come from battery_learner, when it runs)
  const bat = st.battery;
  const pct = bat && bat.percentage != null ? bat.percentage : null;
  const level = pct == null ? '' : pct < 20 ? 'r' : pct < 40 ? 'a' : 'g';
  const lit = pct == null ? 0 : Math.max(pct > 0 ? 1 : 0, Math.round(pct / 10));
  document.querySelectorAll('#batt-meter .gauge-cells i').forEach((c, i) => { c.className = i < lit ? 'on ' + level : ''; });
  $('batt-pct').textContent = pct == null ? '—' : `${pct.toFixed(0)}%`;
  $('batt-pct').classList.toggle('stale', !!bat && bat.age_s > 10);
  const meter = $('batt-meter');
  if (pct == null) { meter.removeAttribute('aria-valuenow'); meter.setAttribute('aria-valuetext', 'unknown'); }
  else { meter.setAttribute('aria-valuenow', pct.toFixed(0)); meter.setAttribute('aria-valuetext', `${pct.toFixed(0)} percent`); }
  $('batt-v').textContent = bat && bat.voltage != null ? `${bat.voltage.toFixed(2)} V` : '—';
  $('batt-pack').textContent = !bat || !bat.pack ? '—' : (/^\d+$/.test(bat.pack) ? `PACK ${bat.pack}` : bat.pack.toUpperCase());
  $('batt-cap').textContent = bat && bat.capacity_ah != null ? `${bat.capacity_ah.toFixed(2)} Ah` : '—';
  $('batt-cap').title = bat && bat.capacity_ah != null ? 'Capacity learned by battery_learner' : 'Learned by battery_learner (not running, or not learned yet)';

  // NEW PACK (battery_learner's pack_swapped service)
  const ps = st.pack_swap;
  if (p === 'battery/new_pack' && ps.pending) pending = null;
  const packSending = ps.pending || pendingOp() === 'battery/new_pack';
  const np = $('newpack');
  np.classList.toggle('sending', packSending);
  np.disabled = !guardOpen('pack') || !ps.available || packSending;
  if (!ps.available && !packSending) closeGuard('pack');
  $('pack-guard').disabled = !ps.available || packSending;
  $('pack-guard').title = ps.available ? '' : ps.reason;
  let packText;
  if (packSending) packText = 'NEW PACK: TELLING BATTERY_LEARNER…';
  else if (ps.last && ps.last.ago_s < 60) packText = `NEW PACK: ${ps.last.ok ? '' : 'FAILED: '}${ps.last.message}`;
  else if (!ps.available) packText = `NEW PACK UNAVAILABLE: ${ps.reason}`;
  else packText = 'NEW PACK: AFTER SWAPPING THE BATTERY, LIFT THE GUARD AND PRESS';
  const pm = $('pack-msg');
  pm.textContent = packText;
  pm.classList.toggle('warn', !ps.available && !packSending && !(ps.last && ps.last.ago_s < 60));
  pm.classList.toggle('err', !packSending && !!ps.last && !ps.last.ok && ps.last.ago_s < 60);

  // Message line: what's happening, else the last result
  if (busy) say(`${busy.message} (${busy.elapsed_s} s)`, 'warn');
  else if (st.last) say(st.last.message, st.last.ok ? null : 'err');
  else say('');

  // Annunciators
  const recentFault = st.last && !st.last.ok && st.last.ago_s < 600 && st.last.op !== 'shutdown';
  if (r.state === 'running') ann('a-stack', r.leftover ? 'a' : 'g', r.leftover ? 'LEFTOVER' : (r.external ? 'RUN (EXT)' : 'RUN'));
  else if (r.state === 'starting') ann('a-stack', 'a', 'STARTING', true);
  else if (r.state === 'stopping') ann('a-stack', 'a', 'SAVING MAP', true);
  else if (recentFault) ann('a-stack', 'r', 'FAULT');
  else ann('a-stack', null, 'OFF');

  const linkVal = { responding: 'RESPONDING', connected: 'CONNECTED', checking: 'CHECKING', not_responding: 'NOT RESPONDING', missing: 'NO CABLE' }[rb.link];
  ann('a-link', checking ? 'a' : linkColor, checking ? 'CHECKING' : linkVal, checking);
  const mode = ros.mode;
  if (!ros.driver) ann('a-mode', null, 'NO DRIVER');
  else if (!mode) ann('a-mode', 'a', 'WAITING', true);
  else ann('a-mode', mode === 'safe' || mode === 'full' ? 'g' : 'a', mode.toUpperCase());
  if (vac.armed) ann('a-vac', 'g', 'ARMED');
  else if (vacMoving) ann('a-vac', 'a', disarming ? 'DISARMING' : 'ARMING', true);
  else if (vacUnknown) ann('a-vac', 'a', 'UNKNOWN', true);
  else if (!ros.available && r.state !== 'stopped') ann('a-vac', 'a', !ros.driver ? 'NO DRIVER' : (mode && mode !== 'safe' && mode !== 'full' ? mode.toUpperCase() + ' MODE' : 'WAITING'));
  else ann('a-vac', null, 'OFF');
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
    `OI ${mode ? mode.toUpperCase() : '—'}`,
    `VAC ${disarming ? 'DISARMING' : arming ? 'ARMING' : vac.armed ? 'ARMED' : 'OFF'}`,
    `SND ${muted ? 'MUTED' : 'ON'}`,
    `BAT ${pct == null ? '—' : pct.toFixed(0) + '%'}${bat && bat.voltage != null ? ' ' + bat.voltage.toFixed(1) + ' V' : ''}`,
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
    // The robot page's three-view layout (camera, map, depth); the frame's name keeps it through the robot page's login.
    const mfd = url + '?layout=mfd';
    if (frame.dataset.src !== mfd) { frame.src = mfd; frame.dataset.src = mfd; }
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
  if (st.power !== 'on' && !st.dry_run) offSeen = true;
  if (lostAt) {
    // Only a power-off the server announced counts: a Wi-Fi drop while the robot is still
    // stopping must not read as done. And the page going away only means this service stopped:
    // the end is when the Jetson stops answering on the network at all, and even that is a
    // sign, not proof, so the last word is to check the Jetson itself.
    if (!offSeen) overlay('SHUTDOWN SEQUENCE', 'LINK LOST', "The Jetson stopped answering before it reported the power-off. Don't remove power. Retrying…", 'warn');
    else if (!silentSince || Date.now() - silentSince < SILENT_MS) {
      overlay('SHUTDOWN SEQUENCE', 'SHUTTING DOWN…', "This page has stopped, but the Jetson is still on the network. Don't remove power yet.", 'warn');
    } else {
      overlay('SHUTDOWN SEQUENCE', 'THE JETSON NO LONGER ANSWERS',
        "Its shutdown looks complete, but this page can't confirm it: check the Jetson itself (power LED, fan) before removing power. To start it again, apply power: it boots and this page comes back by itself.", 'safe');
    }
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
  const started = Date.now();
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), 4000);
  try {
    const res = await fetch('/api/status', { cache: 'no-store', credentials: 'same-origin', signal: abort.signal });
    if (res.status === 401) { location.href = '/login'; return; }
    if (!res.ok) throw new Error(res.status);
    st = await res.json();
    lostAt = null;
    silentSince = null;
    render();
  } catch (e) {
    if (!lostAt) lostAt = Date.now();
    // Refused at once: the Jetson is up (its network still answers). Slow or no answer: it isn't.
    if (Date.now() - started < REFUSED_MS) silentSince = null;
    else if (!silentSince) silentSince = started;
    renderOffline();
  } finally {
    clearTimeout(timer);
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

// Flip-up guards: 'sd' (JETSON SHUTDOWN) and 'vac' (VACUUM ARMED).
const GUARDS = {
  sd: { box: 'guarded', guard: 'guard', sw: 'sd', state: 'sd-state', closed: 'Guard closed', open: 'Guard lifted: flip the switch to shut down' },
  vac: { box: 'vac-guarded', guard: 'vac-guard', sw: 'vac', state: null, closed: '', open: '' },
  pack: { box: 'pack-guarded', guard: 'pack-guard', sw: 'newpack', state: null, closed: '', open: '' },
};
const guardTimers = {};

function guardOpen(name) { return $(GUARDS[name].box).classList.contains('open'); }

function closeGuard(name) {
  const g = GUARDS[name];
  clearTimeout(guardTimers[name]);
  $(g.box).classList.remove('open');
  $(g.guard).setAttribute('aria-expanded', 'false');
  $(g.sw).disabled = true;
  if (g.state) $(g.state).textContent = g.closed;
}

function openGuard(name, autoClose = true) {
  const g = GUARDS[name];
  $(g.box).classList.add('open');
  $(g.guard).setAttribute('aria-expanded', 'true');
  $(g.sw).disabled = false;
  if (g.state) $(g.state).textContent = g.open;
  clearTimeout(guardTimers[name]);
  if (autoClose) {
    guardTimers[name] = setTimeout(() => {
      // An armed (or arming) vacuum switch keeps its guard up (closing it switches the vacuum off).
      if (name === 'vac' && st && (st.vacuum.armed || vacTransition().arming)) return;
      closeGuard(name);
    }, GUARD_CLOSE_MS);
  }
}
function confirmShutdown() {
  const dlg = $('confirm');
  if (typeof dlg.showModal !== 'function') {
    if (window.confirm('Shut down the Jetson? The robot stops first (saving the map), then the Jetson powers off.')) act('shutdown');
    closeGuard('sd');
    return;
  }
  dlg.returnValue = '';
  dlg.showModal();
}

// ---------- layout: collapsible panels and draggable sizes, saved per device ----------
// Tap a panel's header plate to collapse it to its title bar. Drag the grips below the panels
// (and, on a wide screen, the splitter beside the control column) to resize; double-click a grip
// to reset it. Sizes are kept separately for the wide and the phone layout.
const LAYOUT_KEY = 'roomba-control.layout.v1';
const PANELS = ['master', 'sound', 'batt', 'ann', 'disp', 'log'];
const wide = window.matchMedia('(min-width: 980px)');
const SIDE_MIN = 280;
let layout = loadLayout();
let frameAutoHeight = 0;   // the robot page's content height, from its messages (phone layout)

function defaultLayout() { return { collapsed: {}, h: { wide: {}, narrow: {} }, side: null }; }

function loadLayout() {
  try {
    const l = JSON.parse(localStorage.getItem(LAYOUT_KEY));
    if (l && l.collapsed && l.h && l.h.wide && l.h.narrow) return l;
  } catch (e) { /* none saved, or private mode */ }
  return defaultLayout();
}

function saveLayout() {
  try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); } catch (e) { /* private mode */ }
}

function mode() { return wide.matches ? 'wide' : 'narrow'; }
function body(name) { return $(name + '-body'); }

function applyLayout() {
  for (const name of PANELS) {
    const panel = $('panel-' + name);
    const collapsed = !!layout.collapsed[name];
    panel.classList.toggle('collapsed', collapsed);
    panel.querySelector('.plate-toggle').setAttribute('aria-expanded', String(!collapsed));
    const h = layout.h[mode()][name];
    const b = body(name);
    // On a wide screen the display always fills the window's height.
    const sized = !!h && !(name === 'disp' && wide.matches);
    b.style.height = sized ? h + 'px' : '';
    b.classList.toggle('sized', sized);
  }
  const lay = $('layout');
  lay.classList.toggle('disp-collapsed', !!layout.collapsed.disp);
  if (layout.side && wide.matches) lay.style.setProperty('--side-w', clampSide(layout.side) + 'px');
  else lay.style.removeProperty('--side-w');
  applyFrameHeight();
}

function clampSide(w) { return Math.round(Math.max(SIDE_MIN, Math.min(w, window.innerWidth - 360))); }

// Phone layout: the frame follows the robot page's height unless its panel was sized by hand.
function applyFrameHeight() {
  const frame = $('frame');
  const auto = !wide.matches && !layout.h.narrow.disp && frameAutoHeight > 0;
  frame.style.height = auto ? frameAutoHeight + 'px' : '';
}

function togglePanel(name) {
  layout.collapsed[name] = !layout.collapsed[name];
  saveLayout();
  applyLayout();
}

// Drag with mouse, pen or touch: pointer capture keeps the drag even over the robot frame.
function dragger(el, onStart, onMove, onEnd) {
  el.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    try { el.setPointerCapture(ev.pointerId); } catch (e) { /* synthetic */ }
    const start = onStart(ev);
    if (!start) return;
    el.classList.add('dragging');
    document.body.classList.add('resizing');
    document.body.style.cursor = getComputedStyle(el).cursor;
    const move = (e) => onMove(start, e);
    const up = () => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
      el.classList.remove('dragging');
      document.body.classList.remove('resizing');
      document.body.style.cursor = '';
      onEnd(start);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  });
}

function setPanelHeight(name, h) {
  const b = body(name);
  const v = Math.round(Math.max(60, Math.min(h, 4000)));
  layout.h[mode()][name] = v;
  b.style.height = v + 'px';
  b.classList.add('sized');
  if (name === 'disp') applyFrameHeight();
}

function initLayout() {
  for (const name of PANELS) {
    $('panel-' + name).querySelector('.plate-toggle').addEventListener('click', () => togglePanel(name));
  }
  document.querySelectorAll('.grip').forEach((grip) => {
    const name = grip.dataset.grip;
    dragger(grip,
      (ev) => (layout.collapsed[name] ? null : { y: ev.clientY, h: body(name).getBoundingClientRect().height }),
      (s, e) => setPanelHeight(name, s.h + e.clientY - s.y),
      () => saveLayout());
    grip.addEventListener('dblclick', () => { delete layout.h[mode()][name]; saveLayout(); applyLayout(); });
    grip.addEventListener('keydown', (e) => {
      const step = e.key === 'ArrowDown' ? 20 : e.key === 'ArrowUp' ? -20 : 0;
      if (!step || layout.collapsed[name]) return;
      e.preventDefault();
      setPanelHeight(name, body(name).getBoundingClientRect().height + step);
      saveLayout();
    });
  });
  const split = $('vsplit');
  const sideWidth = () => $('side').getBoundingClientRect().width;
  dragger(split,
    (ev) => ({ x: ev.clientX, w: sideWidth() }),
    (s, e) => { layout.side = clampSide(s.w + e.clientX - s.x); applyLayout(); },
    () => saveLayout());
  split.addEventListener('dblclick', () => { layout.side = null; saveLayout(); applyLayout(); });
  split.addEventListener('keydown', (e) => {
    const step = e.key === 'ArrowRight' ? 20 : e.key === 'ArrowLeft' ? -20 : 0;
    if (!step) return;
    e.preventDefault();
    layout.side = clampSide(sideWidth() + step);
    saveLayout();
    applyLayout();
  });
  $('reset-layout').addEventListener('click', () => {
    layout = defaultLayout();
    saveLayout();
    applyLayout();
    // The robot page in the display keeps its own view sizes: reset those too.
    const frame = $('frame');
    if (frame.contentWindow && frame.dataset.src) frame.contentWindow.postMessage({ type: 'roomba-mfd-reset' }, '*');
  });
  // The robot page (three-view) reports its height so the stacked phone layout shows it whole.
  window.addEventListener('message', (e) => {
    if (e.source !== $('frame').contentWindow || !e.data || e.data.type !== 'roomba-mfd-height') return;
    // null: the robot page fills whatever height the frame has (the wide three-view).
    const h = e.data.height;
    if (h === null) frameAutoHeight = 0;
    else if (Number.isFinite(h)) frameAutoHeight = Math.round(Math.max(240, Math.min(h, 4000)));
    else return;
    applyFrameHeight();
  });
  wide.addEventListener('change', applyLayout);
  window.addEventListener('resize', () => { if (layout.side) applyLayout(); });
  applyLayout();
}

document.addEventListener('DOMContentLoaded', () => {
  initLayout();
  $('pwr').addEventListener('click', () => {
    if (!st) return;
    if (st.robot.state === 'stopped') act('start');
    else if (st.robot.state === 'running') act('stop');
  });
  $('comms').addEventListener('click', () => act('comms'));
  $('snd').addEventListener('click', () => {
    if (st) act(st.sound && st.sound.muted ? 'sound/unmute' : 'sound/mute');
  });
  $('guard').addEventListener('click', () => {
    if (guardOpen('sd')) closeGuard('sd'); else openGuard('sd');
  });
  $('vac-guard').addEventListener('click', () => {
    if (!guardOpen('vac')) { openGuard('vac'); return; }
    // Closing the guard over an armed (or arming) switch pushes it off, as on a real panel.
    if (st && (st.vacuum.armed || vacTransition().arming)) act('vacuum/disarm');
    closeGuard('vac');
  });
  $('vac').addEventListener('click', () => {
    if (!st || !guardOpen('vac')) return;
    act(st.vacuum.armed || vacTransition().arming ? 'vacuum/disarm' : 'vacuum/arm');
    openGuard('vac');
  });
  $('pack-guard').addEventListener('click', () => { if (guardOpen('pack')) closeGuard('pack'); else openGuard('pack'); });
  $('newpack').addEventListener('click', () => {
    if (!guardOpen('pack')) return;
    act('battery/new_pack');
    closeGuard('pack');
  });
  document.querySelectorAll('.tone').forEach((b) => b.addEventListener('click', () => {
    if (lastTonePress && Date.now() - lastTonePress < 1000) return;
    lastTonePress = Date.now();
    act('tone/' + b.dataset.tone);
    setTimeout(() => { if (st) render(); }, 1050);
  }));
  $('sd').addEventListener('click', () => {
    if (!guardOpen('sd')) return;
    $('sd').dataset.pos = 'mid';
    confirmShutdown();
  });
  $('confirm').addEventListener('close', () => {
    const ok = $('confirm').returnValue === 'ok';
    closeGuard('sd');
    if (ok) act('shutdown'); else if (st) render();
  });
  $('ov-close').addEventListener('click', () => { dryRunDismissed = true; $('overlay').hidden = true; });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
  poll();
});
