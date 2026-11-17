'use strict';

// ---------- state ----------
const $ = (id) => document.getElementById(id);
const ICON = { person: '🧍', cat: '🐈', dog: '🐕' };
const PETS = new Set(['cat', 'dog']);
const CAMERA_HFOV = 105 * Math.PI / 180;
const FOV_RANGE_M = 3.0;

const S = {
  data: null,
  map: null,            // { meta, img }
  mapVersion: -1,
  zones: [],
  zonesVersion: -1,
  path: [],
  pathVersion: -1,
  view: { x: 0, y: 0, scale: 60, ready: false },  // world center (m), pixels per meter
  drawing: false,
  naming: false,
  goalMode: false,
  draft: [],
  plan: [],
  planVersion: -1,
  obstacles: [],
  obstaclesVersion: -1,
  voice: false,
  lastSpoken: null,
  lastSpokeAt: 0,
  showExcluded: false,
  dirty: true,
  userViewAt: 0,        // last pan / zoom by hand (the three-view map follows the robot otherwise)
};

// ---------- API ----------
async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'roomba-web' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) { location.href = '/login'; throw new Error('signed out'); }
  if (!res.ok) throw new Error((await res.text()) || res.statusText);
  const type = res.headers.get('content-type') || '';
  return type.includes('json') ? res.json() : res.text();
}

// ---------- live connection ----------
function setPill(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  el.className = 'pill' + (cls ? ' ' + cls : '');
}

function connect() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.onopen = () => setPill('pill-conn', 'Live', 'ok');
  ws.onmessage = (ev) => onState(JSON.parse(ev.data));
  ws.onclose = async () => {
    setPill('pill-conn', 'Reconnecting…', 'warn');
    try { await api('GET', '/api/state'); } catch (e) { /* redirected if signed out */ }
    setTimeout(connect, 1500);
  };
}

function onState(d) {
  S.data = d;
  const track = { ok: ['Tracking OK', 'ok'], lost: ['Tracking lost', 'bad'], starting: ['Tracking starting', 'warn'] };
  let [tt, tc] = track[d.tracking] || ['Tracking: ' + d.tracking, 'bad'];
  const vf = d.vslam_filter;
  if (vf && vf.state !== 'idle') { tt = 'Re-acquiring position…'; tc = 'warn'; }
  else if (vf && vf.filtered > 0) tt += ` · ${vf.filtered} glitch${vf.filtered === 1 ? '' : 'es'} filtered`;
  setPill('pill-track', tt, tc);
  if (d.mode === 'mapping') setPill('pill-loc', 'Mapping (new map)', 'ok');
  else if (d.localized) setPill('pill-loc', 'Localized in saved map', 'ok');
  else setPill('pill-loc', 'Localizing… zones inactive', 'warn');
  $('zone-note').textContent = d.mode === 'localization' && !d.localized
    ? 'Zones apply once the camera recognizes a mapped area.'
    : 'Zones are saved with this map. People inside an exclusion zone are ignored; the robot never plans into a keep-out zone (manual driving can still enter it).';

  if (d.map_version !== S.mapVersion) loadMap(d.map_version);
  if (d.zones_version !== S.zonesVersion) loadZones(d.zones_version);
  if (d.path_version !== S.pathVersion) loadPath(d.path_version);
  if (d.plan_version !== S.planVersion) loadPoints('plan', d.plan_version);
  if (d.obstacles_version !== S.obstaclesVersion) loadPoints('obstacles', d.obstacles_version);
  if (!S.view.ready && d.pose) { S.view.x = d.pose.x; S.view.y = d.pose.y; S.view.ready = true; }
  renderDetections();
  renderNav(d);
  renderRobot(d);
  renderGrid(d);
  const e = d.explore;
  $('map-explore').hidden = !e;
  if (e) $('map-explore').textContent = `Explorer: ${e.state}${e.message ? ' · ' + e.message : ''}${e.frontiers != null ? ' · ' + e.frontiers + ' frontier groups' : ''}`;
  S.dirty = true;
}

async function loadPoints(kind, version) {
  S[kind + 'Version'] = version;
  try { S[kind] = (await api('GET', '/api/' + kind)).points; S.dirty = true; } catch (e) { /* retry next state */ }
}

// ---------- navigation instructions ----------
const NAV_GROUP = {
  forward: 'go', bear_left: 'go', bear_right: 'go', turn_left: 'turn', turn_right: 'turn',
  back: 'back', wait: 'wait', stop_obstacle: 'stop', arrived: 'done', failed: 'back', canceled: 'idle', idle: 'idle',
};

function renderNav(d) {
  const n = d.nav;
  $('nav-banner').hidden = !d.nav_available;
  $('btn-goal').disabled = !d.nav_available;
  $('btn-goal').title = d.nav_available ? '' : 'Start navigation on the robot: ~/run_nav.sh';
  const f = d.floor;
  let floorText = '';
  if (f) {
    const how = f.fixed_height ? ' (calibrated)' : (f.floor_visible ? '' : ' (floor not in view – using last estimate)');
    floorText = `Camera ${f.camera_height_m.toFixed(2)} m above the floor${how} · tilt ${f.tilt_deg.toFixed(0)}°`;
  }
  $('floor-note').textContent = floorText;
  if (!d.nav_available) return;
  const code = n ? n.code : 'idle';
  $('nav-banner').className = 'nav-banner ' + (NAV_GROUP[code] || 'idle');
  $('nav-arrow').textContent = n ? n.arrow : '';
  $('nav-text').textContent = n ? n.text : 'Navigation ready';
  $('nav-dist').textContent = n && n.active && n.distance_m > 0 ? `${n.distance_m.toFixed(1)} m to go` : '';
  $('btn-stop-nav').hidden = !((n && n.active) || d.nav_goal);
  speak(code, n ? n.text : '');
}

function speak(code, text) {
  if (!S.voice || !('speechSynthesis' in window) || !text) return;
  const now = Date.now();
  const repeatStop = code === 'stop_obstacle' && now - S.lastSpokeAt > 3000;
  if (code === S.lastSpoken && !repeatStop) return;
  S.lastSpoken = code;
  if (code === 'idle' || code === 'wait') return;
  S.lastSpokeAt = now;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.replace('–', ','));
  u.rate = 1.1;
  speechSynthesis.speak(u);
}

async function setGoal([sx, sy]) {
  const [x, y] = toWorld(sx, sy);
  S.goalMode = false;
  updateToolbar();
  try {
    await api('POST', '/api/goal', { x: Number(x.toFixed(3)), y: Number(y.toFixed(3)) });
    showMsg('');
  } catch (e) {
    showMsg('Could not start navigation: ' + e.message);
  }
}

async function loadMap(version) {
  S.mapVersion = version;
  try {
    const meta = await api('GET', '/api/map.json');
    const img = new Image();
    img.onload = () => { S.map = { meta, img }; S.dirty = true; };
    img.src = '/api/map.png?v=' + meta.version;
  } catch (e) { /* no map yet */ }
}

async function loadZones(version) {
  S.zonesVersion = version;
  try { S.zones = (await api('GET', '/api/zones')).zones; renderZones(); S.dirty = true; } catch (e) { /* retry next state */ }
}

async function loadPath(version) {
  S.pathVersion = version;
  try { S.path = (await api('GET', '/api/path')).points; S.dirty = true; } catch (e) { /* ignore */ }
}

// ---------- lists ----------
function li(cls) { const el = document.createElement('li'); if (cls) el.className = cls; return el; }
function span(cls, text) { const el = document.createElement('span'); el.className = cls; el.textContent = text; return el; }

function renderDetections() {
  const dets = (S.data && S.data.detections) || [];
  const active = dets.filter((d) => !d.excluded);
  const shown = S.showExcluded ? dets : active;
  const excludedCount = dets.length - active.length;
  $('det-count').textContent = `(${active.length}${excludedCount ? `, ${excludedCount} excluded` : ''})`;
  $('tab-count').textContent = active.length;
  $('tab-count').hidden = !active.length;
  const list = $('det-list');
  list.replaceChildren();
  if (!shown.length) { const e = li('empty'); e.textContent = 'Nobody in view'; list.append(e); return; }
  for (const d of shown) {
    const row = li(d.excluded ? 'excluded' : '');
    row.append(span('icon', ICON[d.cls] || '•'));
    const name = span('name', d.cls);
    if (d.excluded) name.append(span('reason', 'excluded: ' + (d.reason || '')));
    row.append(name, span('dist', d.dist.toFixed(2) + ' m'));
    list.append(row);
  }
}

function renderZones() {
  const list = $('zone-list');
  list.replaceChildren();
  if (!S.zones.length) { const e = li('empty'); e.textContent = 'No zones yet'; list.append(e); return; }
  for (const z of S.zones) {
    const row = li();
    row.append(span('zname', z.name));
    const keepout = z.kind === 'keepout';
    row.append(span('zkind' + (keepout ? ' keepout' : ''), keepout ? 'keep-out' : 'exclusion'));
    const del = document.createElement('button');
    del.className = 'ghost';
    del.textContent = '🗑 Delete';
    let armed = null;
    del.onclick = async () => {
      if (!armed) {  // first tap arms, second tap within 4 s deletes
        del.textContent = 'Tap to confirm';
        del.className = 'danger';
        armed = setTimeout(() => { armed = null; del.textContent = '🗑 Delete'; del.className = 'ghost'; }, 4000);
        return;
      }
      clearTimeout(armed);
      del.disabled = true;
      try { await api('DELETE', '/api/zones/' + encodeURIComponent(z.name)); } catch (e) { showMsg(e.message); del.disabled = false; }
    };
    row.append(del);
    list.append(row);
  }
}

// ---------- map view ----------
const canvas = $('map');
const ctx = canvas.getContext('2d');

function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  S.dirty = true;
}
window.addEventListener('resize', resize);
// Layout changes (tabs, three-view) resize the map without a window resize.
if ('ResizeObserver' in window) new ResizeObserver(resize).observe(canvas.parentElement);

function W() { return canvas.getBoundingClientRect(); }
function toScreen(x, y) { const r = W(); return [r.width / 2 + (x - S.view.x) * S.view.scale, r.height / 2 - (y - S.view.y) * S.view.scale]; }
function toWorld(sx, sy) { const r = W(); return [S.view.x + (sx - r.width / 2) / S.view.scale, S.view.y - (sy - r.height / 2) / S.view.scale]; }

function drawGrid(r) {
  const step = S.view.scale > 25 ? 1 : 5;
  const [x0, y1] = toWorld(0, 0);
  const [x1, y0] = toWorld(r.width, r.height);
  ctx.strokeStyle = '#1c222c';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) { const [sx] = toScreen(x, 0); ctx.moveTo(sx, 0); ctx.lineTo(sx, r.height); }
  for (let y = Math.floor(y0 / step) * step; y <= y1; y += step) { const [, sy] = toScreen(0, y); ctx.moveTo(0, sy); ctx.lineTo(r.width, sy); }
  ctx.stroke();
}

function drawMap() {
  if (!S.map) return;
  const m = S.map.meta;
  const [ox, oy] = m.origin;
  const wM = m.width * m.resolution;
  const hM = m.height * m.resolution;
  ctx.save();
  const [sx, sy] = toScreen(ox, oy);  // map image bottom-left
  ctx.translate(sx, sy);
  ctx.rotate(-(m.yaw || 0));
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(S.map.img, 0, -hM * S.view.scale, wM * S.view.scale, hM * S.view.scale);
  ctx.restore();
}

function polyPath(points) {
  ctx.beginPath();
  points.forEach(([x, y], i) => { const [sx, sy] = toScreen(x, y); if (i) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy); });
}

function drawPath() {
  if (S.path.length < 2) return;
  polyPath(S.path);
  ctx.strokeStyle = 'rgba(64, 200, 180, 0.7)';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawZones() {
  ctx.font = '600 12px system-ui, sans-serif';
  for (const z of S.zones) {
    const keepout = z.kind === 'keepout';
    polyPath(z.points);
    ctx.closePath();
    ctx.fillStyle = keepout ? 'rgba(229, 72, 77, 0.22)' : 'rgba(240, 140, 46, 0.18)';
    ctx.fill();
    ctx.strokeStyle = keepout ? '#e5484d' : '#f08c2e';
    ctx.lineWidth = 2;
    ctx.stroke();
    const cx = z.points.reduce((a, p) => a + p[0], 0) / z.points.length;
    const cy = z.points.reduce((a, p) => a + p[1], 0) / z.points.length;
    const [sx, sy] = toScreen(cx, cy);
    ctx.fillStyle = keepout ? '#ff8a8d' : '#f5b97a';
    ctx.textAlign = 'center';
    ctx.fillText(z.name, sx, sy);
  }
  if (S.draft.length) {
    polyPath(S.draft);
    ctx.strokeStyle = '#f5e042';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    S.draft.forEach(([x, y], i) => {
      const [sx, sy] = toScreen(x, y);
      ctx.beginPath();
      ctx.arc(sx, sy, i === 0 ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? '#ffffff' : '#f5e042';
      ctx.fill();
    });
  }
}

function drawCamera(pose) {
  if (!pose) return;
  const [sx, sy] = toScreen(pose.x, pose.y);
  const a = -pose.yaw;  // screen y is flipped
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.arc(sx, sy, FOV_RANGE_M * S.view.scale, a - CAMERA_HFOV / 2, a + CAMERA_HFOV / 2);
  ctx.closePath();
  ctx.fillStyle = 'rgba(88, 166, 255, 0.12)';
  ctx.fill();
  ctx.save();
  ctx.translate(sx, sy);
  ctx.rotate(a);
  ctx.beginPath();
  ctx.moveTo(12, 0);
  ctx.lineTo(-8, 8);
  ctx.lineTo(-4, 0);
  ctx.lineTo(-8, -8);
  ctx.closePath();
  ctx.fillStyle = '#58a6ff';
  ctx.fill();
  ctx.restore();
}

function drawDetections(dets) {
  ctx.font = '600 12px system-ui, sans-serif';
  ctx.textAlign = 'left';
  for (const d of dets) {
    if (d.x === undefined) continue;
    if (d.excluded && !S.showExcluded) continue;
    const [sx, sy] = toScreen(d.x, d.y);
    const color = d.excluded ? '#7d8594' : (PETS.has(d.cls) ? '#4fa3ff' : '#34c77b');
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#0d1015';
    ctx.stroke();
    const label = `${d.cls} ${d.dist.toFixed(2)} m`;
    ctx.fillStyle = 'rgba(13, 16, 21, 0.75)';
    ctx.fillRect(sx + 10, sy - 9, ctx.measureText(label).width + 8, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, sx + 14, sy + 4);
  }
}

function drawNav(d) {
  if (S.obstacles.length) {
    const s = Math.max(2, 0.05 * S.view.scale);
    ctx.fillStyle = 'rgba(255, 77, 77, 0.85)';
    for (const [x, y] of S.obstacles) {
      const [sx, sy] = toScreen(x, y);
      ctx.fillRect(sx - s / 2, sy - s / 2, s, s);
    }
  }
  const navigating = (d.nav && d.nav.active) || d.nav_goal;
  if (navigating && S.plan.length > 1) {
    polyPath(S.plan);
    ctx.strokeStyle = '#b98cff';
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  if (d.nav_goal) {
    const [gx, gy] = toScreen(d.nav_goal.x, d.nav_goal.y);
    ctx.beginPath();
    ctx.arc(gx, gy, 10, 0, Math.PI * 2);
    ctx.strokeStyle = '#b98cff';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(gx, gy, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#b98cff';
    ctx.fill();
  }
}

// The explorer's goal and home from its status (explore: {goal: [x, y], home: [x, y], ...}).
function drawExplore(e) {
  if (!e) return;
  ctx.font = '600 12px system-ui, sans-serif';
  ctx.textAlign = 'left';
  if (Array.isArray(e.home)) {
    const [hx, hy] = toScreen(e.home[0], e.home[1]);
    ctx.fillStyle = 'rgba(13, 16, 21, 0.75)';
    ctx.fillRect(hx - 8, hy - 8, 16, 16);
    ctx.strokeStyle = '#ffb000';
    ctx.lineWidth = 2;
    ctx.strokeRect(hx - 8, hy - 8, 16, 16);
    ctx.fillStyle = '#ffb000';
    ctx.textAlign = 'center';
    ctx.fillText('H', hx, hy + 4);
    ctx.textAlign = 'left';
  }
  if (Array.isArray(e.goal)) {
    const [gx, gy] = toScreen(e.goal[0], e.goal[1]);
    ctx.beginPath();
    ctx.moveTo(gx, gy - 10); ctx.lineTo(gx + 10, gy); ctx.lineTo(gx, gy + 10); ctx.lineTo(gx - 10, gy);
    ctx.closePath();
    ctx.strokeStyle = '#ffb000';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.fillStyle = '#ffb000';
    ctx.fillText('explore', gx + 13, gy + 4);
  }
}

function render() {
  try {
    const d0 = S.data || {};
    // Three-view: keep the robot in the middle unless the map was moved by hand recently.
    if (document.body.dataset.tab === 'mfd' && d0.pose && Date.now() - S.userViewAt > 15000
        && (Math.abs(S.view.x - d0.pose.x) > 0.01 || Math.abs(S.view.y - d0.pose.y) > 0.01)) {
      S.view.x = d0.pose.x; S.view.y = d0.pose.y; S.dirty = true;
    }
    if (S.dirty) {
      S.dirty = false;
      const r = W();
      ctx.clearRect(0, 0, r.width, r.height);
      drawGrid(r);
      drawMap();
      drawPath();
      drawZones();
      const d = S.data || {};
      drawNav(d);
      drawExplore(d.explore);
      drawCamera(d.pose);
      drawDetections(d.detections || []);
      if (!S.map) {
        ctx.fillStyle = '#8c95a6';
        ctx.textAlign = 'center';
        ctx.font = '14px system-ui, sans-serif';
        ctx.fillText('Waiting for the map… move the camera to start mapping', r.width / 2, 24);
      }
    }
  } catch (e) {
    console.error('map render failed', e);
  } finally {
    requestAnimationFrame(render);
  }
}

// ---------- interaction: pan / zoom / draw ----------
const pointers = new Map();
let gesture = null;

function zoomAt(sx, sy, factor) {
  S.userViewAt = Date.now();
  const [wx, wy] = toWorld(sx, sy);
  S.view.scale = Math.min(400, Math.max(5, S.view.scale * factor));
  const [nx, ny] = toWorld(sx, sy);
  S.view.x += wx - nx;
  S.view.y += wy - ny;
  S.dirty = true;
}

function localXY(ev) { const r = W(); return [ev.clientX - r.left, ev.clientY - r.top]; }

canvas.addEventListener('pointerdown', (ev) => {
  try { canvas.setPointerCapture(ev.pointerId); } catch (e) { /* not capturable (synthetic or cancelled) */ }
  pointers.set(ev.pointerId, localXY(ev));
  if (pointers.size === 1) gesture = { start: localXY(ev), last: localXY(ev), moved: 0, touch: ev.pointerType !== 'mouse' };
  else if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    gesture = { pinch: Math.hypot(a[0] - b[0], a[1] - b[1]), moved: 99 };
  }
});

canvas.addEventListener('pointermove', (ev) => {
  if (!pointers.has(ev.pointerId)) return;
  pointers.set(ev.pointerId, localXY(ev));
  if (pointers.size === 2 && gesture && gesture.pinch) {
    const [a, b] = [...pointers.values()];
    const dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
    zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, dist / gesture.pinch);
    gesture.pinch = dist;
  } else if (pointers.size === 1 && gesture && gesture.last) {
    const p = localXY(ev);
    S.userViewAt = Date.now();
    S.view.x -= (p[0] - gesture.last[0]) / S.view.scale;
    S.view.y += (p[1] - gesture.last[1]) / S.view.scale;
    gesture.moved += Math.hypot(p[0] - gesture.last[0], p[1] - gesture.last[1]);
    gesture.last = p;
    S.dirty = true;
  }
});

function endPointer(ev) {
  if (!pointers.has(ev.pointerId)) return;
  pointers.delete(ev.pointerId);
  // Fingers wobble more than a mouse: allow more movement before a tap becomes a pan.
  const tapSlop = gesture && gesture.touch ? 12 : 6;
  if (pointers.size === 0 && gesture && gesture.moved < tapSlop) {
    if (S.goalMode) setGoal(gesture.start);
    else if (S.drawing) addCorner(gesture.start, gesture.touch);
  }
  if (pointers.size === 0) gesture = null;
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);

canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const [sx, sy] = localXY(ev);
  zoomAt(sx, sy, Math.exp(-ev.deltaY * 0.0015));
}, { passive: false });

function addCorner([sx, sy], touch) {
  if (S.naming) return;
  if (S.draft.length >= 3) {
    const [fx, fy] = toScreen(...S.draft[0]);
    if (Math.hypot(sx - fx, sy - fy) < (touch ? 28 : 14)) { finishZone(); return; }
  }
  if (S.draft.length >= 64) return;
  const [x, y] = toWorld(sx, sy);
  S.draft.push([Number(x.toFixed(3)), Number(y.toFixed(3))]);
  updateToolbar();
  S.dirty = true;
}

function setDrawing(on, kind) {
  S.drawing = on;
  S.drawKind = kind || 'exclusion';
  S.naming = false;
  S.draft = [];
  if (on) S.goalMode = false;
  updateToolbar();
  S.dirty = true;
}

function updateToolbar() {
  const placing = S.drawing && !S.naming;
  canvas.classList.toggle('drawing', S.drawing || S.goalMode);
  $('btn-goal').hidden = S.drawing;
  $('btn-goal').textContent = S.goalMode ? '✖ Cancel goal' : '🎯 Go to…';
  $('goal-hint').hidden = !S.goalMode;
  $('btn-draw').hidden = S.drawing || S.goalMode;
  $('btn-draw-keepout').hidden = S.drawing || S.goalMode;
  $('btn-finish').hidden = !placing;
  $('btn-finish').disabled = S.draft.length < 3;
  $('btn-undo').hidden = !placing;
  $('btn-undo').disabled = !S.draft.length;
  $('btn-cancel').hidden = !S.drawing;
  $('draw-hint').hidden = !placing;
  $('name-form').hidden = !S.naming;
}

function showMsg(text) {
  const el = $('map-msg');
  el.textContent = text;
  el.hidden = !text;
}

function defaultZoneName() {
  const prefix = S.drawKind === 'keepout' ? 'keepout_' : 'zone_';
  let i = 1;
  while (S.zones.some((z) => z.name === prefix + i)) i++;
  return prefix + i;
}

function finishZone() {
  if (S.draft.length < 3) return;
  S.naming = true;
  updateToolbar();
  const input = $('zone-name');
  input.value = defaultZoneName();
  input.focus();
  input.select();
}

async function saveZone(ev) {
  ev.preventDefault();
  const name = $('zone-name').value.trim() || defaultZoneName();
  $('btn-save').disabled = true;
  try {
    await api('POST', '/api/zones', { name, points: S.draft, kind: S.drawKind });
    showMsg('');
    setDrawing(false);
  } catch (e) {
    showMsg('Could not save zone: ' + e.message);
  } finally {
    $('btn-save').disabled = false;
  }
}

$('btn-draw').onclick = () => { showMsg(''); setDrawing(true, 'exclusion'); };
$('btn-draw-keepout').onclick = () => { showMsg(''); setDrawing(true, 'keepout'); };
$('btn-goal').onclick = () => { showMsg(''); S.goalMode = !S.goalMode; updateToolbar(); };
$('btn-stop-nav').onclick = async () => {
  try { await api('POST', '/api/goal/cancel'); } catch (e) { showMsg(e.message); }
};
$('btn-voice').onclick = () => {
  S.voice = !S.voice;
  $('btn-voice').textContent = S.voice ? '🔊 Voice on' : '🔇 Voice off';
  if (S.voice && 'speechSynthesis' in window) {
    S.lastSpoken = null;  // speak the current instruction; this tap also unlocks audio on phones
    speechSynthesis.speak(new SpeechSynthesisUtterance('Voice on'));
  }
};
$('btn-cancel').onclick = () => setDrawing(false);
$('btn-finish').onclick = finishZone;
$('name-form').onsubmit = saveZone;
$('btn-back').onclick = () => { S.naming = false; updateToolbar(); };
$('btn-undo').onclick = () => { S.draft.pop(); updateToolbar(); S.dirty = true; };
$('btn-center').onclick = () => {
  const p = S.data && S.data.pose;
  if (p) { S.view.x = p.x; S.view.y = p.y; S.dirty = true; }
};
$('show-excluded').onchange = (ev) => { S.showExcluded = ev.target.checked; renderDetections(); S.dirty = true; };
document.addEventListener('keydown', (ev) => {
  if (S.goalMode && ev.key === 'Escape') { S.goalMode = false; updateToolbar(); return; }
  if (!S.drawing || S.naming) return;
  if (ev.key === 'Escape') setDrawing(false);
  else if (ev.key === 'Enter') finishZone();
  else if (ev.key === 'Backspace') { S.draft.pop(); updateToolbar(); S.dirty = true; }
});

// ---------- robot: status + drive pad ----------
const GOAL_HINT_PERSON = $('goal-hint').textContent;
const R = { v: 0, w: 0, active: false, timer: null, pending: false, keys: new Set(), canDrive: false };

function renderRobot(d) {
  const r = d.robot;
  $('robot-panel').hidden = !r;
  $('pill-robot').hidden = !r;
  $('goal-hint').textContent = r ? 'Tap the map where the robot should go.' : GOAL_HINT_PERSON;
  if (!r) { R.canDrive = false; stopDrive(); return; }
  const b = r.battery || {};
  const pct = b.percent != null ? `${b.percent}%` : `${b.voltage} V`;
  setPill('pill-robot', `🔋 ${pct}${b.charging ? ' ⚡' : ''}`, b.percent != null && b.percent < 20 ? 'warn' : 'ok');
  const control = r.mode === 'safe' || r.mode === 'full';
  $('robot-status').textContent = `· ${control ? 'ready to drive' : r.mode + ' mode'}`;
  $('btn-robot-enable').hidden = control;
  const alerts = [...r.hazards];
  if (!control && r.mode === 'passive') alerts.push('Driving is off (the robot was lifted, hit a cliff, docked, or just started)');
  if (r.message) alerts.push(r.message);
  $('robot-alert').textContent = alerts.join(' · ');
  $('robot-alert').hidden = !alerts.length;
  R.canDrive = control;
  $('drive-pad').classList.toggle('disabled', !control);
  if (!control) stopDrive();
  renderExplore(d.explore, control, d.mode === 'mapping');
}

function renderExplore(e, control, mapping) {
  $('explore-row').hidden = !e;
  if (!e) return;
  const active = e.state === 'exploring' || e.state === 'returning';
  $('btn-explore').hidden = active;
  $('btn-explore').disabled = !control || !mapping;
  $('btn-explore-stop').hidden = !active;
  let text = e.message || '';
  if (e.reached || e.failed) text += ` (${e.reached} reached, ${e.failed} skipped)`;
  if (!mapping && !active) text = 'Exploring adds to the map, so it needs mapping mode: run ~/run_robot.sh mapping';
  $('explore-status').textContent = text;
}

async function exploreAction(action) {
  try { await api('POST', '/api/explore', { action }); } catch (e) {
    $('robot-alert').textContent = e.message; $('robot-alert').hidden = false;
  }
}
$('btn-explore').onclick = () => exploreAction('start');
$('btn-explore-stop').onclick = () => exploreAction('stop');

$('btn-grid').onclick = async () => {
  const on = !(S.data && S.data.floor_grid);
  try { await api('POST', '/api/floor_grid', { on }); } catch (e) { showMsg(e.message); }
};
function renderGrid(d) {
  $('btn-grid').textContent = d.floor_grid ? '▦ Grid on' : '▦ Grid off';
}

function sendDrive() {
  if (R.pending) return;   // never queue up stale commands
  R.pending = true;
  api('POST', '/api/drive', { v: R.v, w: R.w })
    .catch((e) => { $('robot-alert').textContent = e.message; $('robot-alert').hidden = false; })
    .finally(() => { R.pending = false; });
}

function setDrive(nx, ny) {
  // nx, ny in -1..1 (right / up positive), with a small dead zone.
  const dz = (a) => (Math.abs(a) < 0.15 ? 0 : (a - Math.sign(a) * 0.15) / 0.85);
  const r = S.data && S.data.robot;
  const maxV = r ? r.max_speed : 0.2;
  const maxW = r ? r.max_turn : 1.0;
  R.v = Number((dz(ny) * maxV).toFixed(3));
  R.w = Number((-dz(nx) * maxW).toFixed(3));
  if (!R.active) {
    R.active = true;
    sendDrive();
    R.timer = setInterval(sendDrive, 100);
  }
}

function stopDrive() {
  if (!R.active) return;
  R.active = false;
  clearInterval(R.timer);
  R.v = 0; R.w = 0;
  R.pending = false;
  sendDrive();
  $('drive-knob').style.transform = '';
}

const pad = $('drive-pad');
function padMove(ev) {
  const rect = pad.getBoundingClientRect();
  const rad = rect.width / 2;
  let dx = ev.clientX - (rect.left + rad);
  let dy = ev.clientY - (rect.top + rad);
  const n = Math.hypot(dx, dy);
  if (n > rad) { dx *= rad / n; dy *= rad / n; }
  $('drive-knob').style.transform = `translate(${dx}px, ${dy}px)`;
  setDrive(dx / rad, -dy / rad);
}
pad.addEventListener('pointerdown', (ev) => {
  if (!R.canDrive) return;
  ev.preventDefault();
  try { pad.setPointerCapture(ev.pointerId); } catch (e) { /* not capturable */ }
  padMove(ev);
});
pad.addEventListener('pointermove', (ev) => { if (R.active && !R.keys.size) padMove(ev); });
['pointerup', 'pointercancel', 'lostpointercapture'].forEach((t) => pad.addEventListener(t, stopDrive));

const DRIVE_KEYS = { ArrowUp: [0, 1], ArrowDown: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
function keyDrive() {
  if (!R.keys.size) { stopDrive(); return; }
  let x = 0; let y = 0;
  R.keys.forEach((k) => { x += DRIVE_KEYS[k][0]; y += DRIVE_KEYS[k][1]; });
  setDrive(x * 0.7, y);
}
document.addEventListener('keydown', (ev) => {
  if (!(ev.key in DRIVE_KEYS) || !R.canDrive || ev.target.tagName === 'INPUT') return;
  ev.preventDefault();
  R.keys.add(ev.key);
  keyDrive();
});
document.addEventListener('keyup', (ev) => { if (R.keys.delete(ev.key)) keyDrive(); });
window.addEventListener('blur', () => { R.keys.clear(); stopDrive(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) { R.keys.clear(); stopDrive(); } });

$('btn-robot-stop').onclick = async () => {
  R.keys.clear();
  stopDrive();
  try { await api('POST', '/api/robot/stop'); } catch (e) { showMsg(e.message); }
};
$('btn-robot-enable').onclick = async () => {
  $('btn-robot-enable').disabled = true;
  try { await api('POST', '/api/robot/mode', { mode: 'safe' }); } catch (e) {
    $('robot-alert').textContent = e.message; $('robot-alert').hidden = false;
  } finally { setTimeout(() => { $('btn-robot-enable').disabled = false; }, 1500); }
};

// ---------- tabs / layouts + camera and depth stream control ----------
// Tabs: camera, map, depth (one view on a phone; a wide screen shows all three with their
// controls) and mfd: the three views only, side by side or stacked (?layout=mfd opens it; the
// choice is kept for this browser tab). A frame named roomba-mfd (the control page's display)
// opens in it too: the name survives the login redirect, which drops ?layout.
const TABS = ['camera', 'map', 'depth', 'mfd'];
const phone = window.matchMedia('(max-width: 900px)');
const cam = $('camera');
const depth = $('depth');
let camPaused = false;
let camOn = false;
let depthPaused = false;
let depthOn = false;
let depthAvailable = null;   // false: this server has no /depth.mjpg (the Python web server)

function tab() { return document.body.dataset.tab; }

function cameraVisible() {
  return !document.hidden && !camPaused && !viewCollapsed('camera') && (!phone.matches || tab() === 'camera' || tab() === 'mfd');
}

function depthVisible() {
  return !document.hidden && !depthPaused && depthAvailable !== false && !viewCollapsed('depth')
    && (!phone.matches || tab() === 'depth' || tab() === 'mfd');
}

// Only stream while the camera is on screen: saves ~0.4 MB/s and phone battery.
function updateStream() {
  const want = cameraVisible();
  if (want && !camOn) cam.src = '/stream.mjpg?t=' + Date.now();
  if (!want && camOn) cam.removeAttribute('src');
  camOn = want;
  $('camera-paused').hidden = !camPaused;
  $('btn-pause').textContent = camPaused ? '▶ Resume' : '⏸ Pause';
}

function depthMessage(text) {
  $('depth-msg').textContent = text;
  $('depth-msg').hidden = !text;
}

// The robot encodes depth frames only while someone watches /depth.mjpg.
function updateDepth() {
  const want = depthVisible();
  if (want && !depthOn) { depth.src = '/depth.mjpg?t=' + Date.now(); depthMessage('Waiting for depth…'); }
  if (!want && depthOn) depth.removeAttribute('src');
  depthOn = want;
  if (depthAvailable === false) depthMessage('The depth view needs the Rust web server (rust_nodes:=true).');
  else if (depthPaused) depthMessage('Depth paused');
  $('btn-depth-pause').textContent = depthPaused ? '▶ Resume' : '⏸ Pause';
}

// Turbo colours as the server draws them: red (near) to blue (far).
function turbo(x) {
  const p = (c) => c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * c[5]))));
  const v = [
    p([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943]),
    p([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604]),
    p([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]),
  ].map((c) => Math.round(Math.min(1, Math.max(0, c)) * 255));
  return `rgb(${v.join(',')})`;
}

async function loadDepthInfo() {
  const stops = [];
  for (let i = 0; i <= 10; i++) stops.push(`${turbo(0.9 - 0.08 * i)} ${i * 10}%`);
  $('depth-bar').style.background = `linear-gradient(to right, ${stops.join(', ')})`;
  try {
    const res = await fetch('/depth.mjpg', { method: 'HEAD', credentials: 'same-origin' });
    if (res.status === 404) { depthAvailable = false; updateDepth(); return; }
    depthAvailable = true;
    const range = (res.headers.get('X-Depth-Range') || '').split('-').map(Number);
    if (range.length === 2 && range.every((v) => v > 0)) {
      $('depth-near').textContent = `${range[0] / 1000} m`;
      $('depth-far').textContent = `${range[1] / 1000} m`;
    }
  } catch (e) { /* offline: the stream's error handler retries */ }
}

function setTab(t, persist = true) {
  document.body.dataset.tab = t;
  try {
    sessionStorage.setItem('roomba.tab', t);
    if (persist) localStorage.setItem('roomba.tab', t);
  } catch (e) { /* private mode */ }
  $('btn-layout').textContent = t === 'mfd' ? '☰ Full page' : '▦ 3-view';
  mfdApply();
}

document.querySelectorAll('.tabs button').forEach((b) => { b.onclick = () => setTab(b.dataset.tab); });
$('btn-layout').onclick = () => setTab(tab() === 'mfd' ? 'camera' : 'mfd');
$('btn-pause').onclick = () => { camPaused = !camPaused; updateStream(); };
$('btn-depth-pause').onclick = () => { depthPaused = !depthPaused; updateDepth(); };
document.addEventListener('visibilitychange', () => { updateStream(); updateDepth(); });
phone.addEventListener('change', mfdApply);

// Reload the camera stream if it stalls (e.g. after the robot restarts).
cam.onerror = () => {
  if (!camOn) return;
  camOn = false;
  setTimeout(updateStream, 2000);
};
depth.onload = () => { if (depthOn) depthMessage(''); };
depth.onerror = async () => {
  if (!depthOn) return;
  depthOn = false;
  await loadDepthInfo();
  setTimeout(updateDepth, 2000);
};

// ---------- three-view: draggable dividers, collapsible views, saved per device ----------
// Wide screen: the views sit side by side; drag a divider to share the width between its two
// neighbours. Phone: they stack; drag a divider to set the height of the view above it. Tap a
// view's title to collapse it (a collapsed camera or depth view stops its stream). Double-click
// a divider to reset it; "Reset views" resets everything.
const MFD_KEY = 'roomba.mfd.v1';
const VIEWS = ['camera', 'map', 'depth'];
const WRAPS = { camera: '.camera-wrap', map: '.map-wrap', depth: '.depth-wrap' };
let MFD = mfdLoad();

function mfdDefault() { return { grow: { camera: 1, map: 1, depth: 1 }, h: {}, collapsed: {} }; }

function mfdLoad() {
  try {
    const m = JSON.parse(localStorage.getItem(MFD_KEY));
    if (m && m.grow && m.h && m.collapsed) return m;
  } catch (e) { /* none saved, or private mode */ }
  return mfdDefault();
}

function mfdSave() {
  try { localStorage.setItem(MFD_KEY, JSON.stringify(MFD)); } catch (e) { /* private mode */ }
}

function viewSec(v) { return $('sec-' + v); }
function viewWrap(v) { return viewSec(v).querySelector(WRAPS[v]); }
function viewTitle(v) { return viewSec(v).querySelector('h2'); }
function viewCollapsed(v) { return tab() === 'mfd' && !!MFD.collapsed[v]; }

function mfdApply() {
  const on = tab() === 'mfd';
  for (const v of VIEWS) {
    const s = viewSec(v);
    const w = viewWrap(v);
    s.classList.toggle('collapsed', on && !!MFD.collapsed[v]);
    s.style.flexGrow = on && !phone.matches ? String(MFD.grow[v] || 1) : '';
    const h = on && phone.matches ? MFD.h[v] : null;
    w.style.height = h ? h + 'px' : '';
    w.classList.toggle('sized', !!h);
    const title = viewTitle(v);
    if (on) {
      title.setAttribute('role', 'button');
      title.tabIndex = 0;
      title.setAttribute('aria-expanded', String(!MFD.collapsed[v]));
    } else {
      title.removeAttribute('role');
      title.removeAttribute('tabindex');
      title.removeAttribute('aria-expanded');
    }
  }
  updateStream();
  updateDepth();
  resize();
  postHeight();
}

function toggleView(v) {
  MFD.collapsed[v] = !MFD.collapsed[v];
  mfdSave();
  mfdApply();
}

// The nearest open view on one side of divider i (between VIEWS[i] and VIEWS[i + 1]).
function openView(i, dir) {
  for (let k = dir < 0 ? i : i + 1; k >= 0 && k < VIEWS.length; k += dir) {
    if (!MFD.collapsed[VIEWS[k]]) return VIEWS[k];
  }
  return null;
}

// Wide: move `dx` pixels of width from the right neighbour to the left one (flex-grow shares).
function shareWidth(a, b, wa, wb, dx) {
  const g = (MFD.grow[a] || 1) + (MFD.grow[b] || 1);
  const na = Math.max(140, Math.min(wa + wb - 140, wa + dx));
  MFD.grow[a] = g * na / (wa + wb);
  MFD.grow[b] = g - MFD.grow[a];
  viewSec(a).style.flexGrow = String(MFD.grow[a]);
  viewSec(b).style.flexGrow = String(MFD.grow[b]);
}

// Phone: the height of the view above the divider.
function setViewHeight(v, h) {
  MFD.h[v] = Math.round(Math.max(100, Math.min(h, 1600)));
  viewWrap(v).style.height = MFD.h[v] + 'px';
  viewWrap(v).classList.add('sized');
}

function initMfd() {
  for (const v of VIEWS) {
    const title = viewTitle(v);
    title.addEventListener('click', () => { if (tab() === 'mfd') toggleView(v); });
    title.addEventListener('keydown', (e) => {
      if (tab() === 'mfd' && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); toggleView(v); }
    });
  }
  document.querySelectorAll('.mfd-split').forEach((sp) => {
    const i = Number(sp.dataset.split);
    sp.addEventListener('pointerdown', (ev) => {
      if (tab() !== 'mfd' || ev.button !== 0) return;
      const a = openView(i, -1);
      const b = openView(i, 1);
      if (!a || (!phone.matches && !b)) return;
      ev.preventDefault();
      try { sp.setPointerCapture(ev.pointerId); } catch (e) { /* synthetic */ }
      const s = {
        x: ev.clientX, y: ev.clientY,
        wa: viewSec(a).getBoundingClientRect().width, wb: b ? viewSec(b).getBoundingClientRect().width : 0,
        ga: MFD.grow[a] || 1, gb: b ? MFD.grow[b] || 1 : 0,
        h: viewWrap(a).getBoundingClientRect().height,
      };
      sp.classList.add('dragging');
      document.body.classList.add('resizing');
      const move = (e) => {
        if (phone.matches) setViewHeight(a, s.h + e.clientY - s.y);
        else {
          MFD.grow[a] = s.ga;
          MFD.grow[b] = s.gb;
          shareWidth(a, b, s.wa, s.wb, e.clientX - s.x);
        }
      };
      const up = () => {
        sp.removeEventListener('pointermove', move);
        sp.removeEventListener('pointerup', up);
        sp.removeEventListener('pointercancel', up);
        sp.classList.remove('dragging');
        document.body.classList.remove('resizing');
        mfdSave();
        postHeight();
      };
      sp.addEventListener('pointermove', move);
      sp.addEventListener('pointerup', up);
      sp.addEventListener('pointercancel', up);
    });
    sp.addEventListener('dblclick', () => {
      const a = openView(i, -1);
      const b = openView(i, 1);
      if (phone.matches) { if (a) delete MFD.h[a]; } else { if (a) MFD.grow[a] = 1; if (b) MFD.grow[b] = 1; }
      mfdSave();
      mfdApply();
    });
    sp.addEventListener('keydown', (e) => {
      const keys = phone.matches ? { ArrowDown: 20, ArrowUp: -20 } : { ArrowRight: 20, ArrowLeft: -20 };
      const step = keys[e.key];
      const a = openView(i, -1);
      const b = openView(i, 1);
      if (!step || !a || (!phone.matches && !b)) return;
      e.preventDefault();
      if (phone.matches) setViewHeight(a, viewWrap(a).getBoundingClientRect().height + step);
      else shareWidth(a, b, viewSec(a).getBoundingClientRect().width, viewSec(b).getBoundingClientRect().width, step);
      mfdSave();
      postHeight();
    });
  });
  $('btn-mfd-reset').onclick = mfdReset;
  // In the control page's display: its RESET LAYOUT resets the views too.
  window.addEventListener('message', (e) => {
    if (e.source === window.parent && e.data && e.data.type === 'roomba-mfd-reset') mfdReset();
  });
  if ('ResizeObserver' in window) new ResizeObserver(postHeight).observe(document.querySelector('main.grid'));
}

function mfdReset() {
  MFD = mfdDefault();
  mfdSave();
  mfdApply();
}

// Inside a frame (the control page), report the content height so a phone layout can show it
// all without a scrollbar inside the frame.
function postHeight() {
  if (window.parent === window) return;
  const g = document.querySelector('main.grid');
  const height = Math.ceil(g.getBoundingClientRect().bottom + window.scrollY + 12);
  window.parent.postMessage({ type: 'roomba-mfd-height', height }, '*');
}

let savedTab = 'camera';
try {
  const embedded = window.name === 'roomba-mfd' && !sessionStorage.getItem('roomba.tab');
  if (new URLSearchParams(location.search).get('layout') === 'mfd' || embedded) sessionStorage.setItem('roomba.tab', 'mfd');
  savedTab = sessionStorage.getItem('roomba.tab') || localStorage.getItem('roomba.tab') || 'camera';
} catch (e) { /* private mode */ }
initMfd();
setTab(TABS.includes(savedTab) ? savedTab : 'camera', false);
loadDepthInfo().then(updateDepth);
renderZones();
renderDetections();
connect();
requestAnimationFrame(render);