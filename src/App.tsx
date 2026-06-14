// @ts-nocheck
import React, { useRef, useEffect, useState, useCallback } from "react";

/* ============================================================================
 * SwarmLab — эффект "роя", упаковывающегося вокруг защищённого силуэта.
 *
 * Архитектура (всё в одном файле для артефакта, но разнесено по модулям-классам,
 * чтобы было удобно выносить в отдельные файлы при развитии):
 *
 *   DistanceField  — строит знаковое поле расстояний (SDF) из ЧБ-маски.
 *                    Даёт sample(x,y) -> { d, gx, gy }: знаковое расстояние до
 *                    края защищённой зоны и градиент (направление наружу).
 *   SpatialHash    — равномерная сетка для коллизий между мелкими объектами O(n).
 *   Particle       — состояние одного мелкого объекта.
 *   Simulation     — частицы + поле + трансформ главного объекта + конфиг; step().
 *   Renderer       — рисует фон, главный объект (visual ∩ mask), частицы, debug.
 *   <SwarmLab/>    — React-обёртка: панель параметров, загрузка файлов, RAF-цикл.
 *
 * Координаты: всё в "мировых" пикселях канваса. Поле SDF считается один раз в
 * локальном пространстве маски; при семплинге точка переводится world->local
 * через обратный трансформ главного объекта (позиция/масштаб/поворот).
 * ==========================================================================*/

/* ----------------------------- утилиты математики ------------------------ */
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const TAU = Math.PI * 2;
function angLerp(a, b, t) {
  let d = ((b - a + Math.PI) % TAU) - Math.PI;
  if (d < -Math.PI) d += TAU;
  return a + d * t;
}
// радиус ориентированного эллипса (полуоси a вдоль angle, b поперёк) в направлении (dx,dy)
function ellipseRadius(a, b, dx, dy, angle) {
  const ca = Math.cos(angle),
    sa = Math.sin(angle);
  const cphi = dx * ca + dy * sa; // проекция на главную ось
  const sphi = -dx * sa + dy * ca; // проекция на малую ось
  const t = (cphi * cphi) / (a * a) + (sphi * sphi) / (b * b);
  return 1 / Math.sqrt(t > 1e-9 ? t : 1e-9);
}

/* =============================== DistanceField =========================== */
// 1D EDT (Felzenszwalb & Huttenlocher), возвращает квадраты расстояний.
function edt1d(f) {
  const n = f.length;
  const d = new Float64Array(n);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
  return d;
}
// 2D EDT по бинарной маске feature(1)/empty(0): расстояние до ближайшего feature.
function edt2d(binary, w, h) {
  const INF = 1e12;
  const grid = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) grid[i] = binary[i] ? 0 : INF;
  const col = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) col[y] = grid[y * w + x];
    const d = edt1d(col);
    for (let y = 0; y < h; y++) grid[y * w + x] = d[y];
  }
  const row = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) row[x] = grid[y * w + x];
    const d = edt1d(row);
    for (let x = 0; x < w; x++) grid[y * w + x] = Math.sqrt(d[x]);
  }
  return grid;
}

class DistanceField {
  // maskCanvas: canvas с ЧБ/альфа маской. Белое/непрозрачное = защищённая зона.
  constructor(maskCanvas, res = 220, threshold = 0.5) {
    const mw = maskCanvas.width,
      mh = maskCanvas.height;
    const scale = res / Math.max(mw, mh);
    const fw = Math.max(8, Math.round(mw * scale));
    const fh = Math.max(8, Math.round(mh * scale));
    const c = document.createElement("canvas");
    c.width = fw;
    c.height = fh;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(maskCanvas, 0, 0, fw, fh);
    const img = ctx.getImageData(0, 0, fw, fh).data;

    const inside = new Uint8Array(fw * fh); // 1 = защищённая зона
    for (let i = 0; i < fw * fh; i++) {
      const a = img[i * 4 + 3] / 255;
      const lum =
        (0.299 * img[i * 4] + 0.587 * img[i * 4 + 1] + 0.114 * img[i * 4 + 2]) /
        255;
      // непрозрачное И светлое => зона. (поддерживает и альфа-силуэт, и ЧБ)
      inside[i] = a > 0.5 && lum > threshold ? 1 : 0;
    }
    const outside = new Uint8Array(fw * fh);
    for (let i = 0; i < fw * fh; i++) outside[i] = inside[i] ? 0 : 1;

    const distToInside = edt2d(inside, fw, fh); // >0 для точек вне зоны
    const distToOutside = edt2d(outside, fw, fh); // >0 для точек внутри зоны

    // знаковое поле: +снаружи, -внутри (в пикселях поля)
    const sdf = new Float32Array(fw * fh);
    for (let i = 0; i < fw * fh; i++)
      sdf[i] = inside[i] ? -distToOutside[i] : distToInside[i];

    this.sdf = sdf;
    this.fw = fw;
    this.fh = fh;
    this.maskW = mw;
    this.maskH = mh;
    this.pxToWorld = mw / fw; // поле -> локальные пиксели маски (масштаб 1:1 по осям)
  }

  // билинейный семпл знакового расстояния в координатах поля
  sampleRaw(fx, fy) {
    const { sdf, fw, fh } = this;
    fx = clamp(fx, 0, fw - 1.001);
    fy = clamp(fy, 0, fh - 1.001);
    const x0 = fx | 0,
      y0 = fy | 0;
    const tx = fx - x0,
      ty = fy - y0;
    const i = y0 * fw + x0;
    const a = sdf[i],
      b = sdf[i + 1],
      c = sdf[i + fw],
      d = sdf[i + fw + 1];
    return lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
  }
}

/* =============================== SpatialHash ============================= */
class SpatialHash {
  constructor(cell) {
    this.cell = cell;
    this.map = new Map();
  }
  _key(cx, cy) {
    return cx * 73856093 ^ cy * 19349663;
  }
  clear() {
    this.map.clear();
  }
  insert(p) {
    const cx = Math.floor(p.x / this.cell),
      cy = Math.floor(p.y / this.cell);
    const k = this._key(cx, cy);
    let arr = this.map.get(k);
    if (!arr) this.map.set(k, (arr = []));
    arr.push(p);
  }
  forEachNeighbor(p, fn) {
    const cx = Math.floor(p.x / this.cell),
      cy = Math.floor(p.y / this.cell);
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const arr = this.map.get(this._key(cx + dx, cy + dy));
        if (arr) for (let i = 0; i < arr.length; i++) if (arr[i] !== p) fn(arr[i]);
      }
  }
}

/* ================================ Particle =============================== */
let _pid = 0;
class Particle {
  constructor(x, y, dotColor, sizeMul = 1, vfrac = 0.5, eager = 1) {
    this.id = _pid++;
    this.x = x;
    this.y = y;
    this.vx = (Math.random() - 0.5) * 20;
    this.vy = (Math.random() - 0.5) * 20;
    this.angle = Math.random() * TAU;
    this.age = 0;
    this.va = 0;
    this.dot = dotColor;
    this.seed = Math.random();
    this.sizeMul = sizeMul; // случайный множитель размера (рождение)
    this.vfrac = vfrac; // верт. доля высоты в момент рождения (для градиента)
    this.eager = eager; // множитель сноса: часть рыбок рвётся к объекту быстрее
    this.size = 0; // вычисляется в step (для коллизий/рендера)
  }
}

/* =============================== Simulation ============================== */
class Simulation {
  constructor() {
    this.particles = [];
    this.field = null;
    // трансформ главного объекта (центр в мире)
    this.obj = { x: 360, y: 360, scale: 1, rot: 0 };
    this.cfg = null;
    this.hash = new SpatialHash(40);
    this.time = 0; // для когерентной волны по полю
    this._wakeTimer = 0; // время «пробуждения» после движения объекта
    this.pointer = { x: 0, y: 0, inside: false }; // курсор в мире (для «хищника»)
  }

  setField(field) {
    this.field = field;
    // по умолчанию центрируем объект в канвасе при первой установке
  }

  // world -> поле SDF. Возвращает { d, gx, gy } в МИРОВЫХ единицах/направлении.
  // optScale < 1 — семплинг уменьшенной копии (для inner-режима взгляда)
  sampleField(wx, wy, optScale) {
    const f = this.field;
    if (!f) return null;
    const o = this.obj;
    // world -> local маски
    let dx = wx - o.x,
      dy = wy - o.y;
    const cos = Math.cos(-o.rot),
      sin = Math.sin(-o.rot);
    let lx = (dx * cos - dy * sin) / o.scale;
    let ly = (dx * sin + dy * cos) / o.scale;
    // если optScale < 1, семплируем как уменьшенную копию (координаты ближе к центру)
    if (optScale !== undefined && optScale < 1) {
      lx *= optScale;
      ly *= optScale;
    }
    // local (центр) -> пиксели маски -> пиксели поля
    const mx = lx + f.maskW / 2;
    const my = ly + f.maskH / 2;
    const fx = (mx / f.maskW) * f.fw;
    const fy = (my / f.maskH) * f.fh;

    let gx, gy, dField, conf;
    const insideRect =
      fx >= 1 && fx <= f.fw - 2 && fy >= 1 && fy <= f.fh - 2;
    if (insideRect) {
      // внутри прямоугольника поля — точный SDF + градиент центральными разностями
      const e = 1.5;
      dField = f.sampleRaw(fx, fy);
      gx = f.sampleRaw(fx + e, fy) - f.sampleRaw(fx - e, fy);
      gy = f.sampleRaw(fx, fy + e) - f.sampleRaw(fx, fy - e);
      const len = Math.hypot(gx, gy) || 1;
      // у корректного поля |∇|≈2e; падение = медиальная ось/шум => низкая уверенность
      conf = clamp(len / (2 * e), 0, 1);
      gx /= len;
      gy /= len;
    } else {
      // ВНЕ рамки поля — аналитический фолбэк: ближайшая точка рамки + её SDF.
      const cx = clamp(fx, 0, f.fw - 1);
      const cy = clamp(fy, 0, f.fh - 1);
      const ox = fx - cx,
        oy = fy - cy; // вектор наружу от рамки (в пикселях поля)
      const dRect = Math.hypot(ox, oy) || 1e-4;
      gx = ox / dRect; // направление наружу (от зоны)
      gy = oy / dRect;
      dField = f.sampleRaw(cx, cy) + dRect; // суммарное расстояние до силуэта (>0)
      conf = 0; // за рамкой направление к краю ненадёжно -> взгляд на центр
    }
    // повернуть градиент обратно в мир (rot), масштаб направления не меняет
    const c2 = Math.cos(o.rot),
      s2 = Math.sin(o.rot);
    const wgx = gx * c2 - gy * s2;
    const wgy = gx * s2 + gy * c2;
    const dWorld = dField * f.pxToWorld * o.scale;
    return { d: dWorld, gx: wgx, gy: wgy, conf };
  }

  // размер частицы: рост × верт. градиент (заморожен на vfrac рождения) ×
  // случайный множитель. Один размер для коллизий, касания и отрисовки —
  // чтобы рыбки не наползали друг на друга. Заморозка vfrac + мягкий градиент
  // + симметричные коллизии не дают рою всплывать.
  sizeOf(p) {
    const c = this.cfg.particle;
    const grow = clamp(p.age / Math.max(0.01, c.growSeconds), 0, 1);
    const base = lerp(c.spawnSize, c.baseSize, grow);
    const vert = lerp(c.topScale, c.bottomScale, p.vfrac);
    return base * vert * p.sizeMul;
  }

  // граница объекта как жёсткая стенка: если рыбка зашла в зону — выталкиваем
  // ТОЛЬКО позицию (скорость восстановим из перемещения, см. step §3).
  // Решается ВНУТРИ цикла ограничений рядом с коллизиями — поэтому давление
  // переднего ряда расходится по касательной, а не отлетает рывком назад.
  solveWall(p) {
    const s = this.sampleField(p.x, p.y);
    if (!s) return;
    const contact = Math.max(4, this.cfg.object.gap + p.size + 2);
    if (s.d < contact) {
      const pen = contact - s.d;
      p.x += s.gx * pen;
      p.y += s.gy * pen;
    }
  }

  spawn(x, y) {
    const c = this.cfg;
    if (this.particles.length >= c.spawn.maxParticles) return;
    const n = c.spawn.perEmit;
    const r = c.spawn.brushRadius;
    const pal = c.particle.palette;
    const smin = c.particle.sizeRandMin,
      smax = c.particle.sizeRandMax;
    const H = c.canvas.h;
    const burst = c.spawn.burst;
    for (let i = 0; i < n; i++) {
      if (this.particles.length >= c.spawn.maxParticles) break;
      const a = Math.random() * TAU;
      const rr = Math.sqrt(Math.random()) * r;
      const col = pal[(Math.random() * pal.length) | 0];
      const mul = lerp(smin, smax, Math.random());
      const px = x + Math.cos(a) * rr,
        py = y + Math.sin(a) * rr;
      const eager =
        Math.random() < c.physics.eagerFraction ? c.physics.eagerBoost : 1;
      const p = new Particle(px, py, col, mul, clamp(py / H, 0, 1), eager);
      // взрывной разлёт от точки клика
      const v = burst * (0.4 + Math.random() * 0.6);
      p.vx = Math.cos(a) * v;
      p.vy = Math.sin(a) * v;
      this.particles.push(p);
    }
  }

  // заполняет холст плотной равномерной сеткой с запасом 2× за края
  fillCanvas(W, H) {
    const c = this.cfg;
    const pc = c.particle;
    this.clear();
    const maxSize = pc.baseSize * pc.bottomScale * pc.sizeRandMax;
    const spacing = Math.max(8, maxSize * 1.4 * pc.packing);
    const margin = Math.max(W, H) * 0.5;
    const pal = pc.palette;
    for (let yy = -margin; yy < H + margin; yy += spacing) {
      for (let xx = -margin; xx < W + margin; xx += spacing) {
        if (this.particles.length >= c.spawn.maxParticles) return;
        const px = xx + (Math.random() - 0.5) * spacing * 0.25;
        const py = yy + (Math.random() - 0.5) * spacing * 0.25;
        const s = this.sampleField(px, py);
        if (s && s.d < c.object.gap) continue;
        const mul = lerp(pc.sizeRandMin, pc.sizeRandMax, Math.random());
        const pSize = pc.baseSize * mul * lerp(pc.topScale, pc.bottomScale, clamp(py / H, 0, 1));
        let clash = false;
        const existing = this.particles;
        for (let j = existing.length - 1; j >= Math.max(0, existing.length - 20); j--) {
          const q = existing[j];
          if (Math.hypot(px - q.x, py - q.y) < (pSize + q.size) * 0.85) { clash = true; break; }
        }
        if (clash) continue;
        const col = pal[(Math.random() * pal.length) | 0];
        const eager = Math.random() < c.physics.eagerFraction ? c.physics.eagerBoost : 1;
        const p = new Particle(px, py, col, mul, clamp(py / H, 0, 1), eager);
        p.age = pc.growSeconds;
        p.size = pSize;
        p.vx = (Math.random() - 0.5) * 30;
        p.vy = (Math.random() - 0.5) * 30;
        this.particles.push(p);
      }
    }
  }

  erase(x, y, r) {
    const before = this.particles.length;
    this.particles = this.particles.filter(p => Math.hypot(p.x - x, p.y - y) > r);
  }

  shake() {
    const parts = this.particles;
    this._shakeRepel = 0.3; // временно расширяем коллизии на 30%
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      p.vx += (Math.random() - 0.5) * 1200;
      p.vy += (Math.random() - 0.5) * 1200;
      // добавим позиционный разброс — рыбки сразу отлетают друг от друга
      const r = p.size * 0.8;
      p.x += (Math.random() - 0.5) * r;
      p.y += (Math.random() - 0.5) * r;
    }
  }

  push(x, y, r, force) {
    const parts = this.particles;
    const r2 = r * r;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      const dx = p.x - x, dy = p.y - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < r2 && d2 > 1) {
        const d = Math.sqrt(d2);
        const falloff = 1 - d / r;
        p.vx += (dx / d) * force * falloff;
        p.vy += (dy / d) * force * falloff;
      }
    }
  }

  clear() {
    this.particles.length = 0;
  }

  // строит сетку коллизий по текущим позициям
  rebuildHash(pc) {
    const maxHalf = pc.baseSize * pc.bottomScale * Math.max(pc.sizeRandMax, 1);
    this.hash.cell = clamp(maxHalf * 1.5, 24, 120);
    this.hash.clear();
    const parts = this.particles;
    for (let i = 0; i < parts.length; i++) this.hash.insert(parts[i]);
  }

  forEachNeighborFar(p, fn) {
    const r = p.size;
    const halfCells = Math.max(1, Math.ceil(r / this.hash.cell));
    const cx = Math.floor(p.x / this.hash.cell),
      cy = Math.floor(p.y / this.hash.cell);
    for (let dy = -halfCells; dy <= halfCells; dy++)
      for (let dx = -halfCells; dx <= halfCells; dx++) {
        const arr = this.hash.map.get(this.hash._key(cx + dx, cy + dy));
        if (arr) for (let i = 0; i < arr.length; i++) if (arr[i] !== p) fn(arr[i]);
      }
  }

  step(dt, W, H) {
    dt = Math.min(dt, 1 / 30);
    const c = this.cfg;
    const ph = c.physics;
    const pc = c.particle;
    const gap = c.object.gap;
    const wr = pc.widthRatio;
    const parts = this.particles;
    this.time += dt;
    // затухание встряски
    if (this._shakeRepel) this._shakeRepel = Math.max(0, this._shakeRepel - dt * 0.6);
    const wv = c.wave;
    const bh = c.behaviors || {}; // подключаемые режимы поведения
    // режим производительности: «speed» режет повторные перестройки сетки и
    // решение границы до 1 раза за кадр (вместо ×iters) — основная экономия CPU.
    const perf = c.perf || { mode: "quality" };
    const speedMode = perf.mode === "speed" || (perf.mode === "auto" && parts.length > 1400);

    // детекция движения объекта: если двигаем — «пробуждаем» рой на 0.5с
    const objDx = this.obj.x - (this.obj._px || this.obj.x);
    const objDy = this.obj.y - (this.obj._py || this.obj.y);
    this.obj._px = this.obj.x;
    this.obj._py = this.obj.y;
    const objSpeed = Math.hypot(objDx, objDy) / Math.max(dt, 0.001);
    const objVx = objDx / Math.max(dt, 0.001);
    const objVy = objDy / Math.max(dt, 0.001);
    if (objSpeed > 5) this._wakeTimer = 0.5;
    else this._wakeTimer = Math.max(0, this._wakeTimer - dt);
    const isAwake = this._wakeTimer > 0;

    // обновляем размеры
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      p.age += dt;
      p.size = this.sizeOf(p);
    }
    this.rebuildHash(pc); // для проверки «впереди занято» в цикле сил

    // --- 1. силы поля → желаемая скорость → предсказание позиции + ориентация ---
    //     Здесь только «намерение» рыбки: куда она хочет плыть. Реальную позицию
    //     допилит решатель ограничений (§2), а скорость восстановим из неё (§3).
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      // позиция в начале кадра — из неё восстановим скорость после ограничений
      p.px0 = p.x;
      p.py0 = p.y;
      let ax = 0,
        ay = 0;
      const s = this.sampleField(p.x, p.y);
      let faceX = 0,
        faceY = -1;

      // --- соседи: ОДИН проход для blocked + стайности + плотности + мембраны ---
      let blocked = false;
      let alignX = 0, alignY = 0, cohX = 0, cohY = 0, nCount = 0;
      const needFlock = (bh.flock && bh.flock.on) || (bh.density && bh.density.on);
      const memOn = bh.membrane && bh.membrane.on;
      if (s || needFlock || memOn) {
        const pr = p.size;
        const fr = bh.flock ? bh.flock.radius : 70;
        const mr = bh.membrane ? bh.membrane.radius : 0;
        this.forEachNeighborFar(p, (q) => {
          const dx = q.x - p.x, dy = q.y - p.y;
          const d = Math.hypot(dx, dy) || 1;
          if (s && !isAwake && !blocked) {
            const dot = (dx / d) * -s.gx + (dy / d) * -s.gy;
            if (dot > 0.5 && d < (pr + q.size) * ph.blockDist) blocked = true;
          }
          if (needFlock && d < fr) {
            alignX += q.vx; alignY += q.vy;
            cohX += q.x; cohY += q.y;
            nCount++;
          }
          // [membrane] демпфированная пружина к соседям: притяжение ТОЛЬКО за
          // контактом (близких разводят коллизии). Гашение по оси связи убирает
          // хаос — рой держится как тянущаяся плёнка, а не схлопывается.
          if (memOn && d < mr) {
            const rest = (pr + q.size) * pc.packing;
            if (d > rest) {
              const ux = dx / d, uy = dy / d;
              const f = (d - rest) * bh.membrane.stiffness * 18;
              const rvn = (q.vx - p.vx) * ux + (q.vy - p.vy) * uy; // сближение<0
              const damp = rvn * bh.membrane.stiffness * 6;
              ax += ux * (f + damp);
              ay += uy * (f + damp);
            }
          }
        });
      }

      if (s) {
        const maxV = ph.maxSpeed * p.eager; // «жадные» едут быстрее
        let ts = maxV * clamp((s.d - gap - p.size) / ph.slowR, 0, 1);
        if (blocked) ts *= ph.seekBlocked;
        // steer к границе: тянем скорость к желаемой (ts вдоль нормали внутрь).
        // У самого края ts→0, поэтому здесь же мягко гасится подплыв — давить в
        // стенку рыбке больше не нужно, разведение по касательной делает решатель.
        ax += (-s.gx * ts - p.vx) * ph.steer;
        ay += (-s.gy * ts - p.vy) * ph.steer;
        // когерентная волна по полю: соседи колышутся вместе (плавно, не дёргано)
        if (wv.amp > 0) {
          const wf = wv.amp * 10; // усиливаем чтобы было видно
          ax += Math.sin(p.y * wv.scale + this.time * wv.speed) * wf;
          ay += Math.sin(p.x * wv.scale - this.time * wv.speed * 0.9) * wf * 0.7;
        }
        // [swirl] касательное течение вдоль края — сильнее у самой границы
        if (bh.swirl && bh.swirl.on) {
          const band = clamp(1 - (s.d - gap - p.size) / (ph.slowR * 0.9), 0, 1);
          const sw = bh.swirl.strength * maxV * 2.2 * band;
          ax += -s.gy * sw; // касательная (−gy, gx)
          ay += s.gx * sw;
        }
        // [adhere] адгезия: намагничиваем рыбку строго на контактную линию по
        // всему радиусу + гасим скорость «наружу» (анти-отлипание). Получается
        // плотный обмазывающий слой, в отличие от мягкой остановки по умолчанию.
        if (bh.adhere && bh.adhere.on) {
          const off = s.d - (gap + p.size); // >0 = ещё не на линии
          const reach = ph.slowR;
          if (off > -2 && off < reach) {
            const near = 1 - clamp(off / reach, 0, 1); // 1 у самой линии
            const k = bh.adhere.strength;
            ax += -s.gx * off * k * 12; // тянем на поверхность
            ay += -s.gy * off * k * 12;
            const vn = p.vx * s.gx + p.vy * s.gy; // скорость наружу > 0
            if (vn > 0) {
              ax -= s.gx * vn * k * 8 * near;
              ay -= s.gy * vn * k * 8 * near;
            }
          }
        }
        // направление взгляда — всегда к объекту
        const cx = this.obj.x - p.x,
          cy = this.obj.y - p.y;
        const cl = Math.hypot(cx, cy) || 1;
        const ccx = cx / cl,
          ccy = cy / cl;
        if (c.object.faceMode === "center") {
          faceX = ccx;
          faceY = ccy;
        } else if (c.object.faceMode === "inner") {
          // уменьшенная копия: взгляд = lerp(в центр, к границе, innerScale)
          const blend = c.object.innerScale * s.conf;
          const fxe = -s.gx * blend + ccx * (1 - blend);
          const fye = -s.gy * blend + ccy * (1 - blend);
          const fl = Math.hypot(fxe, fye) || 1;
          faceX = fxe / fl;
          faceY = fye / fl;
        } else {
          const w = s.conf;
          let fxe = -s.gx * w + ccx * (1 - w);
          let fye = -s.gy * w + ccy * (1 - w);
          const fl = Math.hypot(fxe, fye) || 1;
          faceX = fxe / fl;
          faceY = fye / fl;
        }
      }

      // [flock] стайность: выравнивание скорости + сплочение к центру группы
      if (needFlock && nCount > 0) {
        const inv = 1 / nCount;
        if (bh.flock && bh.flock.on) {
          ax += (alignX * inv - p.vx) * bh.flock.align * 10;
          ay += (alignY * inv - p.vy) * bh.flock.align * 10;
          ax += (cohX * inv - p.x) * bh.flock.cohesion * 6;
          ay += (cohY * inv - p.y) * bh.flock.cohesion * 6;
        }
        // [density] сброс давления: в давке толкаемся прочь от локального центра
        if (bh.density && bh.density.on && nCount >= bh.density.count) {
          const awayX = p.x - cohX * inv, awayY = p.y - cohY * inv;
          const al = Math.hypot(awayX, awayY) || 1;
          const f = bh.density.strength * (nCount - bh.density.count + 1) * 80;
          ax += (awayX / al) * f;
          ay += (awayY / al) * f;
        }
      }

      // [wander] собственное медленное блуждание курса
      if (bh.wander && bh.wander.on) {
        if (p.wanderAngle === undefined) p.wanderAngle = Math.random() * TAU;
        p.wanderAngle += (Math.random() - 0.5) * dt * 6;
        const wf = bh.wander.strength * ph.maxSpeed * 1.4;
        ax += Math.cos(p.wanderAngle) * wf;
        ay += Math.sin(p.wanderAngle) * wf;
      }

      // [gravity] постоянный снос (стекание / наполнение ёмкости)
      if (bh.gravity && bh.gravity.on) {
        const ga = (bh.gravity.angle * Math.PI) / 180;
        ax += Math.cos(ga) * bh.gravity.strength;
        ay += Math.sin(ga) * bh.gravity.strength;
      }

      // [temp] температура: непрерывная случайная ажитация
      if (bh.temp && bh.temp.on) {
        const tf = bh.temp.strength * ph.maxSpeed * 2.2;
        ax += (Math.random() - 0.5) * tf;
        ay += (Math.random() - 0.5) * tf;
      }

      // [predator] бегство от курсора
      if (bh.predator && bh.predator.on && this.pointer.inside) {
        const dx = p.x - this.pointer.x, dy = p.y - this.pointer.y;
        const d = Math.hypot(dx, dy);
        if (d < bh.predator.radius && d > 0.001) {
          const f = bh.predator.force * (1 - d / bh.predator.radius);
          ax += (dx / d) * f;
          ay += (dy / d) * f;
        }
      }

      // [wake] вихревой след за движущимся объектом
      if (bh.wake && bh.wake.on && objSpeed > 20) {
        const dx = p.x - this.obj.x, dy = p.y - this.obj.y;
        const d = Math.hypot(dx, dy) || 1;
        const reach = 240 * this.obj.scale;
        if (d < reach) {
          const side = Math.sign(dx * objVx + dy * objVy) || 1;
          const fall = (1 - d / reach) * bh.wake.strength * 1.1;
          ax += -objVy * fall * side + objVx * fall * 0.5; // закрутка + снос в кильватер
          ay += objVx * fall * side + objVy * fall * 0.5;
        }
      }

      // [mass] инерция от размера: крупные тяжелее (вялый разгон, ниже предел)
      let respMaxV = ph.maxSpeed * p.eager;
      if (bh.mass && bh.mass.on) {
        const ratio = clamp(pc.baseSize / Math.max(1, p.size), 0.3, 2);
        const resp = lerp(1, ratio, bh.mass.strength);
        ax *= resp;
        ay *= resp;
        respMaxV *= clamp(resp, 0.4, 1.6);
      }

      // интегрируем скорость (с трением), ограничиваем, двигаем ПРЕДСКАЗАННУЮ позицию
      p.vx = (p.vx + ax * dt) * Math.pow(ph.friction, dt * 60);
      p.vy = (p.vy + ay * dt) * Math.pow(ph.friction, dt * 60);
      const sp = Math.hypot(p.vx, p.vy);
      if (sp > respMaxV) {
        p.vx = (p.vx / sp) * respMaxV;
        p.vy = (p.vy / sp) * respMaxV;
      }
      p.x += p.vx * dt;
      p.y += p.vy * dt;

      const desired = Math.atan2(faceY, faceX);
      const smooth = pc.rotationSmoothness || 0;
      if (smooth > 0.001) {
        // signed угловая разница
        let diff = desired - p.angle;
        if (diff > Math.PI) diff -= TAU;
        if (diff < -Math.PI) diff += TAU;
        const targetVA = diff * pc.turnSpeed;
        const t = clamp((1 - smooth) * 60 * dt, 0, 1);
        p.va = lerp(p.va, targetVA, t);
        p.angle += p.va * dt;
      } else {
        p.angle = angLerp(p.angle, desired, clamp(pc.turnSpeed * dt, 0, 1));
      }
    }

    // --- 2. решатель ограничений (PBD): коллизии рыбок + граница объекта ---
    //     Всё позиционно и в ОДНОМ цикле. Граница решается сразу после парных
    //     коллизий на каждой итерации, поэтому, когда задняя рыбка вдавливает
    //     переднюю в объект, стенка тут же выталкивает переднюю, а перекрытие
    //     уходит назад — к толкающей. Давление расходится по касательной (рыбки
    //     облепляют силуэт), а не превращается в «дёрнули внутрь → выкинуло».
    const iters = speedMode ? Math.max(3, Math.round(ph.collisionIters * 0.6)) : ph.collisionIters;
    const push = ph.collisionPush;
    for (let it = 0; it < iters; it++) {
      // в speed-режиме строим сетку один раз (положения за итерации меняются мало)
      if (it === 0 || !speedMode) this.rebuildHash(pc);
      // 2a. парные коллизии — позиционно, симметрично, по ориентированному эллипсу
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        const pa = p.size,
          pb = p.size * wr;
        this.forEachNeighborFar(p, (q) => {
          if (q.id <= p.id) return; // каждую пару обрабатываем один раз
          let dx = p.x - q.x,
            dy = p.y - q.y;
          let dist = Math.hypot(dx, dy);
          if (dist < 1e-4) {
            dx = (p.id - q.id) || 1;
            dy = 0.3;
            dist = Math.hypot(dx, dy);
          }
          const nx = dx / dist,
            ny = dy / dist;
          const rp = ellipseRadius(pa, pb, nx, ny, p.angle);
          const rq = ellipseRadius(q.size, q.size * wr, -nx, -ny, q.angle);
          const packMul = 1 + (this._shakeRepel || 0);
          const minD = (rp + rq) * pc.packing * packMul;
          if (dist < minD) {
            const corr = (minD - dist) * 0.5 * push; // симметрично разводим
            p.x += nx * corr;
            p.y += ny * corr;
            q.x -= nx * corr;
            q.y -= ny * corr;
          }
        });
      }
      // 2b. граница объекта — жёсткая стенка (двигается только рыбка).
      // в speed-режиме решаем стенку только на последней итерации (1×n семплов
      // поля вместо iters×n) — со свежим семплом, без накопления ошибки.
      if (!speedMode || it === iters - 1)
        for (let i = 0; i < parts.length; i++) this.solveWall(parts[i]);
    }

    // --- 3. восстановление скорости из реального перемещения (PBD) ---
    //     v = (x − x0)/dt: скорость отражает, КУДА рыбку реально пустили
    //     ограничения. Упёршаяся в стенку рыбка получает v≈0 сама собой —
    //     никакого накопленного отскока, поэтому нет прыжков и мельтешения.
    const invDt = 1 / dt;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      let nvx = (p.x - p.px0) * invDt;
      let nvy = (p.y - p.py0) * invDt;
      // ограничим всплески от глубоких разлёгиваний (накопленных перекрытий)
      const cap = ph.maxSpeed * p.eager * 1.5;
      const sp = Math.hypot(nvx, nvy);
      if (sp > cap) {
        nvx = (nvx / sp) * cap;
        nvy = (nvy / sp) * cap;
      }
      p.vx = nvx;
      p.vy = nvy;
      // спячка: микродвижения < порога гасим полностью — рой замирает без дрожи
      if (Math.hypot(p.vx, p.vy) < ph.sleepThreshold) {
        p.vx = 0;
        p.vy = 0;
      }
    }

    // --- 3. удаление улетевших далеко за край (край не стенка) ---
    const M = ph.cullMargin;
    if (M < 100000) {
      let any = false;
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        if (p.x < -M || p.x > W + M || p.y < -M || p.y > H + M) {
          any = true;
          break;
        }
      }
      if (any)
        this.particles = parts.filter(
          (p) => p.x >= -M && p.x <= W + M && p.y >= -M && p.y <= H + M
        );
    }
  }
}

/* ================================ Renderer =============================== */
class Renderer {
  constructor() {
    this.bodySprite = null; // canvas для тела мелкого объекта (front = +X)
    this.useProcedural = true;
    this.procShape = "eye"; // форма процедурной частицы
    this.frontOffset = 0; // рад: коррекция "переда" загруженного спрайта
    this.objSprite = null; // visual ∩ mask, кэш
    this.dotScale = 0.22; // размер зрачка процедурной частицы
  }

  // процедурная частица. "eye" печётся в спрайт (белое тело + зрачок),
  // остальные формы рисуются путём прямо в цикле и красятся цветом частицы.
  buildProceduralSprite(shape = "eye", ratio = 0.5, point = 0.4, dot = 0.22) {
    this.useProcedural = true;
    this.procShape = shape || "eye";
    this.dotScale = dot;
    if (this.procShape !== "eye") {
      this.bodySprite = null; // рисуем путём в draw()
      return;
    }
    const len = 120;
    const h = Math.max(8, len * ratio);
    const c = document.createElement("canvas");
    c.width = len;
    c.height = Math.ceil(h);
    const ctx = c.getContext("2d");
    ctx.translate(0, c.height / 2);
    const cp = lerp(0.46, 0.06, clamp(point, 0, 1)); // ближе к кончику = острее
    ctx.beginPath();
    ctx.moveTo(1, 0);
    ctx.bezierCurveTo(len * cp, -h / 2, len * (1 - cp), -h / 2, len - 1, 0);
    ctx.bezierCurveTo(len * (1 - cp), h / 2, len * cp, h / 2, 1, 0);
    ctx.closePath();
    ctx.fillStyle = "#f6f4ef";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(0,0,0,0.35)";
    ctx.stroke();
    this.bodySprite = c;
  }

  // рисует тело частицы в локальном пространстве (front = +X, центр в 0).
  // L = полудлина (= p.size), Wd = полуширина (= p.size*widthRatio).
  // pr = cfg.proc: форм-специфичные параметры читаются вживую.
  drawProcShape(ctx, shape, L, Wd, color, pr) {
    ctx.fillStyle = color;
    switch (shape) {
      case "square": {
        const rad = Math.min(L, Wd) * (pr.round || 0);
        if (rad > 0.5 && ctx.roundRect) {
          ctx.beginPath(); ctx.roundRect(-L, -Wd, L * 2, Wd * 2, rad); ctx.fill();
        } else ctx.fillRect(-L, -Wd, L * 2, Wd * 2);
        break;
      }
      case "needle": {
        const sx = lerp(0, -L * 0.55, pr.point ?? 0.4);
        ctx.beginPath();
        ctx.moveTo(L, 0); ctx.lineTo(sx, -Wd); ctx.lineTo(-L, 0); ctx.lineTo(sx, Wd);
        ctx.closePath(); ctx.fill();
        break;
      }
      case "bird": {
        const notch = -L * (0.05 + (pr.notch ?? 0.4) * 0.6);
        ctx.beginPath();
        ctx.moveTo(L, 0); ctx.lineTo(-L * 0.6, -Wd); ctx.lineTo(notch, 0); ctx.lineTo(-L * 0.6, Wd);
        ctx.closePath(); ctx.fill();
        break;
      }
      case "drop": {
        const back = -L * lerp(0.35, 0.05, pr.point ?? 0.4);
        ctx.beginPath();
        ctx.moveTo(L, 0);
        ctx.quadraticCurveTo(back, -Wd, -L, 0);
        ctx.quadraticCurveTo(back, Wd, L, 0);
        ctx.closePath(); ctx.fill();
        break;
      }
      case "star": {
        const n = Math.max(3, Math.round(pr.spikes ?? 5));
        const inner = clamp(pr.depth ?? 0.45, 0.1, 0.95);
        ctx.beginPath();
        for (let i = 0; i < n * 2; i++) {
          const a = (i * Math.PI) / n - Math.PI / 2;
          const rad = i % 2 ? inner : 1;
          const x = Math.cos(a) * L * rad, y = Math.sin(a) * Wd * rad;
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.closePath(); ctx.fill();
        break;
      }
      case "polygon": {
        const n = Math.max(3, Math.round(pr.sides ?? 6));
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const a = (i / n) * TAU - Math.PI / 2;
          const x = Math.cos(a) * L, y = Math.sin(a) * Wd;
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.closePath(); ctx.fill();
        break;
      }
      case "heart": {
        // симметричное сердце Безье в единичных координатах, остриём в +X
        const cl = -lerp(0.18, 0.6, pr.depth ?? 0.4); // глубина выемки сзади
        ctx.save();
        ctx.scale(L, Wd);
        ctx.beginPath();
        ctx.moveTo(cl, 0);
        ctx.bezierCurveTo(-1.05, -0.15, -0.85, -1.05, -0.2, -1.0);
        ctx.bezierCurveTo(0.25, -0.95, 0.6, -0.4, 1.0, 0);
        ctx.bezierCurveTo(0.6, 0.4, 0.25, 0.95, -0.2, 1.0);
        ctx.bezierCurveTo(-0.85, 1.05, -1.05, 0.15, cl, 0);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
        break;
      }
      case "crescent": {
        // чистый серп = луна между двумя дугами равных кругов в единичном
        // пространстве (масштаб L×Wd). d — смещение вырезающего круга.
        const hole = clamp(pr.hole ?? 0.5, 0.1, 0.9);
        const d = lerp(1.05, 0.5, hole); // больше hole → тоньше серп
        const ax = d / 2;
        const ay = Math.sqrt(Math.max(1e-4, 1 - ax * ax));
        const thO = Math.atan2(ay, ax);       // угол пересечения на внешнем круге
        const thI = Math.atan2(ay, ax - d);   // на вырезающем круге
        ctx.save();
        ctx.scale(L, Wd);
        ctx.beginPath();
        ctx.arc(0, 0, 1, thO, TAU - thO, false);   // внешний лимб (большая дуга слева)
        ctx.arc(d, 0, 1, TAU - thI, thI, true);    // внутренняя дуга-вырез обратно
        ctx.closePath();
        ctx.fill();
        ctx.restore();
        break;
      }
      case "ring": {
        const hole = clamp(pr.hole ?? 0.5, 0.1, 0.9);
        ctx.beginPath();
        ctx.ellipse(0, 0, L, Wd, 0, 0, TAU);
        ctx.ellipse(0, 0, L * hole, Wd * hole, 0, 0, TAU, true);
        ctx.fill("evenodd");
        break;
      }
      case "fish": {
        const t = pr.tail ?? 0.5;
        const point = pr.point ?? 0.4;
        const nose = L;
        const bx = -L * 0.42;          // конец тела / основание хвоста
        const mx = lerp((nose + bx) / 2, nose * 0.1, point); // острее нос → пик ближе к носу
        // тело — симметричная линза
        ctx.beginPath();
        ctx.moveTo(nose, 0);
        ctx.quadraticCurveTo(mx, -Wd, bx, 0);
        ctx.quadraticCurveTo(mx, Wd, nose, 0);
        ctx.closePath(); ctx.fill();
        // хвост — раздвоённый плавник с выемкой
        const ty = Wd * (0.7 + t);
        ctx.beginPath();
        ctx.moveTo(bx + L * 0.06, 0);
        ctx.lineTo(-L, -ty);
        ctx.lineTo(-L * 0.74, 0);
        ctx.lineTo(-L, ty);
        ctx.closePath(); ctx.fill();
        break;
      }
      case "flower": {
        const n = Math.max(3, Math.round(pr.petals ?? 6));
        for (let i = 0; i < n; i++) {
          const a = (i / n) * TAU;
          ctx.save();
          ctx.translate(Math.cos(a) * L * 0.5, Math.sin(a) * Wd * 0.5);
          ctx.rotate(a);
          ctx.beginPath(); ctx.ellipse(0, 0, L * 0.5, Wd * 0.3, 0, 0, TAU); ctx.fill();
          ctx.restore();
        }
        ctx.beginPath(); ctx.arc(0, 0, Math.min(L, Wd) * 0.38, 0, TAU); ctx.fill();
        break;
      }
      case "arrow": {
        const headLen = L * lerp(0.5, 1.0, pr.point ?? 0.5);
        const sw = Wd * 0.42;
        ctx.beginPath();
        ctx.moveTo(L, 0);
        ctx.lineTo(L - headLen, -Wd); ctx.lineTo(L - headLen, -sw);
        ctx.lineTo(-L, -sw); ctx.lineTo(-L, sw);
        ctx.lineTo(L - headLen, sw); ctx.lineTo(L - headLen, Wd);
        ctx.closePath(); ctx.fill();
        break;
      }
      default: { // "dot" и фолбэк — эллипс
        ctx.beginPath();
        ctx.ellipse(0, 0, L, Wd, 0, 0, TAU);
        ctx.fill();
      }
    }
  }

  setSpriteImage(img, frontOffsetRad) {
    const max = 140;
    const r = Math.min(max / img.width, max / img.height, 1);
    const w = Math.round(img.width * r),
      h = Math.round(img.height * r);
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    c.getContext("2d").drawImage(img, 0, 0, w, h);
    this.bodySprite = c;
    this.useProcedural = false;
    this.frontOffset = frontOffsetRad;
  }

  // кэш главного объекта: visual обрезанный по маске (destination-in)
  buildObjSprite(maskCanvas, visualImg) {
    const w = maskCanvas.width,
      h = maskCanvas.height;
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    if (visualImg) {
      // вписываем visual по центру маски
      const r = Math.min(w / visualImg.width, h / visualImg.height);
      const vw = visualImg.width * r,
        vh = visualImg.height * r;
      ctx.drawImage(visualImg, (w - vw) / 2, (h - vh) / 2, vw, vh);
      ctx.globalCompositeOperation = "destination-in";
      ctx.drawImage(maskCanvas, 0, 0, w, h);
      ctx.globalCompositeOperation = "source-over";
    }
    this.objSprite = c;
  }

  draw(ctx, sim, maskCanvas, cfg, mode) {
    const { canvas } = cfg;
    ctx.fillStyle = canvas.bg;
    ctx.fillRect(0, 0, canvas.w, canvas.h);

    const o = sim.obj;

    // защищённый силуэт (debug) — слабая заливка
    if (cfg.object.showMask && maskCanvas) {
      ctx.save();
      ctx.globalAlpha = 0.18;
      ctx.translate(o.x, o.y);
      ctx.rotate(o.rot);
      ctx.scale(o.scale, o.scale);
      ctx.drawImage(maskCanvas, -maskCanvas.width / 2, -maskCanvas.height / 2);
      ctx.restore();
    }

    // мелкие объекты
    const proc = this.useProcedural;
    const shape = this.procShape;
    const sprite = this.bodySprite;
    const foff = proc ? 0 : this.frontOffset;
    const wr = cfg.particle.widthRatio;
    const pathMode = proc && shape !== "eye"; // формы кроме глаза рисуем путём
    if (pathMode || sprite) {
      const sw = sprite ? sprite.width : 0;
      const sh = sprite ? sprite.height : 0;
      for (let i = 0; i < sim.particles.length; i++) {
        const p = sim.particles[i];
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.angle + foff);
        if (pathMode) {
          this.drawProcShape(ctx, shape, p.size, p.size * wr, p.dot, cfg.proc);
        } else {
          const sc = (p.size * 2) / sw; // визуальный размер (он же для коллизий)
          ctx.scale(sc, sc);
          ctx.drawImage(sprite, -sw / 2, -sh / 2);
          // цветной "зрачок" у переднего кончика (только для глаза)
          if (proc) {
            ctx.beginPath();
            ctx.arc(sw * 0.25, 0, sh * this.dotScale, 0, TAU);
            ctx.fillStyle = p.dot;
            ctx.fill();
            const irisR = sh * this.dotScale;
            const pupilR = irisR * 0.45;
            const pupilOff = irisR - pupilR; // смещение внутри радужки к краю
            ctx.beginPath();
            ctx.arc(sw * 0.25 + pupilOff, 0, pupilR, 0, TAU);
            ctx.fillStyle = "#000";
            ctx.fill();
          }
        }
        ctx.restore();
      }
    }

    // визуальный объект внутри силуэта
    if (cfg.object.showVisual && this.objSprite) {
      ctx.save();
      ctx.translate(o.x, o.y);
      ctx.rotate(o.rot);
      ctx.scale(o.scale, o.scale);
      ctx.drawImage(
        this.objSprite,
        -this.objSprite.width / 2,
        -this.objSprite.height / 2
      );
      ctx.restore();
    }

  }
}

/* ===================== дефолтные ассеты (для демо) ====================== */
function makeBottleMask(w = 300, h = 560) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  const cx = w / 2;
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.moveTo(cx - 22, 18);
  ctx.lineTo(cx + 22, 18); // горлышко
  ctx.lineTo(cx + 22, 70);
  ctx.quadraticCurveTo(cx + 26, 110, cx + 60, 150); // плечи
  ctx.quadraticCurveTo(cx + 78, 180, cx + 78, 240);
  ctx.lineTo(cx + 78, h - 40);
  ctx.quadraticCurveTo(cx + 78, h - 14, cx + 50, h - 14);
  ctx.lineTo(cx - 50, h - 14);
  ctx.quadraticCurveTo(cx - 78, h - 14, cx - 78, h - 40);
  ctx.lineTo(cx - 78, 240);
  ctx.quadraticCurveTo(cx - 78, 180, cx - 60, 150);
  ctx.quadraticCurveTo(cx - 26, 110, cx - 22, 70);
  ctx.closePath();
  ctx.fill();
  return c;
}
function makeLabelVisual(maskW, maskH) {
  const c = document.createElement("canvas");
  c.width = maskW;
  c.height = maskH;
  const ctx = c.getContext("2d");
  ctx.save();
  ctx.translate(c.width / 2, c.height / 2 + 40);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#ffffff";
  ctx.font = "600 56px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("your-label", 0, 0);
  ctx.restore();
  return c;
}

/* ===================== процедурные маски силуэтов ====================== */
function makeShapeCanvas(w, h, paint) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#fff";
  paint(ctx, w, h);
  return c;
}
function makeMask(shape) {
  switch (shape) {
    case "bottle":
      return makeBottleMask();
    case "circle":
      return makeShapeCanvas(440, 440, (ctx, w, h) => {
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, w * 0.42, 0, TAU);
        ctx.fill();
      });
    case "heart":
      return makeShapeCanvas(460, 440, (ctx, w, h) => {
        const x = w / 2, y = h * 0.34, s = w * 0.42;
        ctx.beginPath();
        ctx.moveTo(x, y + s * 0.3);
        ctx.bezierCurveTo(x, y, x - s, y - s * 0.2, x - s, y + s * 0.35);
        ctx.bezierCurveTo(x - s, y + s * 0.9, x, y + s * 1.15, x, y + s * 1.5);
        ctx.bezierCurveTo(x, y + s * 1.15, x + s, y + s * 0.9, x + s, y + s * 0.35);
        ctx.bezierCurveTo(x + s, y - s * 0.2, x, y, x, y + s * 0.3);
        ctx.closePath();
        ctx.fill();
      });
    case "star":
      return makeShapeCanvas(460, 460, (ctx, w, h) => {
        const cx = w / 2, cy = h / 2, R = w * 0.46, r = R * 0.42, n = 5;
        ctx.beginPath();
        for (let i = 0; i < n * 2; i++) {
          const ang = (i * Math.PI) / n - Math.PI / 2;
          const rad = i % 2 ? r : R;
          const px = cx + Math.cos(ang) * rad, py = cy + Math.sin(ang) * rad;
          i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
      });
    case "ring":
      return makeShapeCanvas(480, 480, (ctx, w, h) => {
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, w * 0.44, 0, TAU);
        ctx.arc(w / 2, h / 2, w * 0.24, 0, TAU, true); // дырка (evenodd)
        ctx.fill("evenodd");
      });
    case "branch":
      return makeShapeCanvas(420, 560, (ctx, w, h) => {
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        const cx = w / 2;
        const limb = (x1, y1, x2, y2, lw) => {
          ctx.lineWidth = lw;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        };
        limb(cx, h - 20, cx, h * 0.32, 26);       // ствол
        limb(cx, h * 0.62, cx - 110, h * 0.42, 16); // ветка влево
        limb(cx, h * 0.5, cx + 120, h * 0.3, 16);  // ветка вправо
        limb(cx - 70, h * 0.5, cx - 130, h * 0.6, 9);
        limb(cx + 60, h * 0.4, cx + 120, h * 0.5, 9);
        limb(cx, h * 0.32, cx - 50, h * 0.16, 11);
        limb(cx, h * 0.34, cx + 60, h * 0.14, 11);
      });
    case "letterA":
      return makeShapeCanvas(380, 480, (ctx, w, h) => {
        ctx.fillStyle = "#fff";
        ctx.font = "900 460px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("A", w / 2, h / 2 + 24);
      });
    case "drop":
      return makeShapeCanvas(360, 520, (ctx, w, h) => {
        const cx = w / 2;
        ctx.beginPath();
        ctx.moveTo(cx, 30);
        ctx.bezierCurveTo(cx + 150, h * 0.45, cx + 130, h - 30, cx, h - 30);
        ctx.bezierCurveTo(cx - 130, h - 30, cx - 150, h * 0.45, cx, 30);
        ctx.closePath();
        ctx.fill();
      });
    default:
      return makeBottleMask();
  }
}

/* ============================= пресеты-«характеры» ===================== */
// неглубокий мердж переопределений пресета поверх клона дефолтного конфига
function mergeDeep(base, ov) {
  for (const k in ov) {
    const v = ov[k];
    if (v && typeof v === "object" && !Array.isArray(v)) {
      base[k] = mergeDeep(base[k] || {}, v);
    } else {
      base[k] = v;
    }
  }
  return base;
}
// рекомендуемая «полнота» эллипса коллизии под форму
const SHAPE_WR = {
  eye: 0.7, dot: 0.95, square: 0.85, needle: 0.22, bird: 0.6, drop: 0.55,
  star: 0.95, polygon: 0.9, heart: 0.9, crescent: 0.95, ring: 0.95, fish: 0.55, flower: 0.95, arrow: 0.6,
};
const PRESETS = [
  {
    key: "argus", label: "👁 Аргус", mask: "circle", scale: 1.0,
    cfg: {
      canvas: { bg: "#070707" },
      object: { gap: 2, faceMode: "center", showVisual: false, innerScale: 0.5 },
      proc: { shape: "eye", ratio: 0.72, point: 0.32, dot: 0.36 },
      particle: { baseSize: 18, widthRatio: 0.72, topScale: 1, bottomScale: 1, sizeRandMin: 0.85, sizeRandMax: 1.15, palette: ["#b41818", "#8c0e0e", "#c23a3a", "#5a0a0a"] },
      physics: { maxSpeed: 170, sleepThreshold: 9, packing: 0.93, slowR: 180 },
      behaviors: { adhere: { on: true, strength: 0.4 } },
    },
  },
  {
    key: "brutal", label: "🅰 Брутализм", mask: "letterA", scale: 1.0,
    cfg: {
      canvas: { bg: "#b8b5ad" },
      object: { gap: 1, faceMode: "edge", showVisual: false },
      proc: { shape: "square" },
      particle: { baseSize: 13, widthRatio: 0.85, turnSpeed: 40, rotationSmoothness: 0, topScale: 1, bottomScale: 1, sizeRandMin: 1, sizeRandMax: 1, packing: 1.15, palette: ["#111111", "#1c1c1c", "#0a0a0a"] },
      physics: { maxSpeed: 220, sleepThreshold: 5, slowR: 160 },
      behaviors: { temp: { on: true, strength: 0.18 } },
    },
  },
  {
    key: "gold", label: "🜚 Жидкое золото", mask: "drop", scale: 1.0,
    cfg: {
      canvas: { bg: "#100b04" },
      object: { gap: 2, faceMode: "edge", showVisual: false },
      proc: { shape: "dot" },
      particle: { baseSize: 15, widthRatio: 0.95, topScale: 1, bottomScale: 1, sizeRandMin: 0.85, sizeRandMax: 1.2, packing: 0.86, palette: ["#f4cf5a", "#e6b53c", "#caa12f", "#fff0b0", "#b8862a"] },
      physics: { friction: 0.97, maxSpeed: 150, sleepThreshold: 7, slowR: 220 },
      behaviors: { membrane: { on: true, stiffness: 0.4, radius: 64 }, adhere: { on: true, strength: 0.5 } },
    },
  },
  {
    key: "plankton", label: "✦ Планктон", mask: "circle", scale: 0.7,
    cfg: {
      canvas: { bg: "#02040a" },
      object: { gap: 6, faceMode: "center", showVisual: false },
      proc: { shape: "dot" },
      particle: { baseSize: 5, widthRatio: 0.95, growSeconds: 2, topScale: 1, bottomScale: 1, sizeRandMin: 0.6, sizeRandMax: 1.6, packing: 0.9, palette: ["#27f2d6", "#2f8bff", "#b14bff", "#5cff8f", "#ff4bd6"] },
      physics: { friction: 0.9, maxSpeed: 120, sleepThreshold: 2, slowR: 300 },
      behaviors: { wander: { on: true, strength: 0.6 }, flock: { on: true, align: 0.3, cohesion: 0.2, radius: 80 } },
    },
  },
  {
    key: "crows", label: "🐦‍⬛ Вороны", mask: "branch", scale: 1.0,
    cfg: {
      canvas: { bg: "#eceae2" },
      object: { gap: 4, faceMode: "edge", showVisual: false },
      proc: { shape: "bird" },
      particle: { baseSize: 11, widthRatio: 0.6, topScale: 1, bottomScale: 1, sizeRandMin: 0.7, sizeRandMax: 1.3, packing: 1.0, palette: ["#111111", "#000000", "#1f1f1f"] },
      physics: { friction: 0.92, maxSpeed: 320, sleepThreshold: 1, slowR: 260 },
      behaviors: { flock: { on: true, align: 0.7, cohesion: 0.4, radius: 90 }, wander: { on: true, strength: 0.5 } },
    },
  },
  {
    key: "magnet", label: "🧲 Магнит", mask: "ring", scale: 1.0,
    cfg: {
      canvas: { bg: "#0c0d10" },
      object: { gap: 2, faceMode: "edge", showVisual: false },
      proc: { shape: "needle" },
      particle: { baseSize: 14, widthRatio: 0.22, topScale: 1, bottomScale: 1, sizeRandMin: 0.9, sizeRandMax: 1.1, packing: 0.95, palette: ["#9aa0a8", "#c2c8d0", "#7a808a", "#dfe4ea"] },
      physics: { friction: 0.94, maxSpeed: 240, sleepThreshold: 4, slowR: 240 },
      behaviors: { adhere: { on: true, strength: 0.6 }, swirl: { on: true, strength: 0.5 } },
    },
  },
  {
    key: "glitch", label: "▦ Глитч", mask: "bottle", scale: 0.95,
    cfg: {
      canvas: { bg: "#0a0a0a" },
      object: { gap: 2, faceMode: "edge", showVisual: false },
      proc: { shape: "square" },
      particle: { baseSize: 12, widthRatio: 1.0, turnSpeed: 40, topScale: 1, bottomScale: 1, sizeRandMin: 0.7, sizeRandMax: 1.4, packing: 1.1, palette: ["#ff0000", "#00ff00", "#0000ff", "#ffffff", "#ff00ff"] },
      physics: { friction: 0.9, maxSpeed: 360, sleepThreshold: 0, restitution: 0.5, slowR: 200 },
      behaviors: { temp: { on: true, strength: 0.7 } },
    },
  },
  {
    key: "coral", label: "🪸 Коралл", mask: "heart", scale: 1.0,
    cfg: {
      canvas: { bg: "#160a10" },
      object: { gap: 2, faceMode: "edge", showVisual: false },
      proc: { shape: "drop" },
      particle: { baseSize: 13, widthRatio: 0.55, topScale: 0.9, bottomScale: 1.2, sizeRandMin: 0.7, sizeRandMax: 1.3, packing: 0.92, palette: ["#ff6f61", "#ff9a8b", "#ef476f", "#ffb4a2", "#e85d75"] },
      physics: { friction: 0.95, maxSpeed: 180, sleepThreshold: 4, slowR: 240 },
      behaviors: { membrane: { on: true, stiffness: 0.3, radius: 58 }, adhere: { on: true, strength: 0.5 }, gravity: { on: true, strength: 90, angle: 90 } },
    },
  },
  {
    key: "sumi", label: "⟡ Сумиэ", mask: "branch", scale: 1.0,
    cfg: {
      canvas: { bg: "#f4f1ea" },
      spawn: { maxParticles: 220 },
      object: { gap: 6, faceMode: "edge", showVisual: false },
      proc: { shape: "drop" },
      particle: { baseSize: 14, widthRatio: 0.6, growSeconds: 2.5, topScale: 0.8, bottomScale: 1.2, sizeRandMin: 0.6, sizeRandMax: 1.5, packing: 1.0, palette: ["#15110d", "#000000", "#2a241d"] },
      physics: { friction: 0.94, maxSpeed: 130, sleepThreshold: 2, slowR: 260 },
      behaviors: { wander: { on: true, strength: 0.4 }, gravity: { on: true, strength: 70, angle: 90 } },
    },
  },
  {
    key: "vabank", label: "💥 Ва-банк", mask: "star", scale: 1.0,
    cfg: {
      canvas: { bg: "#0a0a0a" },
      spawn: { burst: 380 },
      object: { gap: 2, faceMode: "inner", showVisual: false, innerScale: 0.6 },
      proc: { shape: "eye", ratio: 0.7, point: 0.4, dot: 0.24 },
      particle: { baseSize: 15, widthRatio: 0.7, topScale: 0.7, bottomScale: 1.5, sizeRandMin: 0.4, sizeRandMax: 2.4, packing: 0.96, palette: ["#e23b3b", "#e6772f", "#e6c12f", "#27a567", "#1fb3b3", "#2f6fe2", "#8a3bc9", "#d23b9c"] },
      physics: { friction: 0.92, maxSpeed: 420, sleepThreshold: 3, slowR: 240 },
      behaviors: {
        flock: { on: true, align: 0.4, cohesion: 0.3, radius: 70 },
        swirl: { on: true, strength: 0.5 },
        wander: { on: true, strength: 0.4 },
        temp: { on: true, strength: 0.3 },
        density: { on: true, strength: 0.5, count: 6 },
      },
    },
  },
];

/* ============================= конфиг по умолчанию ====================== */
const DEFAULT_PALETTE = [
  "#e23b3b", "#2f6fe2", "#27a567", "#e6c12f", "#8a3bc9",
  "#1fb3b3", "#e6772f", "#d23b9c", "#3b3b3b", "#5a9e2f",
];
const PALETTE_PRESETS = {
  радуга: ["#e23b3b","#e6772f","#e6c12f","#27a567","#1fb3b3","#2f6fe2","#8a3bc9","#d23b9c"],
  тёплая: ["#e23b3b","#e6772f","#e6c12f","#f0a030","#d44c4c","#ff6b35","#c0392b","#e67e22"],
  холодная: ["#2f6fe2","#1fb3b3","#8a3bc9","#3498db","#2ecc71","#9b59b6","#1abc9c","#2980b9"],
  моно: ["#555555","#777777","#999999","#bbbbbb","#dddddd","#444444","#666666","#888888"],
};
function defaultConfig() {
  return {
    canvas: { w: 720, h: 720, bg: "#0a0a0a" },
    spawn: { perEmit: 5, brushRadius: 26, maxParticles: 1600, burst: 130 },
    object: { gap: 3, showMask: false, showVisual: true, faceMode: "inner", innerScale: 0.55 },
    particle: {
      spawnSize: 4,
      baseSize: 16,
      growSeconds: 1.6,
      topScale: 0.8,
      bottomScale: 1.3,
      turnSpeed: 14,
      rotationSmoothness: 0,
      packing: 0.99,
      widthRatio: 0.7,
      sizeRandMin: 0.8,
      sizeRandMax: 1.2,
      palette: DEFAULT_PALETTE,
      variations: 5,
    },
    proc: {
      shape: "eye", ratio: 0.7, point: 0.4, dot: 0.22,
      round: 0.3, notch: 0.4, spikes: 5, depth: 0.45, sides: 6, hole: 0.5, tail: 0.5, petals: 6,
    },
    physics: {
      friction: 0.93,
      steer: 5,
      slowR: 250,
      maxSpeed: 350,
      blockDist: 0.8,
      seekBlocked: 0.05,
      eagerFraction: 0.15,
      eagerBoost: 2.0,
      cullMargin: 180,
      collisionIters: 8,
      collisionPush: 0.85,
      restitution: 0,
      sleepThreshold: 6,
    },
    wave: { amp: 12, scale: 0.012, speed: 1.2 },
    perf: { mode: "quality" }, // quality | speed | auto — нагрузка решателя
    // подключаемые режимы поведения (каждый включается тумблером в левой панели)
    behaviors: {
      flock:    { on: false, align: 0.5, cohesion: 0.4, radius: 70 }, // стайность boids
      swirl:    { on: false, strength: 0.5 },        // касательное течение вдоль края
      wander:   { on: false, strength: 0.5 },        // собственное блуждание
      mass:     { on: false, strength: 0.6 },        // инерция от размера
      gravity:  { on: false, strength: 220, angle: 90 }, // постоянный снос (90°=вниз)
      density:  { on: false, strength: 0.6, count: 6 },  // сброс давления в давке
      predator: { on: false, radius: 130, force: 900 },  // бегство от курсора
      wake:     { on: false, strength: 1.0 },        // вихревой след за объектом
      temp:     { on: false, strength: 0.5 },        // непрерывная ажитация
      membrane: { on: false, stiffness: 0.25, radius: 60 }, // мягкое тело (PBD-пружины)
      adhere:   { on: false, strength: 0.5 },        // прилипание к поверхности
    },
    field: { res: 220 },
  };
}

/* ================================ UI helpers ============================ */
function Input({ value, onChange, min, max, step, fmt }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState("");
  return editing ? (
    <input
      type="text"
      autoFocus
      className="w-14 text-right bg-neutral-800 text-amber-400 text-[11px] px-1 rounded ring-1 ring-amber-500 outline-none"
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        setEditing(false);
        const v = parseFloat(text);
        if (!isNaN(v)) onChange(clamp(v, min, max));
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.target.blur();
        } else if (e.key === "Escape") {
          setEditing(false);
        }
      }}
    />
  ) : (
    <button
      className="text-amber-400 tabular-nums hover:text-amber-300 cursor-text"
      onMouseDown={(e) => { e.preventDefault(); setText(String(value)); setEditing(true); }}
      title="Нажмите чтобы ввести значение"
    >
      {fmt ? fmt(value) : value}
    </button>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt, desc }) {
  const [info, setInfo] = useState(false);
  const [infoPos, setInfoPos] = useState({ left: 0, top: 0 });
  const infoRef = useRef(null);
  useEffect(() => {
    if (!info) return;
    const handler = (e) => { if (infoRef.current && !infoRef.current.contains(e.target)) setInfo(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [info]);
  const handleLabelMouseDown = (e) => {
    if (infoRef.current && !infoRef.current.contains(e.target)) {
      setInfo(false);
    }
  };
  return (
    <label className="block mb-2.5" onMouseDown={handleLabelMouseDown}>
      <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
        <span className="flex items-center gap-1">
          {label}
          {desc && (
            <span ref={infoRef} className="relative inline-flex items-center">
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  const r = e.currentTarget.getBoundingClientRect();
                  setInfoPos({ left: r.left, top: r.bottom + 4 });
                  setInfo(!info);
                }}
                className="w-3.5 h-3.5 rounded-full bg-neutral-600 text-neutral-300 text-[9px] font-bold leading-none flex items-center justify-center hover:bg-neutral-500 hover:text-white"
              >i</button>
              {info && (
                <div
                  className="fixed z-[100] w-44 bg-neutral-800 border border-neutral-700 rounded-md p-2 text-[10px] leading-relaxed text-neutral-300 shadow-xl pointer-events-none"
                  style={{ left: Math.min(infoPos.left, window.innerWidth - 180), top: infoPos.top }}
                >
                  {desc}
                </div>
              )}
            </span>
          )}
        </span>
        <Input value={value} onChange={onChange} min={min} max={max} step={step} fmt={fmt} />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-amber-500"
      />
    </label>
  );
}
function Section({ title, children, open = true }) {
  const [o, setO] = useState(open);
  return (
    <div className="border-b border-neutral-800">
      <button
        onClick={() => setO(!o)}
        className="w-full text-left px-3 py-2 text-[11px] font-semibold tracking-wide text-neutral-300 uppercase flex justify-between"
      >
        {title}
        <span className="text-neutral-600">{o ? "–" : "+"}</span>
      </button>
      {o && <div className="px-3 pb-3">{children}</div>}
    </div>
  );
}
// тумблер режима поведения: чекбокс + (когда включён) его слайдеры
function Toggle({ label, on, onChange, children }) {
  return (
    <div className="mb-1.5">
      <label className="flex items-center gap-2 text-[11px] py-1 cursor-pointer select-none">
        <input type="checkbox" className="accent-amber-500" checked={on} onChange={(e) => onChange(e.target.checked)} />
        <span className={on ? "text-amber-400 font-medium" : "text-neutral-300"}>{label}</span>
      </label>
      {on && <div className="pl-3 ml-1 border-l border-neutral-800">{children}</div>}
    </div>
  );
}
function FileRow({ label, accept, onFile }) {
  const ref = useRef(null);
  return (
    <div className="mb-2">
      <button
        onClick={() => ref.current?.click()}
        className="w-full text-[11px] px-2 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-left"
      >
        ↑ {label}
      </button>
      <input
        ref={ref}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}

/* ================================ Component ============================= */
export default function App() {
  const canvasRef = useRef(null);
  const simRef = useRef(null);
  const rendRef = useRef(null);
  const maskRef = useRef(null); // текущий canvas маски
  const rafRef = useRef(0);
  const lastT = useRef(0);
  const fpsAcc = useRef({ t: 0, n: 0 }); // усреднение FPS, чтобы счётчик не мельтешил
  const mouse = useRef({ down: false, x: 0, y: 0, dragOff: { x: 0, y: 0 } });

  const [cfg, setCfg] = useState(defaultConfig());
  const [mode, setMode] = useState("spawn"); // spawn | move
  const [paused, setPaused] = useState(false);
  const [count, setCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [frontDeg, setFrontDeg] = useState(0);
  const [recording, setRecording] = useState(false);
  const [activePreset, setActivePreset] = useState(null);
  const mediaRecRef = useRef(null);

  // храним актуальный конфиг в ref, чтобы цикл не пересоздавался
  const cfgRef = useRef(cfg);
  cfgRef.current = cfg;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  // обновление поля SDF из текущей маски
  const rebuildField = useCallback(() => {
    if (!maskRef.current || !simRef.current) return;
    simRef.current.setField(
      new DistanceField(maskRef.current, cfgRef.current.field.res)
    );
  }, []);

  // инициализация (один раз)
  useEffect(() => {
    const sim = new Simulation();
    sim.cfg = cfgRef.current;
    const rend = new Renderer();
    const pp = cfgRef.current.proc;
    rend.buildProceduralSprite(pp.shape, cfgRef.current.particle.widthRatio, pp.point, pp.dot);
    simRef.current = sim;
    rendRef.current = rend;

    // дефолтные ассеты
    const mask = makeBottleMask();
    maskRef.current = mask;
    const visual = makeLabelVisual(mask.width, mask.height);
    rend.buildObjSprite(mask, visual);
    sim.obj.x = cfgRef.current.canvas.w / 2;
    sim.obj.y = cfgRef.current.canvas.h / 2;
    sim.obj.scale = 0.95;
    rebuildField();

    const loop = (t) => {
      const sim = simRef.current,
        rend = rendRef.current,
        cv = canvasRef.current;
      if (sim && rend && cv) {
        sim.cfg = cfgRef.current;
        const ctx = cv.getContext("2d");
        const dt = lastT.current ? (t - lastT.current) / 1000 : 0.016;
        lastT.current = t;
        if (!pausedRef.current)
          sim.step(dt, cfgRef.current.canvas.w, cfgRef.current.canvas.h);
        rend.draw(ctx, sim, maskRef.current, cfgRef.current, modeRef.current);
        // эмиссия при зажатой мыши в режиме spawn
        if (mouse.current.down && modeRef.current === "spawn")
          sim.spawn(mouse.current.x, mouse.current.y);
        // обновляем счётчики ~3 раза в секунду усреднённым FPS (без мельтешения)
        const acc = fpsAcc.current;
        acc.t += dt;
        acc.n += 1;
        if (acc.t >= 0.33) {
          setFps(Math.round(acc.n / acc.t));
          setCount(sim.particles.length);
          acc.t = 0;
          acc.n = 0;
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [rebuildField]);

  // обновляем поле при смене разрешения
  useEffect(() => {
    rebuildField();
  }, [cfg.field.res, rebuildField]);

  // перестраиваем стандартную частицу при смене её параметров
  useEffect(() => {
    const r = rendRef.current;
    if (!r) return;
    if (r.useProcedural) {
      // глаз печётся в спрайт (высота тела = полнота widthRatio); остальные формы
      // рисуются вживую в draw() и не требуют пересборки.
      r.buildProceduralSprite(cfg.proc.shape, cfg.particle.widthRatio, cfg.proc.point, cfg.proc.dot);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg.proc.shape, cfg.particle.widthRatio, cfg.proc.point, cfg.proc.dot]);

  /* ------------------------ загрузка файлов ------------------------- */
  const loadImage = (file) =>
    new Promise((res, rej) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        res(img);
      };
      img.onerror = rej;
      img.src = url;
    });

  const onMaskFile = async (file) => {
    const img = await loadImage(file);
    const c = document.createElement("canvas");
    c.width = img.width;
    c.height = img.height;
    c.getContext("2d").drawImage(img, 0, 0);
    maskRef.current = c;
    // пересобрать visual-кэш под новую маску (если visual уже был — потеряется,
    // т.к. кэш строится из visual; для простоты строим заново при загрузке visual)
    rendRef.current.buildObjSprite(c, null);
    rebuildField();
  };
  const visualImgRef = useRef(null);
  const onVisualFile = async (file) => {
    const img = await loadImage(file);
    visualImgRef.current = img;
    rendRef.current.buildObjSprite(maskRef.current, img);
  };
  const onSpriteFile = async (file) => {
    const img = await loadImage(file);
    rendRef.current.setSpriteImage(img, (frontDeg * Math.PI) / 180);
    set("particle.widthRatio", clamp(img.height / img.width, 0.15, 1));
  };

  // применить пресет-«характер»: конфиг + маска + форма частицы + заполнение
  const applyPreset = (preset) => {
    const sim = simRef.current, rend = rendRef.current;
    if (!sim || !rend) return;
    const c = mergeDeep(structuredClone(defaultConfig()), preset.cfg);
    cfgRef.current = c;
    sim.cfg = c;
    setCfg(c);
    const mask = makeMask(preset.mask);
    maskRef.current = mask;
    rend.buildObjSprite(mask, null);
    sim.setField(new DistanceField(mask, c.field.res));
    rend.buildProceduralSprite(c.proc.shape, c.particle.widthRatio, c.proc.point, c.proc.dot);
    sim.obj.x = c.canvas.w / 2;
    sim.obj.y = c.canvas.h / 2;
    sim.obj.scale = preset.scale ?? 0.95;
    sim.obj.rot = 0;
    sim.clear();
    sim.fillCanvas(c.canvas.w, c.canvas.h); // мгновенный результат
    setActivePreset(preset.key);
  };

  // обновление front offset для загруженного спрайта
  useEffect(() => {
    if (rendRef.current && !rendRef.current.useProcedural)
      rendRef.current.frontOffset = (frontDeg * Math.PI) / 180;
  }, [frontDeg]);

  /* ------------------------ мышь ------------------------------------ */
  const evtPos = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    const sx = cfg.canvas.w / r.width,
      sy = cfg.canvas.h / r.height;
    return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
  };
  const onDown = (e) => {
    const p = evtPos(e);
    mouse.current.down = true;
    mouse.current.x = p.x;
    mouse.current.y = p.y;
    if (mode === "move") {
      const o = simRef.current.obj;
      mouse.current.dragOff = { x: o.x - p.x, y: o.y - p.y };
    } else if (mode === "erase") {
      simRef.current?.erase(p.x, p.y, cfgRef.current.spawn.brushRadius * 2);
    } else if (mode === "push") {
      const r = cfgRef.current.spawn.brushRadius * 3;
      simRef.current?.push(p.x, p.y, r, 600);
    } else {
      simRef.current.spawn(p.x, p.y);
    }
  };
  const onMove = (e) => {
    const p = evtPos(e);
    mouse.current.x = p.x;
    mouse.current.y = p.y;
    if (simRef.current) { simRef.current.pointer.x = p.x; simRef.current.pointer.y = p.y; simRef.current.pointer.inside = true; }
    if (mode === "move" && mouse.current.down) {
      const o = simRef.current.obj;
      o.x = p.x + mouse.current.dragOff.x;
      o.y = p.y + mouse.current.dragOff.y;
    }
    if (mode === "erase" && mouse.current.down) {
      simRef.current?.erase(p.x, p.y, cfgRef.current.spawn.brushRadius * 2);
    }
    if (mode === "push" && mouse.current.down) {
      const r = cfgRef.current.spawn.brushRadius * 3;
      simRef.current?.push(p.x, p.y, r, 600);
    }
  };
  const onUp = () => (mouse.current.down = false);

  /* ------------------------ экспорт --------------------------------- */
  const saveImage = () => {
    const sim = simRef.current, rend = rendRef.current, cv = canvasRef.current, mask = maskRef.current;
    if (!sim || !rend || !cv) return;
    const W = cfgRef.current.canvas.w, H = cfgRef.current.canvas.h;
    const SCALE = 4; // 4× — 720→2880px
    const oc = document.createElement("canvas");
    oc.width = W * SCALE;
    oc.height = H * SCALE;
    const octx = oc.getContext("2d");
    octx.scale(SCALE, SCALE);
    // рисуем с поправкой на размер — в координатах W×H, но пиксели 4×
    const hiCfg = {
      ...cfgRef.current,
      canvas: { ...cfgRef.current.canvas, w: W, h: H },
    };
    rend.draw(octx, sim, mask, hiCfg, modeRef.current);
    const link = document.createElement("a");
    link.download = "swarm-lab.png";
    link.href = oc.toDataURL("image/png");
    link.click();
  };
  const getVideoMime = () => {
    for (const m of ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"])
      if (MediaRecorder.isTypeSupported(m)) return m;
    return "video/webm";
  };
  const toggleRecord = () => {
    const cv = canvasRef.current;
    if (!cv) return;
    if (mediaRecRef.current && mediaRecRef.current.state === "recording") {
      mediaRecRef.current.stop();
      setRecording(false);
      return;
    }
    const stream = cv.captureStream(30);
    const chunks = [];
    const mr = new MediaRecorder(stream, { mimeType: getVideoMime(), videoBitsPerSecond: 25_000_000 });
    mr.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    mr.onstop = () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.download = "swarm-lab.webm";
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    };
    mediaRecRef.current = mr;
    mr.start();
    setRecording(true);
  };

  /* ------------------------ сеттеры конфига ------------------------- */
  const set = (path, v) =>
    setCfg((c) => {
      const n = structuredClone(c);
      const ks = path.split(".");
      let o = n;
      for (let i = 0; i < ks.length - 1; i++) o = o[ks[i]];
      o[ks[ks.length - 1]] = v;
      return n;
    });
  const setObj = (k, v) => {
    if (simRef.current) simRef.current.obj[k] = v;
  };

  return (
    <div className="flex w-full h-screen bg-neutral-950 text-neutral-200 font-sans">
      {/* левая панель — визуал / ассеты */}
      <div className="w-64 bg-neutral-900 border-r border-neutral-800 overflow-y-auto shrink-0">
        <Section title="✨ Пресеты">
          <p className="text-[10px] text-neutral-500 leading-relaxed mb-2">
            Готовые «характеры»: объект + форма + палитра + поведение. Клик — применить и заполнить.
          </p>
          <div className="grid grid-cols-2 gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p.key}
                onClick={() => applyPreset(p)}
                className={`text-[10px] px-2 py-2 rounded text-left leading-tight transition-colors ${
                  activePreset === p.key
                    ? "bg-amber-500 text-black font-medium"
                    : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </Section>

        <Section title="Объект">
          <FileRow label="Маска зоны (ЧБ/альфа)" accept="image/*" onFile={onMaskFile} />
          <FileRow label="Визуал объекта" accept="image/*" onFile={onVisualFile} />
          <Slider label="Масштаб" desc="Размер главного объекта на холсте. Увеличение сжимает SDF — рыбки видят объект больше." value={simRef.current?.obj.scale ?? 1} min={0.3} max={2.5} step={0.01} onChange={(v) => setObj("scale", v)} fmt={(v)=>v.toFixed(2)} />
          <Slider label="Поворот" desc="Угол поворота главного объекта в градусах. Маска и визуал вращаются вместе." value={(simRef.current?.obj.rot ?? 0)} min={-3.14} max={3.14} step={0.01} onChange={(v) => setObj("rot", v)} fmt={(v)=>`${Math.round(v*57.3)}°`} />
          <Slider label="Зазор у носа" desc="Минимальная дистанция от рыбки до объекта. Меньше = рыбки ближе к границе, больше = толстый слой воздуха." value={cfg.object.gap} min={-6} max={30} step={1} onChange={(v) => set("object.gap", v)} />
          <label className="flex items-center gap-2 text-[11px] text-neutral-400 mt-1">
            <input type="checkbox" checked={cfg.object.showMask} onChange={(e) => set("object.showMask", e.target.checked)} /> показать маску
          </label>
          <label className="flex items-center gap-2 text-[11px] text-neutral-400">
            <input type="checkbox" checked={cfg.object.showVisual} onChange={(e) => set("object.showVisual", e.target.checked)} /> показать визуал
          </label>
          <div className="mt-2">
            <div className="text-[11px] text-neutral-400 mb-1">Взгляд агентов</div>
            <div className="flex gap-1">
              {(["edge","inner","center"]).map(m => (
                <button key={m} onClick={() => set("object.faceMode", m)}
                  className={`text-[10px] px-2 py-1 rounded flex-1 ${cfg.object.faceMode===m?"bg-amber-500 text-black":"bg-neutral-800 text-neutral-300"}`}
                >{m==="edge"?"Край":m==="inner"?"Внутри":"Центр"}</button>
              ))}
            </div>
            {cfg.object.faceMode === "inner" && (
              <Slider label="Глубина внутрь" desc="Насколько глубоко внутри объекта точка притяжения. 1 = рыбки смотрят на край, 0.1 = почти в центр." value={cfg.object.innerScale} min={0.1} max={1} step={0.01} onChange={(v) => set("object.innerScale", v)} fmt={(v)=>v.toFixed(2)} />
            )}
          </div>
        </Section>

        <Section title="Агент">
          <FileRow label="Спрайт агента" accept="image/*" onFile={onSpriteFile} />
          <Slider label="Перёд спрайта" desc="Поворот загруженного спрайта: куда смотрит «нос» рыбки. Подберите чтобы спрайт летел головой вперёд." value={frontDeg} min={-180} max={180} step={1} onChange={setFrontDeg} fmt={(v) => `${v}°`} />
          <button
            onClick={() => {
              const pp = cfgRef.current.proc;
              set("proc.shape", "eye");
              set("particle.widthRatio", SHAPE_WR.eye);
              rendRef.current.buildProceduralSprite("eye", SHAPE_WR.eye, pp.point, pp.dot);
            }}
            className="w-full text-[11px] px-2 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 mt-1"
          >
            ↺ Спрайт по умолчанию
          </button>
        </Section>

        <Section title="Форма агента">
          <div className="text-[11px] text-neutral-400 mb-1">Тип частицы</div>
          <div className="grid grid-cols-3 gap-1 mb-2">
            {([
              ["eye","Глаз"],["dot","Точка"],["square","Квадрат"],
              ["needle","Игла"],["bird","Птица"],["drop","Капля"],
              ["fish","Рыба"],["arrow","Стрелка"],["star","Звезда"],
              ["polygon","Поли"],["heart","Сердце"],["flower","Цветок"],
              ["ring","Кольцо"],["crescent","Месяц"],
            ]).map(([s, lbl]) => (
              <button key={s}
                onClick={() => { set("proc.shape", s); set("particle.widthRatio", SHAPE_WR[s]); }}
                className={`text-[10px] px-1 py-1 rounded ${cfg.proc.shape === s ? "bg-amber-500 text-black" : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"}`}
              >{lbl}</button>
            ))}
          </div>

          {/* общая для всех форм «полнота» = эллипс коллизии и пропорция тела */}
          <Slider label="Полнота (шир/длин)" desc="Соотношение ширины к длине частицы. 0.2 = узкая/длинная, 1 = круглая. Влияет и на форму, и на эллипс коллизий." value={cfg.particle.widthRatio} min={0.15} max={1} step={0.01} onChange={(v) => set("particle.widthRatio", v)} fmt={(v)=>v.toFixed(2)} />

          {/* контекстные параметры под выбранную форму */}
          {cfg.proc.shape === "eye" && (<>
            <Slider label="Острота кончика" desc="Насколько заострён передний кончик глаза. 0 = тупой овал, 1 = игла." value={cfg.proc.point} min={0} max={1} step={0.02} onChange={(v) => set("proc.point", v)} fmt={(v)=>v.toFixed(2)} />
            <Slider label="Размер зрачка" desc="Диаметр цветного зрачка на кончике. 0 = без зрачка, 0.45 = полглаза." value={cfg.proc.dot} min={0} max={0.45} step={0.01} onChange={(v) => set("proc.dot", v)} fmt={(v)=>v.toFixed(2)} />
          </>)}
          {(cfg.proc.shape === "drop" || cfg.proc.shape === "needle" || cfg.proc.shape === "fish" || cfg.proc.shape === "arrow") && (
            <Slider label="Острота носа" desc="Насколько вытянут и заострён передний кончик." value={cfg.proc.point} min={0} max={1} step={0.02} onChange={(v) => set("proc.point", v)} fmt={(v)=>v.toFixed(2)} />
          )}
          {cfg.proc.shape === "fish" && (
            <Slider label="Хвост" desc="Размах хвостового плавника." value={cfg.proc.tail} min={0} max={1.2} step={0.05} onChange={(v) => set("proc.tail", v)} fmt={(v)=>v.toFixed(2)} />
          )}
          {cfg.proc.shape === "bird" && (
            <Slider label="Изгиб крыльев" desc="Глубина выреса между крыльями. Больше = острее «галочка»." value={cfg.proc.notch} min={0} max={1} step={0.02} onChange={(v) => set("proc.notch", v)} fmt={(v)=>v.toFixed(2)} />
          )}
          {cfg.proc.shape === "square" && (
            <Slider label="Скругление углов" desc="0 = острый квадрат, 1 = почти круг." value={cfg.proc.round} min={0} max={1} step={0.05} onChange={(v) => set("proc.round", v)} fmt={(v)=>v.toFixed(2)} />
          )}
          {cfg.proc.shape === "star" && (<>
            <Slider label="Лучей" desc="Количество лучей звезды." value={cfg.proc.spikes} min={3} max={12} step={1} onChange={(v) => set("proc.spikes", v)} />
            <Slider label="Глубина лучей" desc="Насколько глубоко вырезаны лучи. Меньше = острее звезда." value={cfg.proc.depth} min={0.1} max={0.9} step={0.05} onChange={(v) => set("proc.depth", v)} fmt={(v)=>v.toFixed(2)} />
          </>)}
          {cfg.proc.shape === "polygon" && (
            <Slider label="Сторон" desc="Число сторон многоугольника. 3 = треугольник, 6 = шестиугольник." value={cfg.proc.sides} min={3} max={10} step={1} onChange={(v) => set("proc.sides", v)} />
          )}
          {cfg.proc.shape === "heart" && (
            <Slider label="Полнота лопастей" desc="Размер верхних долей сердца." value={cfg.proc.depth} min={0.1} max={0.9} step={0.05} onChange={(v) => set("proc.depth", v)} fmt={(v)=>v.toFixed(2)} />
          )}
          {cfg.proc.shape === "flower" && (
            <Slider label="Лепестков" desc="Количество лепестков цветка." value={cfg.proc.petals} min={3} max={12} step={1} onChange={(v) => set("proc.petals", v)} />
          )}
          {(cfg.proc.shape === "ring" || cfg.proc.shape === "crescent") && (
            <Slider label={cfg.proc.shape === "ring" ? "Дырка" : "Серп"} desc={cfg.proc.shape === "ring" ? "Размер внутреннего отверстия кольца." : "Насколько узок серп месяца."} value={cfg.proc.hole} min={0.1} max={0.9} step={0.05} onChange={(v) => set("proc.hole", v)} fmt={(v)=>v.toFixed(2)} />
          )}
        </Section>

        <Section title="Цвета">
          <div className="flex gap-1 flex-wrap px-3 pb-1">
            {Object.entries(PALETTE_PRESETS).map(([name, colors]) => (
              <button
                key={name}
                onClick={() => set("particle.palette", colors)}
                className={`text-[10px] px-2 py-1 rounded ${cfg.particle.palette === colors ? "bg-amber-500 text-black" : "bg-neutral-800 text-neutral-300"}`}
              >
                {name}
              </button>
            ))}
          </div>
          <Slider label="Вариаций" desc="Сколько разных цветов из палитры используется. Больше = пёстрый рой." value={cfg.particle.variations} min={1} max={10} step={1} onChange={(v) => set("particle.variations", v)} />
          <div className="flex gap-1 flex-wrap px-3 pb-2">
            {(cfg.particle.palette || []).slice(0, cfg.particle.variations).map((c, i) => (
              <div key={i} className="w-4 h-4 rounded-full border border-neutral-600" style={{ backgroundColor: c }} title={c} />
            ))}
          </div>
        </Section>

        <Section title="Поведение (режимы)">
          <p className="text-[10px] text-neutral-500 leading-relaxed mb-2">
            Включаемые модели движения. Можно комбинировать.
          </p>

          <Toggle label="Стайность (boids)" on={cfg.behaviors.flock.on} onChange={(v) => set("behaviors.flock.on", v)}>
            <Slider label="Выравнивание" desc="Насколько рыбка подстраивает скорость под соседей. Больше = согласованные потоки." value={cfg.behaviors.flock.align} min={0} max={1} step={0.05} onChange={(v) => set("behaviors.flock.align", v)} fmt={(v)=>v.toFixed(2)} />
            <Slider label="Сплочение" desc="Притяжение к центру локальной группы. Больше = плотные косяки." value={cfg.behaviors.flock.cohesion} min={0} max={1} step={0.05} onChange={(v) => set("behaviors.flock.cohesion", v)} fmt={(v)=>v.toFixed(2)} />
            <Slider label="Радиус стаи" desc="Дистанция, на которой рыбки считаются соседями по стае." value={cfg.behaviors.flock.radius} min={20} max={160} step={5} onChange={(v) => set("behaviors.flock.radius", v)} fmt={(v)=>`${v}px`} />
          </Toggle>

          <Toggle label="Течение вдоль края" on={cfg.behaviors.swirl.on} onChange={(v) => set("behaviors.swirl.on", v)}>
            <Slider label="Сила / направление" desc="Касательное течение у поверхности силуэта. Знак = направление вращения." value={cfg.behaviors.swirl.strength} min={-1} max={1} step={0.05} onChange={(v) => set("behaviors.swirl.strength", v)} fmt={(v)=>v.toFixed(2)} />
          </Toggle>

          <Toggle label="Блуждание" on={cfg.behaviors.wander.on} onChange={(v) => set("behaviors.wander.on", v)}>
            <Slider label="Сила" desc="Медленный дрейф собственного курса — рыбки меандрируют, а не летят по прямой." value={cfg.behaviors.wander.strength} min={0} max={1} step={0.05} onChange={(v) => set("behaviors.wander.strength", v)} fmt={(v)=>v.toFixed(2)} />
          </Toggle>

          <Toggle label="Масса от размера" on={cfg.behaviors.mass.on} onChange={(v) => set("behaviors.mass.on", v)}>
            <Slider label="Сила" desc="Крупные рыбки инертнее (вялый разгон, ниже предел скорости), мелкие — юркие." value={cfg.behaviors.mass.strength} min={0} max={1} step={0.05} onChange={(v) => set("behaviors.mass.strength", v)} fmt={(v)=>v.toFixed(2)} />
          </Toggle>

          <Toggle label="Гравитация / стекание" on={cfg.behaviors.gravity.on} onChange={(v) => set("behaviors.gravity.on", v)}>
            <Slider label="Сила" desc="Постоянный снос. Рой стекает и собирается у основания силуэта." value={cfg.behaviors.gravity.strength} min={0} max={600} step={10} onChange={(v) => set("behaviors.gravity.strength", v)} />
            <Slider label="Направление" desc="Угол сноса в градусах. 90° = вниз, 0° = вправо, 270° = вверх." value={cfg.behaviors.gravity.angle} min={0} max={360} step={5} onChange={(v) => set("behaviors.gravity.angle", v)} fmt={(v)=>`${v}°`} />
          </Toggle>

          <Toggle label="Сброс давления" on={cfg.behaviors.density.on} onChange={(v) => set("behaviors.density.on", v)}>
            <Slider label="Сила" desc="В перенаселённых зонах рыбки толкаются наружу — фронт «дышит», не каменеет." value={cfg.behaviors.density.strength} min={0} max={1} step={0.05} onChange={(v) => set("behaviors.density.strength", v)} fmt={(v)=>v.toFixed(2)} />
            <Slider label="Порог соседей" desc="Со скольких соседей включается расталкивание." value={cfg.behaviors.density.count} min={2} max={15} step={1} onChange={(v) => set("behaviors.density.count", v)} />
          </Toggle>

          <Toggle label="Хищник (курсор)" on={cfg.behaviors.predator.on} onChange={(v) => set("behaviors.predator.on", v)}>
            <p className="text-[10px] text-neutral-500 leading-relaxed mb-1">Рыбки разбегаются от курсора, пока он над холстом.</p>
            <Slider label="Радиус паники" desc="На каком расстоянии от курсора рыбки начинают бежать." value={cfg.behaviors.predator.radius} min={30} max={320} step={10} onChange={(v) => set("behaviors.predator.radius", v)} fmt={(v)=>`${v}px`} />
            <Slider label="Сила бегства" desc="Как резко рыбки удирают от курсора." value={cfg.behaviors.predator.force} min={100} max={2000} step={50} onChange={(v) => set("behaviors.predator.force", v)} />
          </Toggle>

          <Toggle label="След за объектом" on={cfg.behaviors.wake.on} onChange={(v) => set("behaviors.wake.on", v)}>
            <Slider label="Сила" desc="Вихревой кильватер за движущимся силуэтом. Видно при перетаскивании объекта." value={cfg.behaviors.wake.strength} min={0} max={3} step={0.1} onChange={(v) => set("behaviors.wake.strength", v)} fmt={(v)=>v.toFixed(1)} />
          </Toggle>

          <Toggle label="Температура" on={cfg.behaviors.temp.on} onChange={(v) => set("behaviors.temp.on", v)}>
            <Slider label="Ажитация" desc="Непрерывное случайное дрожание роя. Больше = рой «кипит»." value={cfg.behaviors.temp.strength} min={0} max={1} step={0.05} onChange={(v) => set("behaviors.temp.strength", v)} fmt={(v)=>v.toFixed(2)} />
          </Toggle>

          <Toggle label="Мембрана (мягкое тело)" on={cfg.behaviors.membrane.on} onChange={(v) => set("behaviors.membrane.on", v)}>
            <Slider label="Жёсткость" desc="Сила пружин между соседями. Рой держится как тянущаяся плёнка." value={cfg.behaviors.membrane.stiffness} min={0} max={1} step={0.05} onChange={(v) => set("behaviors.membrane.stiffness", v)} fmt={(v)=>v.toFixed(2)} />
            <Slider label="Радиус связей" desc="До какой дистанции соседи связаны пружинами." value={cfg.behaviors.membrane.radius} min={30} max={140} step={5} onChange={(v) => set("behaviors.membrane.radius", v)} fmt={(v)=>`${v}px`} />
          </Toggle>

          <Toggle label="Прилипание к краю" on={cfg.behaviors.adhere.on} onChange={(v) => set("behaviors.adhere.on", v)}>
            <Slider label="Сила" desc="Рыбки приклеиваются к поверхности силуэта, образуя ползущий слой." value={cfg.behaviors.adhere.strength} min={0} max={1} step={0.05} onChange={(v) => set("behaviors.adhere.strength", v)} fmt={(v)=>v.toFixed(2)} />
          </Toggle>
        </Section>
      </div>

      {/* сцена */}
      <div className="flex-1 flex flex-col items-center justify-center p-4 overflow-auto">
        <div className="mb-3 flex items-center gap-3 text-xs text-neutral-400">
          <span className="text-amber-400 font-semibold tracking-widest">
            SWARM·LAB
          </span>
          <span className="tabular-nums w-[7ch] text-right">{String(count).padStart(5, "\u00a0")} объектов</span>
          <span className="tabular-nums w-[5ch] text-right">{String(fps).padStart(3, "\u00a0")} fps</span>
        </div>
        <canvas
          ref={canvasRef}
          width={cfg.canvas.w}
          height={cfg.canvas.h}
          onMouseDown={onDown}
          onMouseMove={onMove}
          onMouseUp={onUp}
          onMouseLeave={() => { onUp(); if (simRef.current) simRef.current.pointer.inside = false; }}
          className="rounded-lg shadow-2xl cursor-crosshair max-w-full max-h-[80vh] border border-neutral-700"
          style={{ touchAction: "none", cursor: mode === "erase" ? "cell" : mode === "move" ? "grab" : mode === "push" ? "move" : "crosshair" }}
        />
        <div className="mt-3 flex gap-2 text-xs">
          <button
            onClick={() => setMode("spawn")}
            className={`px-3 py-1.5 rounded ${
              mode === "spawn"
                ? "bg-amber-500 text-black"
                : "bg-neutral-800 text-neutral-300"
            }`}
          >
            Рождать (ЛКМ)
          </button>
          <button
            onClick={() => setMode("move")}
            className={`px-3 py-1.5 rounded ${
              mode === "move"
                ? "bg-amber-500 text-black"
                : "bg-neutral-800 text-neutral-300"
            }`}
          >
            Двигать
          </button>
          <button
            onClick={() => setPaused((p) => !p)}
            className="px-3 py-1.5 rounded bg-neutral-800 text-neutral-300"
          >
            {paused ? "▶ Пуск" : "⏸ Пауза"}
          </button>
          <button
            onClick={() => setMode("erase")}
            className={`px-3 py-1.5 rounded ${
              mode === "erase"
                ? "bg-amber-500 text-black"
                : "bg-neutral-800 text-neutral-300"
            }`}
          >
            ✕ Ластик
          </button>
          <button
            onClick={() => setMode("push")}
            className={`px-3 py-1.5 rounded ${
              mode === "push"
                ? "bg-amber-500 text-black"
                : "bg-neutral-800 text-neutral-300"
            }`}
          >
            ⊚ Толкать
          </button>
          <button
            onClick={() => simRef.current?.shake()}
            className="px-3 py-1.5 rounded bg-neutral-800 text-neutral-300"
          >
            ↯ Встряхнуть
          </button>
          <button
            onClick={() =>
              simRef.current?.fillCanvas(cfg.canvas.w, cfg.canvas.h)
            }
            className="px-3 py-1.5 rounded bg-neutral-800 text-neutral-300"
          >
            Заполнить
          </button>
          <button
            onClick={() => simRef.current?.clear()}
            className="px-3 py-1.5 rounded bg-neutral-800 text-neutral-300"
          >
            Очистить
          </button>
          <button
            onClick={saveImage}
            className="px-3 py-1.5 rounded bg-neutral-800 text-neutral-300"
          >
            📷 PNG
          </button>
          <button
            onClick={toggleRecord}
            className={`px-3 py-1.5 rounded ${
              recording ? "bg-red-600 text-white" : "bg-neutral-800 text-neutral-300"
            }`}
          >
            {recording ? "⏹ Стоп" : "⏺ Видео"}
          </button>
        </div>
      </div>

      {/* правая панель — физика / симуляция */}
      <div className="w-72 bg-neutral-900 border-l border-neutral-800 overflow-y-auto shrink-0">
        <Section title="⚡ Производительность">
          <div className="flex gap-1">
            {([["quality","Качество"],["auto","Авто"],["speed","Скорость"]]).map(([m, lbl]) => (
              <button key={m} onClick={() => set("perf.mode", m)}
                className={`text-[10px] px-2 py-1.5 rounded flex-1 ${cfg.perf.mode === m ? "bg-amber-500 text-black" : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"}`}
              >{lbl}</button>
            ))}
          </div>
          <p className="text-[10px] text-neutral-500 leading-relaxed mt-2">
            «Скорость» строит сетку коллизий и решает границу 1 раз за кадр (вместо ×проходов) и чуть снижает число проходов — заметный прирост FPS при большом числе частиц ценой лёгкой рыхлости фронта. «Авто» включает это при &gt;1400 частиц.
          </p>
        </Section>

        <Section title="Холст">
          <Slider label="Ширина" desc="Ширина холста в пикселях. Объект автоматически центрируется." value={cfg.canvas.w} min={320} max={2560} step={10} onChange={(v) => { set("canvas.w", v); if (simRef.current) simRef.current.obj.x = v / 2; }} />
          <Slider label="Высота" desc="Высота холста в пикселях. Объект автоматически центрируется." value={cfg.canvas.h} min={320} max={2560} step={10} onChange={(v) => { set("canvas.h", v); if (simRef.current) simRef.current.obj.y = v / 2; }} />
        </Section>

        <Section title="Рождение">
          <Slider label="Кисть (радиус)" desc="Радиус зоны рождения/ластика/толкания вокруг курсора. Больше = шире пятно." value={cfg.spawn.brushRadius} min={2} max={120} step={1} onChange={(v) => set("spawn.brushRadius", v)} />
          <Slider label="За эмиссию" desc="Сколько рыбок рождается за один тик при зажатой кнопке. Больше = гуще поток." value={cfg.spawn.perEmit} min={1} max={30} step={1} onChange={(v) => set("spawn.perEmit", v)} />
          <Slider label="Сила взрыва" desc="Начальная скорость разлёта от точки клика. 0 = рыбки появляются на месте." value={cfg.spawn.burst} min={0} max={400} step={10} onChange={(v) => set("spawn.burst", v)} />
          <Slider label="Лимит объектов" desc="Максимальное число рыбок на холсте. При достижении новые не рождаются." value={cfg.spawn.maxParticles} min={100} max={4000} step={50} onChange={(v) => set("spawn.maxParticles", v)} />
        </Section>

        <Section title="Размер / рост">
          <Slider label="Размер рождения" desc="Начальный размер рыбки сразу после появления. Растёт до базового за время роста." value={cfg.particle.spawnSize} min={1} max={30} step={0.5} onChange={(v) => set("particle.spawnSize", v)} />
          <Slider label="Базовый размер" desc="Финальный размер взрослой рыбки (до множителей). Главный параметр масштаба роя." value={cfg.particle.baseSize} min={4} max={50} step={0.5} onChange={(v) => set("particle.baseSize", v)} />
          <Slider label="Время роста, с" desc="Сколько секунд рыбка растёт от размера рождения до базового. Медленный рост = плавное появление." value={cfg.particle.growSeconds} min={0.1} max={6} step={0.1} onChange={(v) => set("particle.growSeconds", v)} fmt={(v)=>v.toFixed(1)} />
          <Slider label="Масштаб сверху" desc="Множитель размера для рыбок у верхнего края холста. <1 = мелкие сверху." value={cfg.particle.topScale} min={0.2} max={2} step={0.05} onChange={(v) => set("particle.topScale", v)} fmt={(v)=>v.toFixed(2)} />
          <Slider label="Масштаб снизу" desc="Множитель размера для рыбок у нижнего края холста. >1 = крупные снизу." value={cfg.particle.bottomScale} min={0.2} max={3} step={0.05} onChange={(v) => set("particle.bottomScale", v)} fmt={(v)=>v.toFixed(2)} />
          <div className="mt-2 text-[11px] text-neutral-400">Случайный размер при рождении</div>
          <Slider label="Мин. множитель" desc="Минимальный случайный множитель размера. 1 = все одного размера. Меньше = есть мелкие." value={cfg.particle.sizeRandMin} min={0.2} max={2} step={0.05} onChange={(v) => set("particle.sizeRandMin", Math.min(v, cfg.particle.sizeRandMax))} fmt={(v)=>v.toFixed(2)} />
          <Slider label="Макс. множитель" desc="Максимальный случайный множитель размера. Больше = есть крупные рыбы в рое." value={cfg.particle.sizeRandMax} min={0.2} max={3} step={0.05} onChange={(v) => set("particle.sizeRandMax", Math.max(v, cfg.particle.sizeRandMin))} fmt={(v)=>v.toFixed(2)} />
        </Section>

        <Section title="Размер / рост">
          <Slider label="Поворот к краю" desc="Как быстро рыбка разворачивается носом к объекту. Больше = резче, меньше = плавнее." value={cfg.particle.turnSpeed} min={1} max={40} step={0.5} onChange={(v) => set("particle.turnSpeed", v)} />
          <Slider label="Плавность поворота" desc="Сглаживание вращения: 0% — мгновенный поворот как сейчас, 100% — очень плавный, без микродёрганий." value={cfg.particle.rotationSmoothness} min={0} max={0.99} step={0.01} onChange={(v) => set("particle.rotationSmoothness", v)} fmt={(v)=>`${Math.round(v*100)}%`} />
          <Slider label="Дистанция коллизии" desc="Множитель контактной дистанции между рыбками. Ниже 1 = перекрываются плотнее, выше 1 = зазор." value={cfg.particle.packing} min={0.4} max={1.4} step={0.02} onChange={(v) => set("particle.packing", v)} fmt={(v)=>v.toFixed(2)} />
          <p className="text-[10px] text-neutral-500 leading-relaxed -mt-1 mb-2">
            &lt;1 — частицы перекрываются (плотнее, как чешуя); &gt;1 — зазор между ними.
          </p>
          <Slider label="Ширина/длина (эллипс)" desc="Соотношение ширины к длине эллипса коллизии. 0.5 = рыбка вдвое длиннее ширины." value={cfg.particle.widthRatio} min={0.15} max={1} step={0.01} onChange={(v) => set("particle.widthRatio", v)} fmt={(v)=>v.toFixed(2)} />
          <Slider label="Трение" desc="Затухание скорости каждый кадр. 0.99 = почти нет трения (скользят), 0.5 = быстро останавливаются." value={cfg.physics.friction} min={0.5} max={0.99} step={0.01} onChange={(v) => set("physics.friction", v)} fmt={(v)=>v.toFixed(2)} />
          <Slider label="Резкость подплыва" desc="Сила с которой рыбка поворачивает и плывёт к объекту. Больше = быстрее реакция, но может перелететь." value={cfg.physics.steer} min={0.5} max={20} step={0.5} onChange={(v) => set("physics.steer", v)} />
          <Slider label="Мягкость торможения" desc="Дистанция на которой рыбка начинает замедляться перед объектом. Больше = раньше тормозит, мягче паркуется." value={cfg.physics.slowR} min={40} max={500} step={10} onChange={(v) => set("physics.slowR", v)} fmt={(v)=>`${v}px`} />
          <Slider label="Дозаполнение фронта" desc="Насколько заблокированная рыбка продолжает жаться к объекту. 0% = ждёт, 50% = пытается протиснуться." value={cfg.physics.seekBlocked} min={0} max={0.5} step={0.02} onChange={(v) => set("physics.seekBlocked", v)} fmt={(v)=>`${Math.round(v*100)}%`} />
          <p className="text-[10px] text-neutral-500 leading-relaxed -mt-1 mb-2">
            0% — рой стоит как вкопанный (без дрожи), но фронт рыхлее. Выше — фронт плотнее, но появляется лёгкое шевеление.
          </p>
          <Slider label="Волны (амплитуда)" desc="Сила когерентного колебания роя. 0 = нет волн. Больше = рой колышется как трава." value={cfg.wave.amp} min={0} max={80} step={1} onChange={(v) => set("wave.amp", v)} />
          <Slider label="Длина волны" desc="Как часто меняется направление волны по полю. Меньше = крупные волны, больше = частые." value={cfg.wave.scale} min={0.003} max={0.04} step={0.001} onChange={(v) => set("wave.scale", v)} fmt={(v)=>v.toFixed(3)} />
          <Slider label="Скорость волны" desc="Быстрота бегущей волны через рой. 0 = стоячая волна." value={cfg.wave.speed} min={0} max={5} step={0.1} onChange={(v) => set("wave.speed", v)} fmt={(v)=>v.toFixed(1)} />
          <Slider label="Доля &laquo;жадных&raquo;" desc="Процент рыбок которые плывут к объекту быстрее остальных. Создаёт слоистость." value={cfg.physics.eagerFraction} min={0} max={1} step={0.02} onChange={(v) => set("physics.eagerFraction", v)} fmt={(v)=>`${Math.round(v*100)}%`} />
          <Slider label="Скорость &laquo;жадных&raquo;" desc="Во сколько раз быстрее плывут жадные рыбки. 1x = все равны, 3x = прорываются." value={cfg.physics.eagerBoost} min={1} max={5} step={0.25} onChange={(v) => set("physics.eagerBoost", v)} fmt={(v)=>`${v}x`} />
          <p className="text-[10px] text-neutral-500 leading-relaxed -mt-1 mb-2">
            Часть рыбок всегда рвётся коснуться объекта быстрее остальных — без всеобщей давки.
          </p>
          <Slider label="Проходов коллизий" desc="Сколько раз за кадр решаются коллизии. Больше = точнее, но дороже по CPU." value={cfg.physics.collisionIters} min={1} max={12} step={1} onChange={(v) => set("physics.collisionIters", v)} />
          <Slider label="Сила коллизий" desc="Как агрессивно раздвигаются перекрывающиеся рыбки. 0.3 = мягко, 1 = жёстко." value={cfg.physics.collisionPush} min={0.1} max={1} step={0.05} onChange={(v) => set("physics.collisionPush", v)} fmt={(v)=>v.toFixed(2)} />
          <Slider label="Упругость" desc="Сколько энергии сохраняется при столкновении. 0 = полностью гасится (без отскока)." value={cfg.physics.restitution} min={0} max={0.8} step={0.02} onChange={(v) => set("physics.restitution", v)} fmt={(v)=>v.toFixed(2)} />
          <Slider label="Порог спячки" desc="Минимальная скорость при которой рыбка замирает. Выше = тише рой, но может выглядеть мёртвым." value={cfg.physics.sleepThreshold} min={0} max={20} step={0.5} onChange={(v) => set("physics.sleepThreshold", v)} fmt={(v)=>`${v.toFixed(1)}px/s`} />
          <p className="text-[10px] text-neutral-500 leading-relaxed -mt-1 mb-2">
            0 — дрыгание от коллизий; выше — рой замирает при остановке. 3–6 — золотая середина.
          </p>
          <Slider label="Макс. скорость" desc="Максимальная скорость рыбок. Ограничивает как быстро они могут двигаться к объекту." value={cfg.physics.maxSpeed} min={50} max={800} step={10} onChange={(v) => set("physics.maxSpeed", v)} />
          <Slider label="Отлёт за край (удаление)" desc="На сколько пикселей за край холста может уплыть рыбка перед удалением. 0 = удаляются сразу за краем." value={cfg.physics.cullMargin} min={0} max={600} step={10} onChange={(v) => set("physics.cullMargin", v)} fmt={(v)=>`${v}px`} />
        </Section>

        <Section title="Поле SDF" open={false}>
          <Slider label="Разрешение поля" desc="Точность SDF-поля расстояний. Выше = точнее край силуэта, но дольше пересчёт при загрузке маски." value={cfg.field.res} min={80} max={400} step={10} onChange={(v) => set("field.res", v)} />
          <p className="text-[10px] text-neutral-500 leading-relaxed mt-1">
            Выше — точнее край силуэта, но дольше пересчёт при загрузке маски и
            смене разрешения.
          </p>
        </Section>
      </div>
    </div>
  );
}
