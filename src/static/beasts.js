// How a creature looks, from its genes (the look the owner approved in the gallery, 2026-11-26, and the new look of
// 2026-11-28: every gene shows, the coat's colour from the genes; docs/superpowers/specs/2026-11-28-new-look-design.md): a body plan that
// blends a grazer (meat 0), a hunter (meat 1: its meat gut) and a swimmer (swim 1: its gills), and on it every other
// gene's own sign.
// Low-poly and flat-shaded; the parts sharing a material are merged; bodies with the same rounded genes share their
// geometry. Near the camera the full body; far away a simple one of its body plan, drawn in batches (one per plan,
// each body in its own coat colour): this box draws without a graphics card, so every draw counts.
import * as THREE from "three";
import { mergeGeometries } from "/static/BufferGeometryUtils.js";

export const NEAR = 15;  // closer than this to the camera: the full body
export const DOT = 40;  // farther than this: the simplest body, its trunk and head (a few pixels on the screen)
const farMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.75 });  // every far body
const mix = (a, b, t) => a + (b - a) * t;
const clamp01 = (v) => Math.max(0, Math.min(1, v));

// ---------- skins: a grazer's dapples, a hunter's stripes, a swimmer's scales; learning speed adds dark bands
let seed = 5;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const skins = new Map();
function skin(kind, bands) {
  const key = kind + bands;
  if (skins.has(key)) return skins.get(key);
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"), s = 128;
  g.fillStyle = "#fff"; g.fillRect(0, 0, s, s);
  if (kind === "plant") for (let i = 0; i < 40; i++) {
    g.fillStyle = `rgba(255,250,235,${0.5 + rnd() * 0.4})`; g.beginPath();
    g.ellipse(rnd() * s, rnd() * s, 3 + rnd() * 5, 2 + rnd() * 3, rnd() * 3, 0, 7); g.fill();
  }
  if (kind === "meat") for (let i = 0; i < 10; i++) {
    const x = i * s / 10 + rnd() * 3; g.fillStyle = "rgba(20,10,5,0.55)"; g.beginPath();
    g.moveTo(x, 0); g.quadraticCurveTo(x + 5, s / 2, x - 2, s); g.lineTo(x + 3, s); g.quadraticCurveTo(x + 9, s / 2, x + 4, 0); g.fill();
  }
  if (kind === "water") for (let y = 0; y < s; y += 6) for (let x = (y / 6) % 2 * 4; x < s; x += 8) {
    g.strokeStyle = "rgba(0,30,40,0.22)"; g.beginPath(); g.arc(x, y, 4, 0, Math.PI); g.stroke();
  }
  for (let k = 0; k < bands; k++) { const x = s * (0.25 + 0.5 * (k + 0.5) / bands); g.fillStyle = "rgba(25,15,5,0.5)"; g.fillRect(x - 3, 0, 6, s); }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace;
  skins.set(key, t);
  return t;
}

// ---------- the genes a body shows, read from a creature of the live view (rounded, so alike bodies share a model)
function genesOf(c) {  // every gene the body shows, 0..1 in tenths of its range (layers and calls as they are)
  const step = (v) => Math.round(clamp01(v) * 10) / 10;
  const lr = c.learning_speed ? clamp01((Math.log10(c.learning_speed) + 5) / 3) : 0;  // 1e-5 .. 1e-2
  const layers = Math.max(1, Math.round(c.layers || 1));
  // each layer's neurons over 64 (a layer's most at the start): a creature's own layers (`brain` in the view), or a
  // species' typical neurons shared evenly over its layers
  const each = c.brain || Array(layers).fill(c.neurons / layers || 0);
  return {
    size: 1,  // the body is scaled by its size when it is placed (dress)
    meat: step(c.meat_gut ?? 0), swim: step(c.gills ?? 0), lungs: step(c.lungs ?? 0.5), grass: step(c.grass_gut ?? 1),
    love: step((c.love_of_babies ?? 30) / 100), fear: step((c.fear_of_death ?? 0) / 100),
    layers, brain: each.slice(0, layers).map((n) => step(n / 64)),
    learns: step(c.learns_from_others ?? 0), learning: step(lr),
    width: step(((c.vision_width ?? 75) - 30) / 90), reach: step(((c.vision_reach ?? 5) - 1) / 29),
    hearing: step((c.hearing ?? 1) / 4), armour: step(c.armour ?? 0), stealth: step(c.stealth ?? 0),
    calls: Math.round(c.calls || 0), kin: step(c.kin ?? 0),
  };
}
const keyOf = (g) => Object.values(g).join(",");

function coatOf(g) {  // the coat: its hue by diet and water (green-gold grazer, red hunter, blue swimmer); bright by
  // fear of death (warning colours), dull by stealth (camouflage), dark by armour
  const hue = mix(mix(0.3, 0.0, g.meat), 0.6, 0.9 * g.swim);
  const sat = (0.55 + 0.45 * g.fear) * (1 - 0.8 * g.stealth), light = 0.5 - 0.25 * g.armour;
  return new THREE.Color().setHSL(hue, sat, light);
}
function markOf(g) {  // the markings (belly, frill, antennae tips): pink by love of babies, gold by care for kin,
  // violet by learning from others; pale when all three are low
  const parts = [[0.93, g.love], [0.13, g.kin], [0.76, g.learns]];
  let x = 0, y = 0, sum = 0;
  for (const [h, a] of parts) { x += a * Math.cos(h * 2 * Math.PI); y += a * Math.sin(h * 2 * Math.PI); sum += a; }
  const hue = ((Math.atan2(y, x) / (2 * Math.PI)) + 1) % 1;
  return new THREE.Color().setHSL(hue, Math.min(0.95, 0.2 + sum * 0.7), 0.62);
}

// ---------- one body (genes 0..1 except layers 1-3 and calls 0-10), full or simple
export function beastModel(g, detail = "near") {
  const p = g.meat, w = g.swim, h = 1 - p;
  const parts = { skin: [], dark: [], horn: [], white: [], eye: [], fin: [], accent: [], sac: [], mark: [] };
  const put = (list, geo, x, y, z, rx = 0, ry = 0, rz = 0, sx = 1, sy = 1, sz = 1) => {
    const m = new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)),
                                          new THREE.Vector3(sx, sy, sz));
    parts[list].push(geo.clone().applyMatrix4(m));
  };
  const len = mix(1.15, 1.5, p) * mix(1, 1.25, w), tall = mix(0.72, 0.5, p) * mix(1, 0.8, w), wide = mix(0.68, 0.48, p) * mix(1, 0.85, w);
  const legLen = mix(0.62, 0.42, p) * Math.max(0, 1 - 1.1 * w);  // gills 1: no legs, a fish
  const bodyY = legLen + tall * 0.42, round = g.love;
  const near = detail === "near", dot = detail === "dot";
  const ball = (d) => new THREE.IcosahedronGeometry(0.5, near ? d : 0);
  const chest = mix(0.7, 1.6, g.lungs);  // lungs: a deep, round chest
  put("skin", ball(1), len * 0.18, bodyY + tall * 0.05 * h, 0, 0, 0, 0, len * 0.62, tall * mix(1, 1.15, p) * chest, wide * (1 + 0.45 * round) * chest);
  put("skin", ball(1), -len * 0.22, bodyY, 0, 0, 0, 0, len * 0.6, tall * mix(1.05, 0.85, p), wide * (0.95 + 0.45 * round));
  if (!dot && g.grass > 0.1) put("mark", ball(1), -len * 0.05, bodyY - tall * 0.3, 0, 0, 0, 0, len * (0.4 + 0.6 * g.grass), tall * (0.3 + 0.85 * g.grass), wide * (0.6 + 0.85 * g.grass));  // grass gut: a big pale belly
  const social = (g.love + g.kin + g.learns) / 3;
  if (!dot && social > 0.1) put("mark", ball(1), -len * 0.02, bodyY + tall * 0.36, 0, 0, 0, 0, len * (0.55 + 0.45 * social), tall * 0.42, wide * (1.02 + 0.45 * g.love));  // markings: a saddle on the back
  const headR = (0.15 + 0.3 * g.brain[0]) * mix(1, 1.2, p), neckLen = mix(0.55, 0.22, p) * (1 - 0.95 * w);  // layer 1: the head; gills: no neck
  const nx = len * 0.5, ny = bodyY + tall * 0.2, neckA = mix(0.9, 0.25, p);
  if (!dot) put("skin", new THREE.CylinderGeometry(0.11, 0.17, neckLen, near ? 7 : 4), nx + Math.sin(neckA) * neckLen * 0.5, ny + Math.cos(neckA) * neckLen * 0.5, 0, 0, 0, -neckA);
  const hx = nx + Math.sin(neckA) * neckLen + headR * 0.4, hy = ny + Math.cos(neckA) * neckLen;
  put("skin", new THREE.IcosahedronGeometry(headR, near ? 1 : 0), hx, hy, 0, 0, 0, 0, mix(1.15, 1.35, p) * (1 + 0.8 * w), mix(0.95, 0.75, p) * (1 - 0.2 * w), mix(0.9, 1.0, p) * (1 - 0.15 * w));  // a swimmer: a long, flat fish head
  // legs: a grazer's straight with hooves, a hunter's crouched and clawed, a swimmer's short and webbed
  if (!dot && legLen > 0.05) for (const [fx, s] of [[0.32, 1], [0.32, -1], [-0.32, 1], [-0.32, -1]]) {
    const lx = fx * len, lz = s * wide * 0.32;
    if (p < 0.5 || !near) put("skin", new THREE.CylinderGeometry(0.075, 0.06, legLen, near ? 6 : 3), lx, legLen / 2, lz);
    else {
      const up = legLen * 0.55, low = legLen * 0.55, bend = fx > 0 ? 0.35 : -0.45;
      put("skin", new THREE.CylinderGeometry(0.08, 0.06, up, 6), lx - Math.sin(bend) * up * 0.5, legLen - up * 0.45, lz, 0, 0, bend);
      put("skin", new THREE.CylinderGeometry(0.055, 0.045, low, 6), lx - Math.sin(bend) * up + Math.sin(bend) * low * 0.5, low * 0.45, lz, 0, 0, -bend);
    }
  }
  if (near) {
    if (h > 0.5) put("skin", new THREE.IcosahedronGeometry(0.3, 1), len * 0.05, bodyY + tall * 0.42, 0, 0, 0, 0, 1.4, 0.55, 1.1);  // a grazer's hump
    if (w > 0.6) {  // a fish face: a wide mouth line and lips, no muzzle
      put("dark", new THREE.BoxGeometry(headR * 0.9, 0.035, headR * 1.1), hx + headR * 1.1, hy - headR * 0.3, 0, 0, 0, -0.1);
      put("sac", new THREE.IcosahedronGeometry(headR * 0.28, 1), hx + headR * 1.55, hy - headR * 0.25, 0, 0, 0, 0, 0.6, 0.7, 1.3);
    } else if (p < 0.5) {  // a grazer's long muzzle and pale nose
      put("skin", new THREE.CylinderGeometry(headR * 0.55, headR * 0.75, headR * 1.3, 7), hx + headR * 1.05, hy - headR * 0.25, 0, 0, 0, Math.PI / 2 + 0.35);
      put("dark", new THREE.IcosahedronGeometry(headR * 0.42, 0), hx + headR * 1.7, hy - headR * 0.45, 0, 0, 0, 0, 1, 0.8, 1.2);
    } else {  // a hunter's jaws, slightly open, fangs between, and a brow ridge
      put("skin", new THREE.ConeGeometry(headR * 0.62, headR * 1.6, 5), hx + headR * 1.25, hy + headR * 0.05, 0, 0, 0, -Math.PI / 2, 1, 1, 0.8);
      put("dark", new THREE.ConeGeometry(headR * 0.45, headR * 1.3, 5), hx + headR * 1.1, hy - headR * 0.42, 0, 0, 0, -Math.PI / 2 - 0.25, 1, 1, 0.75);
      for (const s of [1, -1]) {
        put("white", new THREE.ConeGeometry(0.035 + 0.02 * p, 0.12 + 0.12 * p, 4), hx + headR * 1.45, hy - headR * 0.2, s * headR * 0.22, Math.PI);
        put("white", new THREE.ConeGeometry(0.03, 0.09 + 0.08 * p, 4), hx + headR * 1.25, hy - headR * 0.32, s * headR * 0.18, 0);
      }
      put("dark", new THREE.BoxGeometry(headR * 0.9, 0.05, headR * 1.3), hx + headR * 0.25, hy + headR * 0.55, 0, 0, 0, -0.25);
    }
    // eyes: a grazer's to the sides, a hunter's to the front (vision width moves them); their size its vision reach
    const eyeR = 0.035 + 0.12 * g.reach, sideA = Math.min(1.45, mix(1.2, 0.45, p) + 0.5 * (g.width - 0.5) * 2);
    for (const s of [1, -1]) {
      const ex = hx + headR * 0.75 * Math.cos(sideA), ez = s * headR * 0.85 * Math.sin(sideA);
      put("white", new THREE.IcosahedronGeometry(eyeR, 1), ex, hy + headR * 0.3, ez);
      put("eye", new THREE.IcosahedronGeometry(eyeR * 0.55, 0), ex + eyeR * 0.6 * Math.cos(sideA), hy + headR * 0.32, ez + s * eyeR * 0.6 * Math.sin(sideA), 0, 0, 0, p > 0.5 ? 0.5 : 1, 1, 1);
    }
    // ears by hearing: a grazer's broad leaves to the side, a hunter's pointed and upright
    if (g.hearing > 0.08) for (const s of [1, -1]) {
      const e = (0.08 + 0.95 * g.hearing) * (1 - 0.8 * w);  // big ears at full hearing; small on a fish
      if (p < 0.5) put("skin", new THREE.ConeGeometry(e * 0.5, e * 1.6, 4), hx - headR * 0.3, hy + headR * 0.6, s * headR * 0.7, s * 1.1, 0, 0.4, 1, 1, 0.35);
      else put("skin", new THREE.ConeGeometry(e * 0.4, e * 1.4, 4), hx - headR * 0.35, hy + headR * 0.95, s * headR * 0.45, s * 0.15, 0, 0.1, 1, 1, 0.6);
    }
    // brain layers 2 and 3: a grazer's horns grow a tine each, a hunter's skull a crest plate each, as big as that layer's neurons
    if (h > 0.5 && w <= 0.6) for (const s of [1, -1]) {
      put("horn", new THREE.TorusGeometry(headR * 0.75, 0.035, 4, 8, Math.PI * 0.8), hx - headR * 0.25, hy + headR * 0.75, s * headR * 0.35, 0, s * 0.5, 0.2);
      for (let k = 1; k < g.layers; k++) { const L = 0.1 + 0.55 * (g.brain[k] ?? 0.5); put("horn", new THREE.ConeGeometry(0.03 + 0.04 * (g.brain[k] ?? 0.5), L, 4), hx - headR * (0.5 + 0.25 * k), hy + headR * 1.2 + L * 0.45, s * headR * (0.5 + 0.2 * k), 0, 0, 0.6 - 0.2 * k); }
    } else for (let k = 1; k < g.layers; k++) { const L = 0.12 + 0.45 * (g.brain[k] ?? 0.5); put("accent", new THREE.ConeGeometry(L * 0.45, L, 3), hx - headR * 0.1 - (k - 1) * L * 0.6, hy + headR * 0.8 + L * 0.3, 0, 0, 0, 0.4, 1, 1, 0.3); }
    if (g.learns > 0.05) for (const s of [1, -1]) {  // antennae: learning from others
      const L = 0.15 + 0.85 * g.learns;
      put("dark", new THREE.CylinderGeometry(0.012, 0.012, L, 4), hx + headR * 0.2, hy + headR + L * 0.45, s * 0.07, s * 0.35, 0, -0.2);
      put("mark", new THREE.IcosahedronGeometry(0.06, 0), hx + headR * 0.2 + L * 0.2, hy + headR + L * 0.9, s * (0.07 + L * 0.33));
    }
    const tx = -len * 0.55, ty = bodyY + tall * 0.05;  // tail: a tuft, a long taper, or a fin
    if (w > 0.4) put("fin", new THREE.ConeGeometry(0.28 + 0.2 * w, 0.5, 3), tx - 0.25, ty, 0, 0, 0, Math.PI / 2, 1, 1, 0.12);
    else if (p > 0.5) put("skin", new THREE.ConeGeometry(0.08, 0.9 * p, 5), tx - 0.4 * p, ty - 0.05, 0, 0, 0, Math.PI / 2 + 0.35);
    else { put("skin", new THREE.CylinderGeometry(0.02, 0.025, 0.28, 4), tx - 0.08, ty - 0.12, 0, 0, 0, 0.6); put("dark", new THREE.ConeGeometry(0.06, 0.12, 5), tx - 0.16, ty - 0.27, 0, 0, 0, Math.PI); }
    if (w > 0.15) {  // a swimmer's back fin, side fins and gills
      put("fin", new THREE.ConeGeometry(0.3 * w + 0.1, 0.55 * w + 0.15, 3), -len * 0.05, bodyY + tall * 0.55, 0, 0, Math.PI / 2, -0.35, 1, 1, 0.12);
      for (const s of [1, -1]) put("fin", new THREE.ConeGeometry(0.3 * w + 0.05, 0.7 * w + 0.1, 3), len * 0.2, bodyY - tall * 0.25, s * wide * 0.55, s * 1.2, 0, -0.9, 1, 1, 0.12);  // flippers in place of legs
      for (let k = 0; k < 5; k++) for (const s of [1, -1]) put("dark", new THREE.BoxGeometry(0.02, headR * 0.9 * w + 0.05, 0.01), hx - headR * (0.65 + 0.12 * k), hy - headR * 0.05, s * headR * 0.85, 0, 0, 0.2);
    }
    const quills = Math.round(g.fear * 14);  // fear of death: quills down the back
    for (let k = 0; k < quills; k++) for (const s of [1, -1]) {
      const t = (k + 0.5) / quills - 0.5;
      put("dark", new THREE.ConeGeometry(0.035, 0.25 + 0.6 * g.fear, 4), t * len * 0.9, bodyY + tall * 0.5, s * wide * 0.13, s * 0.35, 0, -t * 1.0 + 0.5);
    }
    const plates = Math.round(g.armour * 9);  // armour: bands of bone; at full armour a shell over the whole body
    for (let k = 0; k < plates; k++) {
      const t = (k + 0.5) / 9 - 0.5;
      put("horn", new THREE.IcosahedronGeometry(0.5, 1), t * len * 1.05, bodyY + tall * 0.12, 0, 0, 0, -0.1, len * 0.16, tall * (0.75 + 0.45 * g.armour), wide * (1.15 + 0.3 * g.armour));
    }
    if (g.armour > 0.7) put("horn", new THREE.IcosahedronGeometry(headR * 0.9, 1), hx - headR * 0.1, hy + headR * 0.35, 0, 0, 0, 0, 1.2, 0.6, 1.1);  // a helmet
    for (const s of [1, -1]) put("dark", new THREE.IcosahedronGeometry(0.02 + 0.07 * g.lungs, 0), hx + headR * mix(1.75, 1.95, p) * (1 + 0.2 * w), hy + headR * 0.05, s * headR * 0.22);  // lungs: nostrils
    if (g.calls > 0) put("sac", new THREE.IcosahedronGeometry(0.06 + 0.035 * g.calls, 1), hx - headR * 0.1, hy - headR * 0.95, 0);  // calls: a throat sac
    if (g.kin > 0.05) {  // care for kin: a frill around the neck, as wide as the care
      const r = 0.14 + 0.5 * g.kin;
      put("mark", new THREE.CylinderGeometry(r, 0.12, 0.05, 14), nx + Math.sin(neckA) * neckLen * 0.35, ny + Math.cos(neckA) * neckLen * 0.35, 0, 0, 0, -neckA - 0.25);
    }
  }
  const base = coatOf(g), markColour = markOf(g);
  const kind = w > 0.5 ? "water" : p > 0.5 ? "meat" : "plant";
  const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, flatShading: true, roughness: 0.75, ...extra });
  const mats = {
    skin: mat(base, { map: near ? skin(kind, Math.round(g.learning * 4)) : null, roughness: mix(0.7, 0.35, w) + 0.25 * g.stealth, metalness: 0.1 * w }),
    dark: mat(base.clone().multiplyScalar(0.45)), horn: mat(0xd8ccb0, { roughness: 0.5 }), white: mat(0xf4f1e6, { roughness: 0.3 }),
    eye: mat(p > 0.5 ? 0xffb020 : 0x1a1208, { emissive: p > 0.5 ? 0x5a2a00 : 0x000000, roughness: 0.2 }),
    fin: mat(new THREE.Color(0.25, 0.6, 0.8), { transparent: true, opacity: 0.85, side: THREE.DoubleSide }),
    accent: mat(markColour, { roughness: 0.4 }), sac: mat(0xf08aa8, { roughness: 0.25 }), mark: mat(markColour, { roughness: 0.5 }),
  };
  const group = new THREE.Group();
  if (!near) {  // far away: one mesh, each part's colour on its corners; the skin white, so a batch paints each coat
    const all = [], white = new THREE.Color(1, 1, 1);
    for (const [name, geos] of Object.entries(parts)) {
      for (const x of geos) {
        const c = name === "skin" ? white : name === "dark" ? new THREE.Color(0.45, 0.45, 0.45) : mats[name].color;
        const geo = x.index ? x.toNonIndexed() : x, n = geo.attributes.position.count, rgb = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) rgb.set([c.r, c.g, c.b], i * 3);
        geo.deleteAttribute("uv"); geo.setAttribute("color", new THREE.BufferAttribute(rgb, 3));
        all.push(geo);
      }
    }
    group.add(new THREE.Mesh(mergeGeometries(all), farMaterial));
    return group;
  }
  for (const [name, geos] of Object.entries(parts)) {
    if (!geos.length) continue;
    group.add(new THREE.Mesh(mergeGeometries(geos.map((x) => (x.index ? x.toNonIndexed() : x))), mats[name]));
  }
  return group;
}

// ---------- what the planet uses: near, a full body for each creature (shared by alike genes); far, a simple body of
// its plan (meat gut and gills in quarters) drawn in one batch per plan, coloured by its coat; every hunger ring in one batch
const models = new Map(), farModels = new Map();
const PLAIN = { size: 1, lungs: 0.5, grass: 1, love: 0.3, fear: 0, layers: 1, brain: [0.5], learns: 0, learning: 0, width: 0.5, reach: 0.5,
                hearing: 0.4, armour: 0, stealth: 0, calls: 0, kin: 0 };
const ringColours = [...Array(11).keys()].map((k) => new THREE.Color().setHSL(0.033 * k, 0.85, 0.5));
function ringColour(energy) { return ringColours[Math.round(clamp01(energy) * 10)]; }  // green full, yellow half, red hungry
function farGeometry(key) {
  if (!farModels.has(key)) {
    const [meat, swim, detail] = key.split(",");
    farModels.set(key, beastModel({ ...PLAIN, meat: Number(meat), swim: Number(swim) }, detail).children[0].geometry);
  }
  return farModels.get(key);
}
// A small picture of one body as the planet draws it (genes 0..1, as `beastModel`), for the legend: drawn once each.
let pictureRenderer = null;
const pictures = new Map();
export function bodyPicture(genes, size = 120) {
  const key = JSON.stringify([genes, size]);  // kept by its size too: a big picture is never a small one stretched
  if (pictures.has(key)) return pictures.get(key);
  if (!pictureRenderer) {
    pictureRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    pictureRenderer.setPixelRatio(1);
  }
  pictureRenderer.setSize(size, size, false);
  pictureRenderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xe6f0ff, 0x40301c, 1.1));
  const sun = new THREE.DirectionalLight(0xfff0d0, 2.0); sun.position.set(5, 10, 7); scene.add(sun);
  const g = { ...PLAIN, meat: 0, swim: 0, ...genes }, body = beastModel(g, "near");
  body.scale.setScalar(Math.min(g.size, 1.4)); body.rotation.y = -0.75; scene.add(body);
  // the camera at its usual distance, or further when the body would not fit, aimed at the body's middle: no part cut
  const box = new THREE.Box3().setFromObject(body), middle = box.getCenter(new THREE.Vector3());
  const fits = (box.getSize(new THREE.Vector3()).length() / 2) / Math.sin(THREE.MathUtils.degToRad(14));
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.copy(middle).addScaledVector(new THREE.Vector3(0, 1.65, 6.4).normalize(), Math.max(6.6, fits));
  camera.lookAt(middle);
  pictureRenderer.render(scene, camera);
  const c = document.createElement("canvas"); c.width = c.height = size;
  c.getContext("2d").drawImage(pictureRenderer.domElement, 0, 0);
  pictures.set(key, c);
  return c;
}

// A species' typical member (its `now` or `start` in species.json: each gene its members' mean), drawn as the planet
// draws a creature with those genes; the species' names for the feelings are the creature's own.
export function speciesPicture(t, size = 120) {
  return genesPicture({ ...t, fear_of_death: t.fear, love_of_babies: t.love, learns_from_others: t.learns }, size);
}

// A creature's genes by the world's own names (`brain`: each layer's neurons, or a species' `neurons` and `layers`), drawn as the planet draws that creature: the
// species card's and the Make page's bodies.
export function genesPicture(c, size = 120) {
  return bodyPicture({ ...genesOf(c), size: c.size ?? 1 }, size);
}

export const BEASTS = {
  key: (c) => keyOf(genesOf(c)),
  make(c) {  // a full body, for a creature near the camera
    const g = genesOf(c), key = keyOf(g);
    if (!models.has(key)) models.set(key, beastModel(g, "near"));
    return models.get(key).clone();
  },
  dress(body, c) { body.scale.setScalar(c.radius); },  // a model is about 2 long: drawn at its body's true size
  farKey(c, detail) { const g = genesOf(c); return `${g.meat},${g.swim},${detail}`; },  // detail: far or dot
  farGeometry,
  farMaterial,
  coat: (c) => coatOf(genesOf(c)),
  ringGeometry: new THREE.RingGeometry(0.78, 0.92, 24).rotateX(-Math.PI / 2).scale(1.2, 1.2, 1.2).translate(0, 0.02, 0),
  ringMaterial: new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthWrite: false }),
  ring: (c) => ringColour(c.energy),
};
