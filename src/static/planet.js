// The live view of a watched world: its planet in 3D (three.js, served from this repository).
// The page hands it each picture of the world (`/view`: points on the planet, facings, terrain caps) and draws it:
// textured grass, its seas, lakes and rivers (a picture of the world's water, `/water`), rocks rising out of it, ponds,
// forests, food, creatures at their true size; drag to spin, wheel or pinch to zoom; click
// a creature to follow it; its point of view shows only what it senses (the world's own rules, src/senses.py).
import * as THREE from "three";
import { OrbitControls } from "/static/OrbitControls.js";
import { mergeGeometries } from "/static/BufferGeometryUtils.js";
import { BEASTS, NEAR, DOT } from "/static/beasts.js";
export { bodyPicture, speciesPicture, genesPicture } from "/static/beasts.js";  // the legend's, the species' and the
// Make page's bodies (the page has only this module)

const UP = new THREE.Vector3(0, 1, 0);
const NEAR_FAR = { near: 0.05, far: 4000 };
const ABOVE_FOV = 45;  // degrees, top to bottom: the free camera's (from its eyes, as wide as its vision)

let state = null;  // the one planet on the page

const look = BEASTS;  // how creatures look: their genes' bodies (src/static/beasts.js)

// ---------- small helpers
const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const WATER_LIFT = 0.03;  // the water's surface above the ground: the rings are drawn above it, on land and in water
function slerp(a, b, t) {  // between two unit vectors, along the ball
  const dot = Math.max(-1, Math.min(1, a.dot(b)));
  const angle = Math.acos(dot);
  if (angle < 1e-6) return a.clone();
  const s = Math.sin(angle);
  return a.clone().multiplyScalar(Math.sin((1 - t) * angle) / s).add(b.clone().multiplyScalar(Math.sin(t * angle) / s));
}
function standOn(object, p, R, lift, facing) {  // object's +y along the ground's normal at p, its +x along `facing`
  const n = p.clone().normalize();
  object.position.copy(n).multiplyScalar(R + lift);
  if (facing) {
    const f = facing.clone().sub(n.clone().multiplyScalar(facing.dot(n))).normalize();
    const side = new THREE.Vector3().crossVectors(f, n);
    object.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(f, n, side));
  } else {
    object.quaternion.setFromUnitVectors(UP, n);
  }
}
function noise3(x, y, z, seed) {  // smooth value noise from a few sines: enough for grass patches and rock shapes
  return Math.sin(x * 1.7 + seed) * Math.sin(y * 2.3 + seed * 1.3) * Math.sin(z * 1.9 + seed * 0.7)
    + 0.5 * Math.sin(x * 4.1 + seed * 2.1) * Math.sin(y * 3.7 + seed) * Math.sin(z * 4.3 + seed * 1.7);
}

// ---------- the ground: a fine ball, green with lighter and darker patches, and a blade texture laid on it
function bladeTexture(renderer) {
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#ffffff"; g.fillRect(0, 0, 256, 256);
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 6000; i++) {
    const x = rnd() * 256, y = rnd() * 256, l = 55 + rnd() * 45;
    g.strokeStyle = `hsl(0, 0%, ${l}%)`; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + (rnd() - 0.5) * 3, y - 2 - rnd() * 5); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}
function makeGround(R, renderer) {
  const geometry = new THREE.SphereGeometry(R, 96, 64);  // light enough to draw without a graphics card (192 x 128 cost 10 ms a frame)
  const pos = geometry.attributes.position, colours = new Float32Array(pos.count * 3), c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) / R, y = pos.getY(i) / R, z = pos.getZ(i) / R;
    const patch = noise3(x * 5, y * 5, z * 5, 3.1), broad = noise3(x * 2, y * 2, z * 2, 1.7);  // smooth: the fine look is the blades
    c.setHSL(0.27 + 0.025 * patch, 0.42 + 0.05 * broad, 0.27 + 0.05 * patch + 0.02 * broad);
    colours.set([c.r, c.g, c.b], i * 3);
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  const blades = bladeTexture(renderer);
  blades.repeat.set(Math.round(R * 4.4), Math.round(R * 2.2));  // about two tiles per unit of ground
  return new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ vertexColors: true, map: blades }));
}

// ---------- rocks rising out of the grass, ponds a little sunk, trees in their forests
function rockGeometry(seed) {
  const geometry = new THREE.IcosahedronGeometry(1, 2), pos = geometry.attributes.position, w = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    w.fromBufferAttribute(pos, i);
    w.multiplyScalar(1 + 0.25 * noise3(w.x * 1.6, w.y * 1.6, w.z * 1.6, seed));
    if (w.y < 0) w.y *= 0.25;  // its flat foot sunk in the ground
    pos.setXYZ(i, w.x, w.y, w.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}
function makeTerrain(terrain, R) {
  const group = new THREE.Group();
  const rockMat = new THREE.MeshStandardMaterial({ color: 0x8b847c, roughness: 0.95, flatShading: true });
  const pondMat = new THREE.MeshStandardMaterial({ color: 0x2f6fb8, roughness: 0.15, metalness: 0.1 });
  let seed = 1;
  const rocks = [], ponds = [], at = new THREE.Object3D();  // every rock in one piece, every pond in one: two draws
  for (const o of terrain.obstacles || []) {
    for (const cap of o.caps) {
      const p = v3(cap), r = cap[3];
      at.scale.set(1, 1, 1);
      if (o.kind === "rock") {
        const geometry = rockGeometry(seed++);
        standOn(at, p, R, -0.15 * r);
        at.scale.set(r, r * (0.8 + 0.5 * ((seed * 0.37) % 1)), r);
        at.rotateY(seed * 1.7);
        at.updateMatrix(); rocks.push(geometry.applyMatrix4(at.matrix));
      } else {
        standOn(at, p, R, 0.03);
        at.updateMatrix(); ponds.push(new THREE.CircleGeometry(r, 40).rotateX(-Math.PI / 2).applyMatrix4(at.matrix));
      }
    }
  }
  if (rocks.length) group.add(new THREE.Mesh(mergeGeometries(rocks), rockMat));
  if (ponds.length) group.add(new THREE.Mesh(mergeGeometries(ponds), pondMat));
  const trees = terrain.trees || [];
  if (trees.length) {
    const trunks = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.16, 0.24, 1.3, 7).translate(0, 0.65, 0),
                                           new THREE.MeshStandardMaterial({ color: 0x5b3d22 }), trees.length);
    const crowns = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1).translate(0, 2.1, 0),
                                           new THREE.MeshStandardMaterial({ color: 0x2c5f2a, flatShading: true }), trees.length);
    const o = new THREE.Object3D();
    trees.forEach((t, i) => {
      standOn(o, v3(t.p), R, 0); o.rotateY(i * 2.39); o.scale.setScalar(t.canopy / 1.5); o.updateMatrix();
      trunks.setMatrixAt(i, o.matrix);
      o.scale.set(t.canopy, t.canopy, t.canopy); o.updateMatrix(); crowns.setMatrixAt(i, o.matrix);
    });
    group.add(trunks, crowns);
  }
  return group;
}

// ---------- food: tan seeds, red berries (fruit), dark lumps (meat)
// A corpse: a lying body (a flattened grey-brown trunk and head) with pale ribs, unlike grass, fruit or a creature
// (the owner, 2026-11-28: corpses "should look different then regualr food"); its colours on its corners.
const corpseGeometry = (() => {
  const part = (geo, colour, x, y, z, sx, sy, sz, rz = 0) => {
    const g = geo.toNonIndexed().scale(sx, sy, sz).rotateZ(rz).translate(x, y, z), c = new THREE.Color(colour);
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 3).map((_, i) => [c.r, c.g, c.b][i % 3]), 3));
    g.deleteAttribute("uv");
    return g;
  };
  const parts = [part(new THREE.IcosahedronGeometry(0.5, 1), 0x6b5a4e, 0, 0.12, 0, 1.6, 0.35, 0.8),
                 part(new THREE.IcosahedronGeometry(0.28, 0), 0x5e4e44, 0.95, 0.1, 0, 1.2, 0.6, 0.9)];
  for (let k = 0; k < 4; k++) parts.push(part(new THREE.CylinderGeometry(0.03, 0.03, 0.9, 5).rotateX(Math.PI / 2), 0xe8dcc4, -0.3 + k * 0.2, 0.28, 0, 1, 1, 1));
  return mergeGeometries(parts);
})();

const FOOD_KINDS = {
  plant: { geometry: new THREE.OctahedronGeometry(1, 0), material: new THREE.MeshStandardMaterial({ color: 0xc9a46a, flatShading: true }) },
  fruit: { geometry: new THREE.IcosahedronGeometry(1, 1), material: new THREE.MeshStandardMaterial({ color: 0xe0483b, roughness: 0.4 }) },
  meat: { geometry: corpseGeometry, material: new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.8 }) },
};
const foodKind = (f) => (f.fruit ? "fruit" : f.meat ? "meat" : "plant");

// ---------- making the planet
export function makePlanet(container, { onPick } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  container.innerHTML = "";
  container.appendChild(renderer.domElement);
  renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%"; renderer.domElement.style.display = "block";
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d12);
  scene.add(new THREE.HemisphereLight(0xdfefff, 0x3a2a1a, 0.9));
  const sun = new THREE.DirectionalLight(0xfff3dd, 1.6); sun.position.set(90, 70, 110);
  const camera = new THREE.PerspectiveCamera(ABOVE_FOV, 1, NEAR_FAR.near, NEAR_FAR.far);
  camera.add(sun); scene.add(camera);  // the sun over the viewer's shoulder: the side in view is lit
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.enablePan = false; controls.zoomSpeed = 1.1; controls.rotateSpeed = 0.6;
  state = { container, renderer, scene, camera, controls, onPick, R: null, terrainKey: null, ground: null, terrain: null,
            food: {}, beasts: new THREE.Group(), herd: new THREE.Group(), batches: new Map(), marks: new THREE.Group(), follow: null, view: "above", picture: null,
            dragged: false };
  scene.add(state.beasts, state.herd, state.marks);
  const size = () => {
    const w = container.clientWidth || 800, h = container.clientHeight || 500;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  new ResizeObserver(size).observe(container); size();
  let down = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;  // a drag spins; a click picks
    pick(e);
  });
  return state;
}

function pick(event) {
  if (!state) return;
  if (state.onGround) return state.onGround(event);  // placing the owner's creature: the click is for the ground
  if (state.onPick) state.onPick(creatureAt(event));
}
// While the owner places a creature, each click on the planet goes to `onGround` (null: clicks pick creatures again).
export function placing(onGround) { if (state) state.onGround = onGround; }

// ---------- each picture of the world
export function updatePlanet(picture, before, t, options) {
  if (!state) return;
  const R = picture.radius;
  state.picture = picture;
  showWater(options && options.run, R);
  if (state.R !== R) {  // a new planet: its ground, and the camera out where the whole ball shows
    if (state.ground) state.scene.remove(state.ground);
    state.ground = makeGround(R, state.renderer); state.scene.add(state.ground);
    state.R = R;
    state.controls.minDistance = R + 2.5; state.controls.maxDistance = R * 7;  // closest: 2.5 above the ground
    state.camera.position.set(0, R * 0.6, R * 3.0); state.controls.target.set(0, 0, 0);
  }
  const terrainKey = JSON.stringify((picture.terrain || {}).obstacles || []).length + ":" + ((picture.terrain || {}).trees || []).length;
  if (state.terrainKey !== terrainKey) {
    if (state.terrain) state.scene.remove(state.terrain);
    state.terrain = makeTerrain(picture.terrain || { obstacles: [], trees: [] }, R); state.scene.add(state.terrain);
    state.terrainKey = terrainKey;
  }
  const earlier = new Map((before ? before.creatures : []).map((c) => [c.id, c]));
  const placed = picture.creatures.map((c) => {
    const from = earlier.get(c.id) || c;
    return { c, p: slerp(v3(from.p), v3(c.p), t), f: v3(c.facing) };
  });
  const followed = state.follow === null ? null : placed.find((x) => x.c.id === state.follow);
  const sensed = followed && state.view === "eyes" ? sensedBy(followed.c, picture) : null;  // its eyes: only what it senses
  state.lastPlaced = placed;
  drawFood(picture, R, sensed);
  drawBeasts(placed, R, sensed, followed, options || {});
  steer(followed, R);
}

// the world's water: its picture (blue where water, clear on land) wrapped on a ball just above the ground
function showWater(run, R) {
  const key = `${run}:${R}`;
  if (!run || state.waterKey === key) return;
  state.waterKey = key;
  if (state.water) { state.scene.remove(state.water); state.water = null; }
  new THREE.TextureLoader().load(`/water?run=${encodeURIComponent(run)}`, (picture) => {
    if (state.waterKey !== key) return;  // another world since
    picture.colorSpace = THREE.SRGBColorSpace;
    picture.generateMipmaps = false; picture.minFilter = THREE.LinearFilter;  // plain sampling: sharpened or mipmapped,
    // this ball cost 12 ms a frame on this box (no graphics card); plain, under 4
    state.water = new THREE.Mesh(new THREE.SphereGeometry(R + WATER_LIFT, 128, 64),
                                 new THREE.MeshLambertMaterial({ map: picture, alphaTest: 0.5 }));
    state.scene.add(state.water);
  }, undefined, () => {});  // a world without water (or from before it): no picture, no water
}

function drawFood(picture, R, sensed) {
  const lists = { plant: [], fruit: [], meat: [] };
  picture.foods.forEach((f, i) => { if (!sensed || sensed.foods.has(i)) lists[foodKind(f)].push(f); });
  const o = new THREE.Object3D();
  for (const [kind, list] of Object.entries(lists)) {
    let mesh = state.food[kind];
    if (!mesh || mesh.count < list.length || mesh.instanceMatrix.count < list.length) {
      if (mesh) state.scene.remove(mesh);
      mesh = new THREE.InstancedMesh(FOOD_KINDS[kind].geometry, FOOD_KINDS[kind].material, Math.max(64, list.length * 2));
      state.food[kind] = mesh; state.scene.add(mesh);
    }
    list.forEach((f, i) => {
      const s = Math.max(0.12, f.radius * Math.sqrt(f.left ?? 1)) * (kind === "plant" || kind === "meat" ? 1.4 : 1.0);  // smaller as it is eaten
      standOn(o, v3(f.p), R, s * 0.6); o.scale.setScalar(s); o.updateMatrix(); mesh.setMatrixAt(i, o.matrix);
    });
    mesh.count = list.length; mesh.instanceMatrix.needsUpdate = true;
  }
}

function drawBeasts(placed, R, sensed, followed, options) {
  // near the camera each creature has its full body; far away it joins its body plan's batch; all rings are one batch
  const keep = new Set(), far = new Map(), eye = state.camera.position, at = new THREE.Object3D(), ringAt = new THREE.Object3D();
  const shown = sensed ? placed.filter((x) => x === followed || sensed.sees.has(x.c.id)) : placed;
  const rings = batch("rings", look.ringGeometry, look.ringMaterial, shown.length);
  shown.forEach((x, i) => {
    standOn(at, x.p, R, 0, x.f); at.scale.setScalar(x.c.radius); at.updateMatrix();  // its true size, as rocks and trees
    standOn(ringAt, x.p, R, WATER_LIFT + 0.01, x.f); ringAt.scale.setScalar(x.c.radius); ringAt.updateMatrix();  // above the water
    rings.setMatrixAt(i, ringAt.matrix); rings.setColorAt(i, look.ring(x.c)); rings.userData.ids[i] = x.c.id;
    const away = at.position.distanceTo(eye);
    if (away >= NEAR) {
      const key = look.farKey(x.c, away >= DOT ? "dot" : "far");
      if (!far.has(key)) far.set(key, []);
      far.get(key).push([x.c, at.matrix.clone()]);
      return;
    }
    keep.add(x.c.id);
    let body = state.beasts.children.find((b) => b.userData.id === x.c.id);
    const key = look.key(x.c);
    if (body && body.userData.key !== key) { state.beasts.remove(body); body = null; }
    if (!body) {
      body = look.make(x.c);
      body.userData.id = x.c.id; body.userData.key = key;
      state.beasts.add(body);
    }
    standOn(body, x.p, R, 0, x.f);
    look.dress(body, x.c, { followed: x === followed, hovered: x.c.id === options.hover });
    if (x === followed && state.view === "eyes") body.visible = false;  // the camera is inside its head
    else body.visible = true;
  });
  done(rings, shown.length);
  for (const b of [...state.beasts.children]) if (!keep.has(b.userData.id)) state.beasts.remove(b);
  for (const [key, mesh] of state.batches) if (key !== "rings" && !far.has(key)) done(mesh, 0);
  for (const [key, list] of far) {
    const mesh = batch(key, look.farGeometry(key), look.farMaterial, list.length);
    list.forEach(([c, matrix], i) => { mesh.setMatrixAt(i, matrix); mesh.setColorAt(i, look.coat(c)); mesh.userData.ids[i] = c.id; });
    done(mesh, list.length);
  }
  drawMarks(shown, R, sensed, followed, options);
  drawActions(shown, R);
}
function batch(key, geometry, material, n) {  // one draw for many: grows when it must, its ids say who is who
  let mesh = state.batches.get(key);
  if (!mesh || mesh.instanceMatrix.count < n) {
    if (mesh) state.herd.remove(mesh);
    mesh = new THREE.InstancedMesh(geometry, material, Math.max(64, n * 2));
    mesh.frustumCulled = false; mesh.userData.ids = [];
    state.batches.set(key, mesh); state.herd.add(mesh);
  }
  return mesh;
}
function done(mesh, n) {
  mesh.count = n; mesh.userData.ids.length = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.boundingSphere = null;  // picking measures it again
}

// vision cones, call rings and the follow ring, flat on the ground
const markMats = {
  cone: new THREE.MeshBasicMaterial({ color: 0xfff2a8, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false }),
  call: new THREE.MeshBasicMaterial({ color: 0xf4d35e, transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }),
  ring: new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, depthWrite: false }),
  heard: new THREE.MeshBasicMaterial({ color: 0xa0b4ff, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false }),
};
const coneCache = new Map();
function coneGeometry(width) {  // a fan of the vision width, radius 1, lying flat, pointing along +x
  const key = Math.round(width / 5) * 5;
  if (!coneCache.has(key)) {
    const half = (key / 2) * Math.PI / 180;
    coneCache.set(key, new THREE.CircleGeometry(1, 24, -half, 2 * half).rotateX(-Math.PI / 2));
  }
  return coneCache.get(key);
}
function drawMarks(shown, R, sensed, followed, options) {
  state.marks.clear();
  const ground = (mesh, p, f, lift) => { standOn(mesh, p, R, lift, f); state.marks.add(mesh); };
  for (const x of shown) {
    const close = state.camera.position.length() < R * 1.6;  // from afar every cone would wash the grass out
    const coned = sensed ? x === followed : (options.cones && close) || x.c.id === options.hover || x === followed;
    if (coned) {
      const cone = new THREE.Mesh(coneGeometry(x.c.vision_width), markMats.cone);
      cone.scale.set(x.c.vision_reach, 1, x.c.vision_reach); ground(cone, x.p, x.f, 0.02);
    }
    if (x.c.call) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.94, 1, 48).rotateX(-Math.PI / 2), markMats.call);
      ring.scale.setScalar(Math.max(x.c.call[1], 0.5)); ground(ring, x.p, x.f, WATER_LIFT + 0.01);
    }
  }
  if (followed) {
    const ring = new THREE.Mesh(new THREE.RingGeometry(1.0, 1.12, 40).rotateX(-Math.PI / 2), markMats.ring);
    ring.scale.setScalar(Math.max(followed.c.radius * 3, 0.6)); ground(ring, followed.p, followed.f, WATER_LIFT + 0.02);
  }
  if (sensed) for (const x of state.lastPlaced || []) {
    if (!sensed.hears.has(x.c.id)) continue;  // a sound: a blur where it came from, never its size
    const blur = new THREE.Mesh(new THREE.CircleGeometry(0.8, 24).rotateX(-Math.PI / 2), markMats.heard);
    ground(blur, x.p, null, 0.05);
  }
}

// ---------- what each one is doing this step: a small picture above it (always facing the camera, never too small to
// see), a red strike line to whom it attacks, an orange line to whom it chases. One draw per kind of picture.
export function markCanvas(kind) {  // the picture of one kind of mark (the legend draws the same)
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d");
  const disc = (fill) => { g.fillStyle = fill; g.beginPath(); g.arc(32, 32, 26, 0, 7); g.fill(); };
  if (kind === "eat") {  // a green open bite: it is eating
    g.strokeStyle = "#36c46a"; g.lineWidth = 9; g.lineCap = "round";
    g.beginPath(); g.arc(32, 32, 20, 0.75, Math.PI * 2 - 0.75); g.stroke();
  } else if (kind === "miss") {  // a dotted yellow bite: it bit at nothing
    g.strokeStyle = "#f2c230"; g.lineWidth = 6; g.setLineDash([6, 7]); g.beginPath(); g.arc(32, 32, 24, 0, 7); g.stroke();
  } else if (kind === "hit") {  // a red flash: it was hit
    g.fillStyle = "#ff3b30"; g.beginPath();
    for (let k = 0; k < 16; k++) { const r = k % 2 ? 12 : 30, a = k * Math.PI / 8; g.lineTo(32 + r * Math.cos(a), 32 + r * Math.sin(a)); }
    g.fill();
  } else if (kind === "star") {  // a red star on whom it strikes
    g.fillStyle = "#ff5a4a"; g.beginPath();
    for (let k = 0; k < 10; k++) { const r = k % 2 ? 11 : 28, a = -Math.PI / 2 + k * Math.PI / 5; g.lineTo(32 + r * Math.cos(a), 32 + r * Math.sin(a)); }
    g.fill();
  } else if (kind === "flee") {  // a yellow "!": it runs away
    disc("rgba(20,20,20,0.75)"); g.fillStyle = "#ffd23f"; g.font = "bold 46px sans-serif"; g.textAlign = "center";
    g.textBaseline = "middle"; g.fillText("!", 32, 35);
  } else if (kind === "chase") {  // an orange arrow: it chases
    g.fillStyle = "#ff9a2e"; g.beginPath(); g.moveTo(10, 14); g.lineTo(54, 32); g.lineTo(10, 50); g.lineTo(20, 32); g.fill();
  } else if (kind === "baby") {  // a pink heart: it had a baby
    g.fillStyle = "#ff7eb6"; g.beginPath(); g.moveTo(32, 54);
    g.bezierCurveTo(4, 34, 10, 8, 32, 22); g.bezierCurveTo(54, 8, 60, 34, 32, 54); g.fill();
  }
  return c;
}
function markPicture(kind) {
  const t = new THREE.CanvasTexture(markCanvas(kind)); t.colorSpace = THREE.SRGBColorSpace;
  return new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false });
}
const MARK_KINDS = ["eat", "miss", "hit", "star", "flee", "chase", "baby"];
const markPlane = new THREE.PlaneGeometry(1, 1);
const markLines = {
  strike: new THREE.LineBasicMaterial({ color: 0xff3b30 }), chase: new THREE.LineBasicMaterial({ color: 0xff9a2e }),
};
function chased(x, shown, R) {  // whom a chaser goes for: the nearest smaller one in its field of vision (as the world)
  const p = x.p.clone().normalize(), half = Math.cos((x.c.vision_width / 2) * Math.PI / 180), { size } = x.c;
  let best = null, bestAway = Infinity;
  for (const y of shown) {
    if (y === x || y.c.size >= size) continue;
    const q = y.p.clone().normalize(), away = R * p.angleTo(q);
    if (away > x.c.vision_reach || away >= bestAway) continue;
    const t = q.clone().sub(p.clone().multiplyScalar(p.dot(q))), l = t.length();
    if (x.c.vision_width < 360 && l > 0 && t.dot(x.f) / l < half) continue;
    best = y; bestAway = away;
  }
  return best;
}
function drawActions(shown, R) {
  if (!state.markBatches) {
    state.markBatches = new Map(MARK_KINDS.map((kind) => {
      const mesh = new THREE.InstancedMesh(markPlane, markPicture(kind), 256);
      mesh.frustumCulled = false; mesh.renderOrder = 2; state.marks.parent.add(mesh);
      return [kind, mesh];
    }));
    state.markLineSets = Object.fromEntries(Object.entries(markLines).map(([kind, material]) => {
      const line = new THREE.LineSegments(new THREE.BufferGeometry(), material);
      line.frustumCulled = false; state.marks.parent.add(line);
      return [kind, line];
    }));
  }
  const lists = Object.fromEntries(MARK_KINDS.map((kind) => [kind, []])), lines = { strike: [], chase: [] };
  const byId = new Map(shown.map((x) => [x.c.id, x]));
  for (const x of shown) {
    const c = x.c;
    if (c.did === "EAT") lists.eat.push(x);
    else if (c.did === "EAT_MISS") lists.miss.push(x);
    else if (c.did === "HIT_BY_ATTACK") lists.hit.push(x);
    if (c.did === "ATTACK" && c.target !== null && c.target !== undefined && byId.has(c.target)) {
      lines.strike.push([x, byId.get(c.target)]); lists.star.push(byId.get(c.target));
    }
    if (c.intent === "RUN_AWAY") lists.flee.push(x);
    if (c.intent === "CHASE") {
      lists.chase.push(x);
      const prey = chased(x, shown, R);
      if (prey) lines.chase.push([x, prey]);
    }
    if (c.baby) lists.baby.push(x);
  }
  const eye = state.camera.position, o = new THREE.Object3D(), stacked = new Map();
  for (const kind of MARK_KINDS) {
    let mesh = state.markBatches.get(kind);
    const list = lists[kind];
    if (mesh.instanceMatrix.count < list.length) {  // more of a kind than room: a bigger batch
      state.marks.parent.remove(mesh);
      mesh = new THREE.InstancedMesh(markPlane, mesh.material, list.length * 2);
      mesh.frustumCulled = false; mesh.renderOrder = 2; state.marks.parent.add(mesh); state.markBatches.set(kind, mesh);
    }
    list.forEach((x, i) => {
      const n = x.p.clone().normalize(), up = stacked.get(x) || 0;
      const size = Math.max(x.c.radius * 0.8, n.clone().multiplyScalar(R).distanceTo(eye) * 0.015);  // small, a few pixels at least
      o.position.copy(n).multiplyScalar(R + x.c.radius * 2.0 + size * (0.6 + up));  // above its head, never on its body
      o.quaternion.copy(state.camera.quaternion); o.scale.setScalar(size); o.updateMatrix(); mesh.setMatrixAt(i, o.matrix);
      stacked.set(x, up + 1.05);  // two marks on one creature: one above the other
    });
    mesh.count = list.length; mesh.instanceMatrix.needsUpdate = true;
  }
  for (const [kind, pairs] of Object.entries(lines)) {
    const at = new Float32Array(pairs.length * 6);
    pairs.forEach(([a, b], i) => {
      a.p.clone().normalize().multiplyScalar(R + a.c.radius).toArray(at, i * 6);
      b.p.clone().normalize().multiplyScalar(R + b.c.radius).toArray(at, i * 6 + 3);
    });
    state.markLineSets[kind].geometry.setAttribute("position", new THREE.BufferAttribute(at, 3));
  }
}

// ---------- the camera: free to spin and zoom; following keeps one creature in the middle, seen from above or from
// its own eyes
function steer(followed, R) {
  const { camera, controls } = state;
  if (followed && state.view === "eyes") {
    controls.enabled = false;
    const n = followed.p.clone().normalize(), f = followed.f.clone().sub(n.clone().multiplyScalar(followed.f.dot(n))).normalize();
    const r = followed.c.radius;  // its body is drawn about 1.4 radii tall, its head at its front
    const eye = n.clone().multiplyScalar(R + r * 1.3).add(f.clone().multiplyScalar(r));
    camera.position.copy(eye); camera.up.copy(n);
    camera.lookAt(eye.clone().add(f.clone().multiplyScalar(10)).sub(n.clone().multiplyScalar(1.5)));  // ahead, a little down
    const wide = Math.min(120, Math.max(30, followed.c.vision_width)) * Math.PI / 180;  // as wide as its vision
    const fov = 2 * Math.atan(Math.tan(wide / 2) / camera.aspect) * 180 / Math.PI;
    if (Math.abs(camera.fov - fov) > 0.01) { camera.fov = fov; camera.updateProjectionMatrix(); }
    return;
  }
  if (camera.fov !== ABOVE_FOV) { camera.fov = ABOVE_FOV; camera.updateProjectionMatrix(); }
  controls.enabled = true;
  camera.up.set(0, 1, 0);
  if (followed) {  // the ball turns under the camera: the creature stays in the middle, at the same height
    const distance = camera.position.length();
    const want = followed.p.clone().normalize().multiplyScalar(distance);
    camera.position.lerp(want, 0.2);
  }
  controls.update();
}

export function followCreature(id) { if (state) state.follow = id; }
export function zoomBy(factor) {  // the page's zoom buttons: nearer (factor > 1) or farther, within the controls' limits
  if (!state) return;
  const { camera, controls } = state, d = camera.position.distanceTo(controls.target);
  const want = Math.max(controls.minDistance, Math.min(controls.maxDistance, d / factor));
  camera.position.sub(controls.target).setLength(want).add(controls.target); controls.update();
}
export function groundAt(event) {  // the point of the ball under the pointer (length 1, as the world keeps points), or null
  if (!state || !state.R) return null;
  const box = state.renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(mouse, state.camera);
  const hit = ray.ray.intersectSphere(new THREE.Sphere(new THREE.Vector3(), state.R), new THREE.Vector3());
  return hit ? hit.normalize().toArray() : null;
}
export function creatureAt(event) {  // the creature under the pointer, or null
  if (!state) return null;
  const box = state.renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(mouse, state.camera);
  const batched = [...state.batches].filter(([key]) => key !== "rings").map(([, mesh]) => mesh);
  for (const hit of ray.intersectObjects([...state.beasts.children, ...batched], true)) {
    if (hit.instanceId !== undefined && hit.object.userData.ids) return hit.object.userData.ids[hit.instanceId];
    let o = hit.object;
    while (o && o.userData.id === undefined) o = o.parent;
    if (o) return o.userData.id;
  }
  return null;
}
export function view(which) { if (state) state.view = which; }  // a followed creature: "above" or "eyes"
export function render() { if (state) state.renderer.render(state.scene, state.camera); }

// frames drawn back to back, each one finished: how fast this machine draws the planet (no waiting on the page clock)
export function bench(n = 30) {
  if (!state) return null;
  const gl = state.renderer.getContext(), times = [], pixel = new Uint8Array(4);
  for (let i = 0; i < n; i++) {
    const s = performance.now();
    state.controls.update(); state.renderer.render(state.scene, state.camera);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);  // waits until the frame is drawn (gl.finish may not)
    times.push(performance.now() - s);
  }
  times.sort((a, b) => a - b);
  const ms = times[Math.floor(n / 2)];
  return { msPerFrame: +ms.toFixed(1), framesPerSecond: +(1000 / ms).toFixed(1), creatures: (state.lastPlaced || []).length };
}

// What `me` senses (the world's rules, src/senses.py): seen within its vision reach and inside its cone; heard when not
// seen, within its hearing, and it moved last step. Distances along the ground; food is never heard.
export function sensedBy(me, picture) {
  const R = picture.radius, p = v3(me.p).normalize(), f = v3(me.facing);
  const half = Math.cos((me.vision_width / 2) * Math.PI / 180);
  const along = (q) => R * Math.atan2(new THREE.Vector3().crossVectors(p, q).length(), p.dot(q));
  const ahead = (q) => { const t = q.clone().sub(p.clone().multiplyScalar(p.dot(q))); const l = t.length(); return l ? t.dot(f) / l : 1; };
  const seen = (q) => { const d = along(q); return d === 0 || (d <= me.vision_reach && (me.vision_width >= 360 || ahead(q) >= half)); };
  const sees = new Set(), hears = new Map(), foods = new Set();
  for (const c of picture.creatures) {
    if (c.id === me.id) continue;
    const q = v3(c.p);
    if (seen(q)) sees.add(c.id);
    else if (c.moved && along(q) <= me.hearing) hears.set(c.id, along(q));
  }
  picture.foods.forEach((food, i) => { if (seen(v3(food.p))) foods.add(i); });
  return { sees, hears, foods };
}
