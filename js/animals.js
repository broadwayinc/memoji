/*
    animals.js builds the cute 3D animal heads ("memoji") with three.js.

    The heads are sculpted in heads-sdf.js (soft shapes blended like clay, pastel colour blocks) and
    meshed in a Web Worker. Here they get their moving parts: glossy bead eyes that follow your gaze,
    blink, turn into happy arcs and slant when cross or sad; brows that show up with an expression;
    a mouth that smiles, frowns and opens; springy ears; and everything is driven every frame by the
    52 ARKit style blendshapes MediaPipe Face Landmarker returns.

    buildAnimal(id) returns a rig: { object, update(bs, dt, motion, time), dispose() }.
*/
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { HEADS, meshPart, surfacePoint } from "./heads-sdf.js";

export const ANIMALS = [
    { id: "bunny", name: "Bunny", emoji: "🐰" },
    { id: "bear", name: "Bear", emoji: "🐻" },
    { id: "hamster", name: "Hamster", emoji: "🐹" },
    { id: "beaver", name: "Beaver", emoji: "🦫" },
    { id: "lion", name: "Lion", emoji: "🦁" },
    { id: "tiger", name: "Tiger", emoji: "🐯" },
    { id: "koala", name: "Koala", emoji: "🐨" },
    { id: "monkey", name: "Monkey", emoji: "🐵" },
    { id: "gorilla", name: "Gorilla", emoji: "🦍" },
    { id: "alien", name: "Alien", emoji: "👽" },
    { id: "ghost", name: "Ghost", emoji: "👻" },
];

export function animalInfo(id) {
    return ANIMALS.find((a) => a.id === id) || null;
}

/* ------------------------------------------------------------------ math helpers */

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (e0, e1, x) => {
    const t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
};
const v3 = (x, y, z) => new THREE.Vector3(x, y, z);
const dirv = (a) => v3(a[0], a[1], a[2]).normalize();

/* Orient an object so its local +Z points along `forward` and +Y is as close as possible to `up`. */
function orient(obj, forward, up = v3(0, 1, 0)) {
    const z = forward.clone().normalize();
    const x = new THREE.Vector3().crossVectors(up, z);
    if (x.lengthSq() < 1e-6) x.set(1, 0, 0);
    x.normalize();
    const y = new THREE.Vector3().crossVectors(z, x).normalize();
    obj.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
}

/* ------------------------------------------------------------------ materials */

// Soft clay / vinyl toy: matte with a gentle sheen and a soft, wide highlight.
function clayMaterial(opts = {}) {
    return new THREE.MeshPhysicalMaterial({
        color: opts.color ?? 0xffffff,
        vertexColors: !!opts.vertexColors,
        roughness: opts.roughness ?? 0.55,
        metalness: 0,
        clearcoat: opts.clearcoat ?? 0.22,
        clearcoatRoughness: 0.45,
        sheen: opts.sheen ?? 0.35,
        sheenRoughness: 0.8,
        sheenColor: new THREE.Color(0xffffff),
        iridescence: opts.iridescence ?? 0,
        side: opts.side ?? THREE.FrontSide,
    });
}

function glossMaterial(color, opts = {}) {
    return new THREE.MeshPhysicalMaterial({
        color,
        roughness: opts.roughness ?? 0.18,
        clearcoat: 1,
        clearcoatRoughness: 0.05,
        metalness: 0,
        emissive: opts.emissive ?? 0x000000,
        emissiveIntensity: opts.emissiveIntensity ?? 1,
    });
}

/* ------------------------------------------------------------------ meshes from the sculpts */

const geoCache = new Map(); // "id/part" -> BufferGeometry
const pending = new Map(); // "id/part" -> Promise
let worker = null;
let workerFailed = false;
const workerJobs = new Map();

function toGeometry(data) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data.position, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(data.normal, 3));
    g.setAttribute("color", new THREE.BufferAttribute(data.color, 3));
    g.setIndex(new THREE.BufferAttribute(data.index, 1));
    const names = Object.keys(data.morphs || {});
    if (names.length) {
        g.morphAttributes.position = names.map((n) => new THREE.BufferAttribute(data.morphs[n], 3));
        g.morphTargetsRelative = true;
    }
    g.userData.morphNames = names;
    g.computeBoundingSphere();
    return g;
}

function getWorker() {
    if (worker || workerFailed) return worker;
    try {
        worker = new Worker(new URL("./heads-worker.js", import.meta.url), { type: "module" });
        worker.onmessage = (e) => {
            const job = workerJobs.get(e.data.key);
            if (!job) return;
            workerJobs.delete(e.data.key);
            if (e.data.error) job.reject(new Error(e.data.error));
            else job.resolve(e.data.data);
        };
        worker.onerror = () => {
            workerFailed = true;
            for (const job of workerJobs.values()) job.reject(new Error("worker failed"));
            workerJobs.clear();
        };
    } catch (err) {
        workerFailed = true;
        worker = null;
    }
    return worker;
}

// The mesh of a part, built right now if it is not ready yet.
function geometryFor(id, part) {
    const key = id + "/" + part;
    if (!geoCache.has(key)) {
        const data = meshPart(id, part);
        geoCache.set(key, data ? toGeometry(data) : null);
    }
    return geoCache.get(key);
}

/*
    Builds an animal's meshes in the Web Worker (or here, if workers are unavailable), so a later
    buildAnimal() is instant. Resolves when they are ready.
*/
export function ensureAnimal(id) {
    if (!HEADS[id]) return Promise.resolve();
    const parts = ["head", "ear"].filter((p) => HEADS[id][p]);
    return Promise.all(parts.map((part) => {
        const key = id + "/" + part;
        if (geoCache.has(key)) return null;
        if (pending.has(key)) return pending.get(key);
        const w = getWorker();
        const p = (w
            ? new Promise((resolve, reject) => {
                workerJobs.set(key, { resolve, reject });
                w.postMessage({ id, part, key });
            })
            : Promise.resolve(meshPart(id, part)))
            .catch(() => meshPart(id, part))
            .then((data) => {
                if (!geoCache.has(key)) geoCache.set(key, data ? toGeometry(data) : null);
            })
            .finally(() => pending.delete(key));
        pending.set(key, p);
        return p;
    }));
}

export function prewarmAnimals(ids = ANIMALS.map((a) => a.id)) {
    return ids.reduce((chain, id) => chain.then(() => ensureAnimal(id)), Promise.resolve());
}

/* ------------------------------------------------------------------ eyes */

/*
    An eye.
    style "bead": a small glossy black bead, as in a vinyl toy. It slides a little to follow your gaze,
      lids in the head's own colour come down to blink or to look cross / sad (they tilt), and with a big
      smile the bead turns into a happy arc.
    style "void": a big black glossy shape (alien, ghost) that squashes to blink.
*/
function makeEye({ r = 0.09, style = "bead", lidColor = "#ffffff", liner = "#3b2b2b", shape = [1, 1, 1], tilt = 0, side = 1 }) {
    const group = new THREE.Group();
    group.userData.isEye = true;
    const ballGroup = new THREE.Group();
    group.add(ballGroup);
    const parts = { group, ball: ballGroup, style, side, tilt };
    const glintMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, toneMapped: false });

    if (style === "void") {
        const ball = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), glossMaterial(0x14121a, { roughness: 0.1 }));
        ball.scale.set(shape[0], shape[1], shape[2]);
        ballGroup.add(ball);
        ballGroup.rotation.z = tilt;
        parts.voidBall = ball;
        parts.baseScale = shape.slice();
        const g1 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.17, 16, 12), glintMat);
        g1.position.set(r * 0.32 * -side, r * 0.38 * shape[1], r * shape[2] * 0.93);
        g1.scale.z = 0.35;
        ballGroup.add(g1);
        parts.glints = [g1];
    } else {
        const ball = new THREE.Mesh(new THREE.SphereGeometry(r, 40, 28), glossMaterial(0x1d1a22, { roughness: 0.14 }));
        ballGroup.add(ball);
        parts.bead = ball;
        const g1 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.24, 16, 12), glintMat);
        g1.position.set(r * 0.3 * -side, r * 0.38, r * 0.86);
        g1.scale.z = 0.3;
        const g2 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.1, 12, 8), glintMat);
        g2.position.set(r * -0.28 * -side, r * -0.3, r * 0.93);
        g2.scale.z = 0.3;
        ballGroup.add(g1, g2);
        parts.glints = [g1, g2];
        // Happy eyes: a little arc, like ^ ^.
        const arc = new THREE.Mesh(new THREE.TorusGeometry(r * 0.78, r * 0.2, 10, 28, Math.PI), new THREE.MeshStandardMaterial({ color: 0x1d1a22, roughness: 0.4 }));
        arc.position.set(0, -r * 0.35, r * 0.75);
        arc.visible = false;
        group.add(arc);
        parts.arc = arc;
        // Lids in the colour of the head around the eye, with a dark line on the edge.
        const lidMat = clayMaterial({ color: lidColor });
        lidMat.side = THREE.DoubleSide;
        const lidPivot = new THREE.Group();
        group.add(lidPivot);
        const upper = new THREE.Mesh(new THREE.SphereGeometry(r * 1.12, 40, 20, 0, Math.PI * 2, 0, Math.PI / 2), lidMat);
        const lower = new THREE.Mesh(new THREE.SphereGeometry(r * 1.1, 40, 20, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), lidMat);
        const rim = new THREE.Mesh(new THREE.TorusGeometry(r * 1.12, r * 0.09, 8, 48), new THREE.MeshStandardMaterial({ color: liner, roughness: 0.5 }));
        rim.rotation.x = Math.PI / 2;
        upper.add(rim);
        lidPivot.add(upper, lower);
        parts.lidPivot = lidPivot;
        parts.upper = upper;
        parts.lower = lower;
    }

    parts.update = ({ blink, squint, wide, lookX, lookY, angry, sad, happy }) => {
        const tiltZ = side * (angry * 0.45 - sad * 0.42);
        if (parts.voidBall) {
            const s = Math.max(0.07, 1 - blink * 0.95 - squint * 0.2 + wide * 0.15 - happy * 0.3);
            parts.voidBall.scale.y = parts.baseScale[1] * s;
            for (const g of parts.glints) g.visible = s > 0.35;
            ballGroup.rotation.set(-lookY * 0.12, lookX * 0.14, parts.tilt + tiltZ);
            return;
        }
        // Gaze: the bead slides inside its socket.
        ballGroup.position.set(clamp(lookX, -1, 1) * r * 0.32, clamp(lookY, -1, 1) * r * 0.26, 0);
        const size = 1 + wide * 0.15;
        // A big smile closes the eye into an arc.
        const h = clamp(happy * (1 - blink), 0, 1);
        parts.bead.scale.set(size, size * Math.max(0.05, 1 - h), size);
        for (const g of parts.glints) g.visible = h < 0.5;
        parts.arc.visible = h > 0.5;
        parts.arc.scale.setScalar(0.8 + 0.2 * h);
        parts.arc.rotation.z = tiltZ * 0.5;
        // Lids: hidden (rolled back) when relaxed; angry, sad, squint and blink bring them down.
        parts.lidPivot.rotation.z = tiltZ;
        const down = clamp(angry * 0.85 + sad * 0.55 + squint * 0.4, 0, 1);
        const b = clamp(blink, 0, 1);
        parts.upper.rotation.x = lerp(lerp(-1.55, -0.35, down), 0.26, b);
        parts.lower.rotation.x = lerp(1.55 - squint * 0.5, 0.05, b * 0.85);
        parts.upper.visible = parts.upper.rotation.x > -1.45;
        parts.lower.visible = parts.lower.rotation.x < 1.45;
    };
    return parts;
}

/* ------------------------------------------------------------------ expressions */

/*
    Turns MediaPipe's raw blendshapes into what a cartoon face shows: smile, frown, angry, sad,
    surprised, pucker... Raw values are small and mixed (browInnerUp rises both when sad and when
    surprised), so they are combined and amplified here, once, for every rig.
    The engine may add _blinkLeft/_blinkRight (from the eye opening) and _lookXLeft/_lookYLeft...
    (from the iris position), which are more reliable than the blendshapes alone.
*/
export function readExpression(bs) {
    const g = (n) => bs[n] || 0;
    const smile = clamp(((g("mouthSmileLeft") + g("mouthSmileRight")) / 2) * 1.3, 0, 1);
    const frown = clamp(((g("mouthFrownLeft") + g("mouthFrownRight")) / 2) * 3.2 + g("mouthShrugLower") * 0.2 - smile * 0.6, 0, 1);
    const sneer = (g("noseSneerLeft") + g("noseSneerRight")) / 2;
    // A big smile pulls the brows down and scrunches the nose too: that is not anger.
    const browDownL = Math.max(0, g("browDownLeft") - smile * 0.8), browDownR = Math.max(0, g("browDownRight") - smile * 0.8);
    const angry = clamp(((browDownL + browDownR) / 2) * 2.4 + sneer * 0.6 * (1 - smile), 0, 1);
    const outerUp = (g("browOuterUpLeft") + g("browOuterUpRight")) / 2;
    const wide = (g("eyeWideLeft") + g("eyeWideRight")) / 2;
    const surprise = clamp(outerUp * 1.8 + wide * 0.5 - angry * 0.6, 0, 1);
    const sad = clamp((g("browInnerUp") - outerUp * 0.7) * 2.4 + frown * 0.5 - smile * 0.5, 0, 1);
    const blink = (s) => clamp(Math.max(smooth(0.35, 0.7, g("eyeBlink" + s)), g("_blink" + s)), 0, 1);
    const look = (s, side) => {
        const ix = bs["_lookX" + s], iy = bs["_lookY" + s];
        const bx = side > 0 ? g("eyeLookOut" + s) - g("eyeLookIn" + s) : g("eyeLookIn" + s) - g("eyeLookOut" + s);
        const by = g("eyeLookUp" + s) - g("eyeLookDown" + s);
        return [clamp(ix ?? bx * 1.6, -1, 1), clamp(iy ?? by * 1.6, -1, 1)];
    };
    const [lookXL, lookYL] = look("Left", 1);
    const [lookXR, lookYR] = look("Right", -1);
    return {
        smile, frown, angry, sad, surprise, sneer,
        jaw: g("jawOpen"), pucker: g("mouthPucker"), funnel: g("mouthFunnel"),
        stretch: (g("mouthStretchLeft") + g("mouthStretchRight")) / 2,
        puff: g("cheekPuff"), side: g("mouthLeft") - g("mouthRight"),
        blinkL: blink("Left"), blinkR: blink("Right"),
        squintL: g("eyeSquintLeft"), squintR: g("eyeSquintRight"),
        wideL: g("eyeWideLeft"), wideR: g("eyeWideRight"),
        lookXL, lookYL, lookXR, lookYR,
        browUpL: g("browOuterUpLeft"), browUpR: g("browOuterUpRight"), browDownL, browDownR,
    };
}

/* ------------------------------------------------------------------ mouth */

// A tube whose path can be changed every frame without making new geometry.
function makeTube(n, radial, material) {
    const pos = new Float32Array(n * radial * 3);
    const nor = new Float32Array(n * radial * 3);
    const idx = [];
    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < radial; j++) {
            const a = i * radial + j, b = i * radial + ((j + 1) % radial);
            const c = (i + 1) * radial + j, d = (i + 1) * radial + ((j + 1) % radial);
            idx.push(a, c, b, b, c, d);
        }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
    g.setIndex(idx);
    const mesh = new THREE.Mesh(g, material);
    mesh.frustumCulled = false;
    mesh.setPath = (pts, radius) => {
        for (let i = 0; i < n; i++) {
            const p = pts[i], a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
            let tx = b[0] - a[0], ty = b[1] - a[1];
            const l = Math.hypot(tx, ty) || 1;
            const nx = -ty / l, ny = tx / l;
            const rr = radius * (0.35 + 0.65 * Math.sqrt(Math.sin((Math.PI * i) / (n - 1))));
            for (let j = 0; j < radial; j++) {
                const th = (j / radial) * Math.PI * 2;
                const c = Math.cos(th), s = Math.sin(th);
                const dx = nx * c, dy = ny * c, dz = s;
                const k = (i * radial + j) * 3;
                pos[k] = p[0] + dx * rr; pos[k + 1] = p[1] + dy * rr; pos[k + 2] = p[2] + dz * rr;
                nor[k] = dx; nor[k + 1] = dy; nor[k + 2] = dz;
            }
        }
        g.attributes.position.needsUpdate = true;
        g.attributes.normal.needsUpdate = true;
    };
    return mesh;
}

/*
    The mouth is drawn from two curves that change every frame:
    - the lip line, whose corners go up with a smile and down with a frown ("w" style for cats,
      bears and bunnies: two little arcs under the nose);
    - an opening that hangs under it when the jaw drops: a dark, concave inside with a tongue,
      round when puckered or saying "oh".
    Teeth: "buck" / "bigbuck" (bunny, hamster, beaver), "fangs" (lion, tiger), "rows" (gorilla),
    "shark" (rows of pointy teeth).
*/
function makeMouth({ w = 0.12, style = "smile", line = "#3a1d1a", teeth = null, tongue = "#ef7f8f", cavity = "#3c0f16", curv = 1.0 }) {
    const group = new THREE.Group();
    const lineR = Math.max(0.011, w * 0.075);

    const NU = 24, NV = 6;
    const opos = new Float32Array((NU + 1) * (NV + 1) * 3);
    const ocol = new Float32Array((NU + 1) * (NV + 1) * 3);
    const oidx = [];
    const edge = new THREE.Color(cavity), deep = new THREE.Color(cavity).multiplyScalar(0.35);
    for (let j = 0; j <= NV; j++) {
        for (let i = 0; i <= NU; i++) {
            const k = j * (NU + 1) + i;
            const c = edge.clone().lerp(deep, Math.sin((Math.PI * j) / NV));
            ocol.set([c.r, c.g, c.b], k * 3);
            if (i < NU && j < NV) {
                const a = k, b = k + 1, cc = k + NU + 1, d = k + NU + 2;
                oidx.push(a, cc, b, b, cc, d);
            }
        }
    }
    const ogeo = new THREE.BufferGeometry();
    ogeo.setAttribute("position", new THREE.BufferAttribute(opos, 3));
    ogeo.setAttribute("color", new THREE.BufferAttribute(ocol, 3));
    ogeo.setIndex(oidx);
    const opening = new THREE.Mesh(ogeo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, side: THREE.DoubleSide }));
    opening.frustumCulled = false;
    group.add(opening);

    const tng = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshPhysicalMaterial({ color: tongue, roughness: 0.35, clearcoat: 0.4 }));
    group.add(tng);

    const lineMat = new THREE.MeshStandardMaterial({ color: line, roughness: 0.55, side: THREE.DoubleSide });
    const NL = 33;
    const lip = makeTube(NL, 8, lineMat);
    group.add(lip);
    let philtrum = null;
    if (style === "w") {
        philtrum = new THREE.Mesh(new THREE.CapsuleGeometry(lineR * 0.85, w * 0.3, 6, 10), lineMat);
        group.add(philtrum);
    }

    const toothMat = new THREE.MeshPhysicalMaterial({ color: teeth === "bigbuck" ? 0xffeec4 : 0xfffcf4, roughness: 0.22, clearcoat: 0.7 });
    const teethParts = [];
    if (teeth === "buck" || teeth === "bigbuck") {
        const big = teeth === "bigbuck";
        for (const s of [-1, 1]) {
            const t = new THREE.Mesh(new RoundedBoxGeometry(w * (big ? 0.4 : 0.32), w * (big ? 0.62 : 0.44), w * 0.15, 3, w * 0.07), toothMat);
            teethParts.push({ mesh: t, kind: "buck", x: s * w * (big ? 0.21 : 0.17), h: w * (big ? 0.62 : 0.44) });
            group.add(t);
        }
    } else if (teeth === "fangs") {
        const cone = new THREE.ConeGeometry(w * 0.075, w * 0.26, 12);
        for (const s of [-1, 1]) {
            const t = new THREE.Mesh(cone, toothMat);
            t.rotation.z = Math.PI;
            teethParts.push({ mesh: t, kind: "fang", u: s * 0.5 });
            group.add(t);
        }
    } else if (teeth === "rows") {
        const t = new THREE.Mesh(new RoundedBoxGeometry(1, 1, 1, 2, 0.25), toothMat);
        teethParts.push({ mesh: t, kind: "row" });
        group.add(t);
    } else if (teeth === "shark") {
        const cone = new THREE.ConeGeometry(w * 0.07, w * 0.2, 12);
        for (const row of [1, -1]) {
            for (let i = 0; i < 7; i++) {
                const t = new THREE.Mesh(cone, toothMat);
                t.rotation.z = row > 0 ? Math.PI : 0;
                teethParts.push({ mesh: t, kind: "shark", row, u: (i / 6 - 0.5) * 1.6 });
                group.add(t);
            }
        }
    }

    const pts = Array.from({ length: NL }, () => [0, 0, 0]);
    const parts = { group };
    parts.update = (e) => {
        const open = clamp(smooth(0.05, 0.55, e.jaw) + e.funnel * 0.3, 0, 1);
        const sm = clamp(e.smile - e.frown * 1.1, -1, 1);
        const round = clamp(e.pucker * 1.3 + e.funnel * 0.8, 0, 1);
        const W = w * (1 + 0.28 * Math.max(0, sm) + 0.12 * e.stretch - 0.5 * e.pucker);
        const lift = w * 0.6 * sm;
        const wDip = style === "w" ? 0.32 * w * (1 - 0.6 * round) : 0;
        // The mouth sits on the curved muzzle: it bends back to the sides and up / down.
        const zc = (x, y = 0) => -curv * (x * x + 0.7 * y * y);
        const top = (x) => {
            const u = clamp(x / W, -1, 1);
            return -wDip * Math.sin(Math.PI * Math.abs(u)) + lift * u * u + open * 0.05 * w * (1 - u * u);
        };
        // Lip line
        for (let i = 0; i < NL; i++) {
            const x = W * (i / (NL - 1) * 2 - 1);
            pts[i][0] = x;
            pts[i][1] = top(x);
            pts[i][2] = zc(x, pts[i][1]) + lineR * 0.5;
        }
        lip.setPath(pts, lineR * (style === "w" ? 1 : 1 - 0.45 * open));
        lip.visible = !(style === "o" && open > 0.2);
        if (philtrum) {
            philtrum.position.set(0, w * 0.18, lineR * 0.5);
            philtrum.scale.y = 1 - 0.3 * round;
        }

        // Opening
        const Wo = W * (style === "w" ? 0.8 : 0.94) * (1 - 0.3 * round);
        const H = w * (0.12 + 1.35 * open) * (1 + 0.35 * round);
        const k = lerp(0.75, 0.45, round);
        opening.visible = open > 0.03;
        if (opening.visible) {
            for (let j = 0; j <= NV; j++) {
                const v = j / NV;
                for (let i = 0; i <= NU; i++) {
                    const u = (i / NU) * 2 - 1;
                    const x = Wo * u;
                    const t = top(x);
                    const b = t - open * H * Math.pow(Math.max(0, 1 - u * u), k);
                    const q = (j * (NU + 1) + i) * 3;
                    opos[q] = x;
                    opos[q + 1] = lerp(t, b, v);
                    // Drawn on the skin (a carved hole would be hidden by the head itself); the shading gives the depth.
                    opos[q + 2] = zc(x, opos[q + 1]) - 0.001;
                }
            }
            ogeo.attributes.position.needsUpdate = true;
            ogeo.computeBoundingSphere();
        }
        const t0 = top(0);
        const depthH = open * H;
        tng.visible = open > 0.12;
        tng.scale.set(Wo * 0.6, Math.max(0.001, depthH * 0.3), w * 0.06);
        tng.position.set(e.side * Wo * 0.1, t0 - depthH * 0.72, zc(0, t0 - depthH * 0.72) - w * 0.035);

        for (const p of teethParts) {
            const m = p.mesh;
            if (p.kind === "buck") {
                m.position.set(p.x, t0 - p.h * 0.45 - lineR * 0.4, zc(p.x) + w * 0.03);
                m.visible = true;
            } else if (p.kind === "fang") {
                const x = Wo * p.u;
                m.visible = open > 0.08;
                m.scale.setScalar(clamp(open * 3, 0.01, 1));
                m.position.set(x, top(x) - w * 0.11, zc(x) - w * 0.02);
            } else if (p.kind === "row") {
                m.visible = open > 0.1;
                m.scale.set(Wo * 1.3, w * 0.16, w * 0.1);
                m.position.set(0, t0 - w * 0.1, zc(0, t0) - w * 0.04);
            } else if (p.kind === "shark") {
                const x = Wo * p.u * 0.95;
                const u = x / Wo;
                const t = top(x);
                const y = p.row > 0 ? t - w * 0.09 : t - open * H * Math.pow(Math.max(0, 1 - u * u), k) + w * 0.09;
                m.visible = open > 0.1;
                m.scale.setScalar(clamp(open * 2.5, 0.01, 1));
                m.position.set(x, y, zc(x) - w * 0.04);
            }
        }
        group.position.x = e.side * w * 0.25;
    };
    return parts;
}

/* ------------------------------------------------------------------ springs */

class Spring {
    constructor(k = 90, c = 9) {
        this.k = k; this.c = c;
        this.x = 0; this.y = 0; this.vx = 0; this.vy = 0;
    }
    step(dt, fx, fy) {
        dt = Math.min(dt, 1 / 30);
        this.vx += (-this.k * this.x - this.c * this.vx + fx) * dt;
        this.vy += (-this.k * this.y - this.c * this.vy + fy) * dt;
        this.x = clamp(this.x + this.vx * dt, -0.6, 0.6);
        this.y = clamp(this.y + this.vy * dt, -0.6, 0.6);
    }
}


/* ------------------------------------------------------------------ the generic animal */

function heartNose(spec, s) {
    const size = spec.size;
    const geo = new THREE.SphereGeometry(size, 40, 28);
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
        const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
        // A rounded inverted triangle.
        if (spec.shape === "oval") {
            // A big rounded oval, taller than wide (koala).
            p.setXYZ(i, x * 0.9, y * 1.15, z * 0.62);
        } else {
            const k = 1 - 0.38 * smooth(0.2 * size, -size, y);
            p.setXYZ(i, x * k * (spec.wide ?? 1.35), y * 0.8, z * 0.7);
        }
    }
    geo.computeVertexNormals();
    const nose = new THREE.Mesh(geo, glossMaterial(spec.color, { roughness: 0.3 }));
    nose.position.copy(s.p).addScaledVector(s.n, size * 0.2);
    orient(nose, s.n.clone().lerp(v3(0, 0, 1), 0.5));
    if (spec.nostrils) {
        const nm = new THREE.MeshStandardMaterial({ color: new THREE.Color(spec.color).multiplyScalar(0.45), roughness: 0.6 });
        for (const side of [1, -1]) {
            const n = new THREE.Mesh(new THREE.SphereGeometry(size * 0.22, 12, 8), nm);
            n.scale.set(1.4, 0.7, 0.5);
            n.position.set(side * size * 0.45 * (spec.wide ?? 1.35), -size * 0.15, size * 0.55);
            nose.add(n);
        }
    }
    return nose;
}

function buildStandard(id) {
    const spec = HEADS[id];
    const sdf = spec.head.sdf;
    const root = new THREE.Group();
    const surf = (dir) => {
        const s = surfacePoint(sdf, dir);
        return { p: v3(...s.p), n: v3(...s.n) };
    };
    const colorAt = (p) => {
        const c = spec.head.color(p.x, p.y, p.z, 0, 0, 1);
        return new THREE.Color(c[0], c[1], c[2]); // linear, as stored
    };

    const headGeo = geometryFor(id, "head");
    const head = new THREE.Mesh(headGeo, clayMaterial({ vertexColors: true, iridescence: id === "alien" ? 0.35 : 0 }));
    head.updateMorphTargets();
    root.add(head);
    const morphIndex = {};
    (headGeo.userData.morphNames || []).forEach((n, i) => (morphIndex[n] = i));

    // Eyes
    const eyes = [];
    if (spec.eyes) {
        const e = spec.eyes;
        for (const side of [1, -1]) {
            const s = surf([e.dir[0] * side, e.dir[1], e.dir[2]]);
            const lidColor = colorAt(s.p.clone().add(v3(0, e.r * 2, 0)));
            const eye = makeEye({ r: e.r, style: e.style || "bead", lidColor, shape: e.shape, tilt: (e.tilt || 0) * side, side, liner: spec.mouth?.line });
            eye.group.position.copy(s.p).addScaledVector(s.n, -(e.sink ?? 0.5) * e.r);
            orient(eye.group, s.n.clone().multiplyScalar(0.4).add(v3(0, 0, 1)).normalize());
            root.add(eye.group);
            eyes.push(eye);
        }
    }

    // Brows: hidden on a relaxed face, they appear with an expression.
    const brows = [];
    if (spec.brows && spec.eyes) {
        const e = spec.eyes;
        const bm = clayMaterial({ color: spec.brows.color, roughness: 0.6 });
        for (const side of [1, -1]) {
            const s = surf([e.dir[0] * side * 1.05, e.dir[1] + spec.brows.lift, e.dir[2]]);
            const brow = new THREE.Mesh(new THREE.CapsuleGeometry(e.r * 0.32, e.r * 1.5, 8, 16), bm);
            brow.scale.z = 0.5;
            const holder = new THREE.Group();
            holder.position.copy(s.p).addScaledVector(s.n, 0.005);
            orient(holder, s.n);
            holder.add(brow);
            root.add(holder);
            brows.push({ holder, brow, base: holder.position.clone(), side });
        }
    }

    // Nose
    let nose = null;
    if (spec.nose) {
        nose = heartNose(spec.nose, surf(spec.nose.dir));
        root.add(nose);
    } else if (spec.nostrils) {
        const n = spec.nostrils;
        nose = new THREE.Group();
        const m = new THREE.MeshStandardMaterial({ color: n.color, roughness: 0.5 });
        for (const side of [1, -1]) {
            const s = surf([n.dir[0] + n.gap * side, n.dir[1], n.dir[2]]);
            const d = new THREE.Mesh(new THREE.SphereGeometry(n.r, 16, 12), m);
            d.scale.set(1.3, 0.8, 0.4);
            d.position.copy(s.p);
            orient(d, s.n);
            nose.add(d);
        }
        root.add(nose);
    }
    if (nose) nose.userData.base = nose.position.clone();

    // Mouth
    let mouth = null;
    if (spec.mouth) {
        const m = spec.mouth;
        const s = surf(m.dir);
        mouth = makeMouth({ w: m.w, style: m.style, line: m.line, teeth: m.teeth, cavity: m.cavity ?? "#8a3a48", tongue: m.tongue ?? "#f59aa8", curv: m.curv });
        const holder = new THREE.Group();
        holder.position.copy(s.p).addScaledVector(s.n, 0.006);
        orient(holder, s.n.clone().lerp(v3(0, 0, 1), 0.3));
        holder.add(mouth.group);
        root.add(holder);
        mouth.holder = holder;
        mouth.base = holder.position.clone();
    }

    // Whiskers: three short strokes on each side of the muzzle.
    if (spec.whiskers) {
        const wm = new THREE.MeshStandardMaterial({ color: spec.whiskers.color, roughness: 0.5 });
        const len = spec.whiskers.len;
        for (const side of [1, -1]) {
            const s = surf([0.26 * side, -0.33, 0.92]);
            for (let k = 0; k < 3; k++) {
                const a = s.p.clone().addScaledVector(s.n, 0.012).add(v3(0, (1 - k) * 0.045, 0));
                const c = a.clone().add(v3(side * len, (1 - k) * 0.05, -0.05));
                const b = a.clone().lerp(c, 0.5).addScaledVector(s.n, 0.02);
                root.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, b, c), 10, 0.009, 6), wm));
            }
        }
    }

    // Ears
    const ears = [];
    if (spec.ear && spec.ears) {
        const earGeo = geometryFor(id, "ear");
        const E = spec.ears;
        for (const side of [1, -1]) {
            const s = surf([E.dir[0] * side, E.dir[1], E.dir[2]]);
            const base = new THREE.Group();
            base.position.copy(s.p).addScaledVector(s.n, -(E.sink ?? 0.1));
            const up = s.n.clone().lerp(v3(0, 1, 0), E.upright ?? 0.55).normalize();
            const x = new THREE.Vector3().crossVectors(up, v3(0, 0, 1)).normalize();
            const z = new THREE.Vector3().crossVectors(x, up).normalize();
            base.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, up, z));
            base.rotateZ(-(E.splay || 0) * side);
            base.rotateX(-(E.tilt || 0));
            const pivot = new THREE.Group();
            const ear = new THREE.Mesh(earGeo, clayMaterial({ vertexColors: true }));
            if (side < 0) ear.scale.x = -1;
            pivot.add(ear);
            base.add(pivot);
            root.add(base);
            ears.push({ pivot, spring: new Spring(E.k ?? 80, E.c ?? 7), side, floppy: E.floppy ?? 1 });
        }
    }

    const ctx = { root, surf };
    const extraUpdate = spec.antennae ? antennae(ctx) : null;

    function update(bs, dt, motion, time) {
        const e = readExpression(bs);
        const infl = head.morphTargetInfluences;
        if (infl) {
            if (morphIndex.jaw !== undefined) infl[morphIndex.jaw] = e.jaw * 0.9;
            if (morphIndex.puff !== undefined) infl[morphIndex.puff] = clamp(e.puff * 1.5 + e.pucker * 0.15, 0, 1.3);
            if (morphIndex.smile !== undefined) infl[morphIndex.smile] = e.smile;
        }
        const happy = smooth(0.55, 0.85, e.smile) * smooth(0.25, 0.55, Math.max(e.squintL, e.squintR) + e.smile * 0.3);
        for (const eye of eyes) {
            const L = eye.side > 0;
            eye.update({
                blink: L ? e.blinkL : e.blinkR,
                squint: (L ? e.squintL : e.squintR) * 0.35,
                wide: (L ? e.wideL : e.wideR) + e.surprise * 0.5,
                lookX: L ? e.lookXL : e.lookXR,
                lookY: L ? e.lookYL : e.lookYR,
                angry: e.angry, sad: e.sad, happy,
            });
        }
        // Brows grow in with the expression and move with it.
        const show = clamp((e.angry + e.sad + e.surprise) * 2, 0, 1);
        for (const b of brows) {
            const L = b.side > 0;
            const up = e.surprise * 0.9 + e.sad * 0.25 + (L ? e.browUpL : e.browUpR) * 0.3 - e.angry * 0.45;
            b.holder.position.copy(b.base).add(v3(0, clamp(up, -0.6, 1) * 0.08, 0));
            b.brow.visible = show > 0.02;
            b.brow.scale.set(show, show, 0.5 * show);
            b.brow.rotation.set(0, 0, Math.PI / 2 + b.side * (e.angry * 0.55 - e.sad * 0.5));
        }
        if (nose) nose.position.copy(nose.userData.base).add(v3(0, e.sneer * 0.03 + e.smile * 0.008, 0));
        if (mouth) {
            mouth.update(e);
            mouth.holder.position.copy(mouth.base).add(v3(0, -e.jaw * 0.07, 0));
        }
        const acc = motion?.accel || v3();
        const ang = motion?.angVel || v3();
        for (const ear of ears) {
            const fx = (-acc.x * 0.012 - ang.z * 2.5 + ang.y * 1.2 * ear.side) * ear.floppy;
            const fy = (acc.y * 0.012 + ang.x * 2.5 + acc.z * 0.006) * ear.floppy;
            ear.spring.step(dt, fx, fy);
            ear.pivot.rotation.z = ear.spring.x - ear.side * (e.sad * 0.25 - e.surprise * 0.08) * ear.floppy;
            ear.pivot.rotation.x = ear.spring.y + e.sad * 0.2 * ear.floppy;
        }
        if (extraUpdate) extraUpdate({ bs, e, dt, time, motion });
    }

    return {
        object: root,
        update,
        dispose() {
            root.traverse((o) => {
                if (o.isMesh) {
                    const shared = [...geoCache.values()].includes(o.geometry);
                    if (!shared) o.geometry.dispose();
                    (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
                }
            });
        },
    };
}

/* ------------------------------------------------------------------ alien antennae */

function antennae(ctx) {
    const parts = [];
    for (const side of [1, -1]) {
        const s = ctx.surf([0.32 * side, 0.95, 0.05]);
        const base = new THREE.Group();
        base.position.copy(s.p).addScaledVector(s.n, -0.02);
        orient(base, v3(0, 0, 1), s.n.clone().lerp(v3(0, 1, 0), 0.5));
        const pivot = new THREE.Group();
        const curve = new THREE.QuadraticBezierCurve3(v3(0, 0, 0), v3(0.05 * side, 0.25, 0.05), v3(0.12 * side, 0.45, 0));
        const stalk = new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.025, 10), clayMaterial({ color: "#a6e3b6", roughness: 0.45 }));
        const ball = new THREE.Mesh(
            new THREE.SphereGeometry(0.075, 24, 16),
            glossMaterial("#fff4a8", { emissive: "#ffe88a", emissiveIntensity: 0.7 })
        );
        ball.position.copy(curve.getPoint(1));
        pivot.add(stalk, ball);
        base.add(pivot);
        ctx.root.add(base);
        parts.push({ pivot, ball, sp: new Spring(40, 3), side });
    }
    return ({ dt, motion, time }) => {
        const a = motion?.accel || v3();
        const w = motion?.angVel || v3();
        for (const p of parts) {
            p.sp.step(dt, -a.x * 0.015 - w.z * 3, a.y * 0.012 + w.x * 3);
            p.pivot.rotation.z = p.sp.x + Math.sin(time * 2.1 + p.side) * 0.04;
            p.pivot.rotation.x = p.sp.y;
            p.ball.material.emissiveIntensity = 0.7 + 0.3 * Math.sin(time * 3 + p.side);
        }
    };
}

/* ------------------------------------------------------------------ ghost (its own build) */

function buildGhost() {
    const root = new THREE.Group();
    // Profile: a dome on top flowing into a flared sheet with a hem.
    const pts = [];
    const N = 40;
    for (let i = 0; i <= N; i++) {
        const t = i / N;
        if (t < 0.5) {
            const a = (t / 0.5) * (Math.PI / 2);
            pts.push(new THREE.Vector2(Math.sin(a) * 1.0 + 0.0001, Math.cos(a) * 1.2));
        } else {
            const u = (t - 0.5) / 0.5;
            pts.push(new THREE.Vector2(1.0 + 0.16 * u * u, -u * 1.75));
        }
    }
    const geo = new THREE.LatheGeometry(pts, 96);
    const base = geo.attributes.position.array.slice();
    const mat = new THREE.MeshPhysicalMaterial({
        color: "#fbfbff", roughness: 0.55, sheen: 0.6, sheenColor: new THREE.Color("#dfe6ff"), sheenRoughness: 0.5,
        side: THREE.DoubleSide,
    });
    const body = new THREE.Mesh(geo, mat);
    root.add(body);

    const eyes = [];
    for (const side of [1, -1]) {
        const eye = makeEye({ r: 0.15, style: "void", shape: [0.78, 1.15, 0.38], side });
        const d = dirv([0.28 * side, 0.1, 0.95]);
        eye.group.position.copy(d).multiplyScalar(1.0);
        orient(eye.group, d.clone().lerp(v3(0, 0, 1), 0.6));
        root.add(eye.group);
        eyes.push(eye);
    }
    // Little floating brows: the ghost has no fur, but can still look cross or worried.
    const brows = [];
    const bm = new THREE.MeshStandardMaterial({ color: "#8e93a8", roughness: 0.6 });
    for (const side of [1, -1]) {
        const d = dirv([0.28 * side, 0.36, 0.89]);
        const holder = new THREE.Group();
        holder.position.copy(d).multiplyScalar(1.0);
        orient(holder, d);
        const brow = new THREE.Mesh(new THREE.CapsuleGeometry(0.02, 0.1, 6, 12), bm);
        brow.scale.z = 0.5;
        holder.add(brow);
        root.add(holder);
        brows.push({ holder, brow, side, base: holder.position.clone() });
    }
    for (const side of [1, -1]) {
        const b = new THREE.Mesh(new THREE.SphereGeometry(0.12, 24, 16), new THREE.MeshBasicMaterial({ color: "#ffb3c6", transparent: true, opacity: 0.55 }));
        const d = dirv([0.5 * side, -0.15, 0.85]);
        b.position.copy(d).multiplyScalar(0.99);
        b.scale.set(1, 0.6, 0.25);
        orient(b, d);
        root.add(b);
    }
    const mouth = makeMouth({ w: 0.1, style: "o", line: "#1a1820", cavity: "#2a1420", tongue: "#f08aa0", curv: 0.8 });
    // Upright, just in front of the sheet, so an open mouth hanging down stays in front of it too.
    const mh = new THREE.Group();
    mh.position.set(0, -0.24, 1.035);
    mh.add(mouth.group);
    root.add(mh);

    const pos = geo.attributes.position;
    function update(bs, dt, motion, time) {
        const e = readExpression(bs);
        const happy = smooth(0.55, 0.85, e.smile);
        eyes[0].update({ blink: e.blinkL, squint: e.squintL * 0.4 + e.smile * 0.3, wide: e.wideL + e.surprise * 0.4, lookX: e.lookXL, lookY: e.lookYL, angry: e.angry, sad: e.sad, happy });
        eyes[1].update({ blink: e.blinkR, squint: e.squintR * 0.4 + e.smile * 0.3, wide: e.wideR + e.surprise * 0.4, lookX: e.lookXR, lookY: e.lookYR, angry: e.angry, sad: e.sad, happy });
        for (const b of brows) {
            const up = e.surprise * 0.9 + e.sad * 0.3 - e.angry * 0.5;
            b.holder.position.copy(b.base).add(v3(0, up * 0.07, 0));
            b.brow.rotation.set(0, 0, Math.PI / 2 + b.side * (e.angry * 0.6 - e.sad * 0.5));
            b.brow.visible = e.angry + e.sad + e.surprise > 0.15;
        }
        mouth.update(e);
        // Flutter the hem and float a little.
        const acc = motion?.accel || v3();
        for (let i = 0; i < pos.count; i++) {
            const x = base[i * 3], y = base[i * 3 + 1], z = base[i * 3 + 2];
            if (y < -0.2) {
                const w = smooth(-0.2, -1.75, y);
                const phi = Math.atan2(z, x);
                const wave = Math.sin(phi * 7 + time * 3.2) * 0.07 * w + Math.sin(phi * 3 - time * 2) * 0.03 * w;
                pos.setXYZ(i, x * (1 + wave * 0.5) - acc.x * 0.0004 * w, y + wave * 0.6, z * (1 + wave * 0.5));
            }
        }
        pos.needsUpdate = true;
        geo.computeVertexNormals();
        root.position.y = Math.sin(time * 1.6) * 0.04;
        root.rotation.z = Math.sin(time * 1.1) * 0.03;
    }
    return {
        object: root,
        update,
        dispose() {
            root.traverse((o) => {
                if (o.isMesh) {
                    o.geometry.dispose();
                    (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
                }
            });
        },
    };
}

/* ------------------------------------------------------------------ public */

/*
    Where the head sits in MediaPipe's canonical face space (centimetres, origin inside the head).
    The canonical face has its eyes at y ≈ 2.7, chin at y ≈ -9.4 and is ±7.7 wide; a real head with
    hair reaches about 13 to 15 cm above the eyes. The animals have their eyes at the middle of the
    head, so a head of radius 15 centred at eye height covers from about 14 cm above the eyes to
    14 cm below (well past the chin) and the full width with hair. The engine then slides it so its
    eyes are exactly in front of the person's eyes.
*/
export const HEAD_CENTER = [0, 2.7, -4.0];
export const HEAD_SCALE = 15.6;
// Custom models (normalised to a head about 2.2 units wide) use the size real heads have.
export const CUSTOM_CENTER = [0, 1.4, -2.2];
export const CUSTOM_SCALE = 13.0;

/*
    Build a rig, in face space: it can be put straight under the face's pose matrix.
    It is built at once; call ensureAnimal(id) first to have its meshes made in the background.
*/
export function buildAnimal(id) {
    if (!HEADS[id] && id !== "ghost") id = "bear";
    const rig = id === "ghost" ? buildGhost() : buildStandard(id);
    const holder = new THREE.Group();
    holder.position.set(HEAD_CENTER[0], HEAD_CENTER[1], HEAD_CENTER[2]);
    holder.scale.setScalar(id === "ghost" ? HEAD_SCALE * 0.95 : HEAD_SCALE);
    holder.add(rig.object);
    return { ...rig, object: holder, id };
}
