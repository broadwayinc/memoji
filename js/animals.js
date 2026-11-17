/*
    animals.js builds the cute 3D animal heads ("memoji") procedurally with three.js.

    Every head is made in a unit space (head radius about 1) and is driven every frame by the
    52 ARKit style blendshapes that MediaPipe Face Landmarker returns (eyeBlinkLeft, jawOpen, ...).
    Nothing is downloaded: shapes are deformed high resolution spheres, and the fur, stripes,
    muzzles and blush are painted into textures from a function of the surface direction.

    buildAnimal(id) returns a rig: { object, update(bs, dt, motion, time), dispose() }.
*/
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

export const ANIMALS = [
    { id: "bunny", name: "Bunny", emoji: "🐰" },
    { id: "bear", name: "Bear", emoji: "🐻" },
    { id: "hamster", name: "Hamster", emoji: "🐹" },
    { id: "beaver", name: "Beaver", emoji: "🦫" },
    { id: "lion", name: "Lion", emoji: "🦁" },
    { id: "tiger", name: "Tiger", emoji: "🐯" },
    { id: "turtle", name: "Turtle", emoji: "🐢" },
    { id: "shark", name: "Shark", emoji: "🦈" },
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

// "#rrggbb" to [r, g, b] in 0..255 (sRGB, for painting canvases).
function rgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const mixc = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

/*
    Soft ellipsoidal mask on the unit sphere: 1 inside, 0 outside, with a soft edge.
    c is the centre direction (need not be normalized), r the radii per axis.
*/
function blobMask(d, c, r, soft = 0.35) {
    const dx = (d.x - c[0]) / r[0];
    const dy = (d.y - c[1]) / r[1];
    const dz = (d.z - c[2]) / (r[2] ?? r[0]);
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return 1 - smooth(1 - soft, 1 + soft * 0.25, dist);
}

// Cheap deterministic hash noise for felt grain.
function hash2(x, y) {
    const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
    return s - Math.floor(s);
}

/* ------------------------------------------------------------------ shared textures */

let furNormalTex = null;
/*
    A tileable normal map of fine noise. With a small normalScale it reads as felt / short fur,
    and catches the light the way the soft Memoji materials do.
*/
function furNormal() {
    if (furNormalTex) return furNormalTex;
    const N = 256;
    const h = new Float32Array(N * N);
    const rnd = (x, y) => hash2(((x % N) + N) % N, ((y % N) + N) % N);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            // Two octaves of smoothed value noise plus grain.
            let v = 0;
            for (const [f, a] of [[8, 0.5], [32, 0.35], [128, 0.15]]) {
                const fx = (x / N) * f, fy = (y / N) * f;
                const ix = Math.floor(fx), iy = Math.floor(fy);
                const tx = fx - ix, ty = fy - iy;
                const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
                const w = (X, Y) => rnd(((X % f) + f) % f * 7, ((Y % f) + f) % f * 13);
                const n0 = lerp(w(ix, iy), w(ix + 1, iy), sx);
                const n1 = lerp(w(ix, iy + 1), w(ix + 1, iy + 1), sx);
                v += lerp(n0, n1, sy) * a;
            }
            h[y * N + x] = v;
        }
    }
    const c = document.createElement("canvas");
    c.width = c.height = N;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(N, N);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const l = h[y * N + ((x - 1 + N) % N)], r = h[y * N + ((x + 1) % N)];
            const u = h[((y - 1 + N) % N) * N + x], d = h[((y + 1) % N) * N + x];
            const nx = (l - r) * 2.5, ny = (u - d) * 2.5, nz = 1;
            const len = Math.hypot(nx, ny, nz);
            const i = (y * N + x) * 4;
            img.data[i] = ((nx / len) * 0.5 + 0.5) * 255;
            img.data[i + 1] = ((ny / len) * 0.5 + 0.5) * 255;
            img.data[i + 2] = ((nz / len) * 0.5 + 0.5) * 255;
            img.data[i + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
    furNormalTex = new THREE.CanvasTexture(c);
    furNormalTex.wrapS = furNormalTex.wrapT = THREE.RepeatWrapping;
    furNormalTex.repeat.set(6, 3);
    return furNormalTex;
}

/*
    Paint a texture for a sphere-mapped surface (three's SphereGeometry UVs) from fn(dir) -> [r,g,b].
    dir is the unit direction of the texel, so the painting does not depend on later deformation.
*/
function paintSphere(fn, W = 640, H = 320, grain = 0.035) {
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(W, H);
    const d = new THREE.Vector3();
    for (let j = 0; j < H; j++) {
        const theta = ((j + 0.5) / H) * Math.PI;
        const st = Math.sin(theta), ct = Math.cos(theta);
        for (let i = 0; i < W; i++) {
            const phi = ((i + 0.5) / W) * Math.PI * 2;
            d.set(-Math.cos(phi) * st, ct, Math.sin(phi) * st);
            const col = fn(d);
            const g = 1 + (hash2(i, j) - 0.5) * grain * 2;
            const k = (j * W + i) * 4;
            img.data[k] = clamp(col[0] * g, 0, 255);
            img.data[k + 1] = clamp(col[1] * g, 0, 255);
            img.data[k + 2] = clamp(col[2] * g, 0, 255);
            img.data[k + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    return tex;
}

/* ------------------------------------------------------------------ materials */

function furMaterial(opts = {}) {
    const m = new THREE.MeshPhysicalMaterial({
        color: opts.color ?? 0xffffff,
        map: opts.map ?? null,
        roughness: opts.roughness ?? 0.78,
        sheen: opts.sheen ?? 0.7,
        sheenRoughness: 0.45,
        sheenColor: new THREE.Color(opts.sheenColor ?? 0xffffff),
        normalMap: opts.smooth ? null : furNormal(),
        normalScale: new THREE.Vector2(opts.normalScale ?? 0.22, opts.normalScale ?? 0.22),
        clearcoat: opts.clearcoat ?? 0,
        clearcoatRoughness: 0.35,
        iridescence: opts.iridescence ?? 0,
        iridescenceIOR: 1.4,
        side: opts.side ?? THREE.FrontSide,
    });
    return m;
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
        transparent: !!opts.opacity,
        opacity: opts.opacity ?? 1,
    });
}

/* ------------------------------------------------------------------ deformable head */

/*
    A head is a unit sphere whose radius is changed by gaussian bumps (muzzle, cheeks, brow ridge),
    then tapered and scaled per axis. surface(dir) returns a point on it and its normal, so eyes,
    noses and ears sit exactly on the skin.
*/
class HeadShape {
    constructor({ scale = [1, 1, 1], bumps = [], taper = null } = {}) {
        this.scale = scale;
        this.bumps = bumps.map((b) => ({ c: dirv(b.c), a: b.a, s: b.s ?? 0.3, k: b.k ?? [1, 1, 1] }));
        this.taper = taper;
    }

    radius(d, extra) {
        let r = 1;
        const all = extra ? this.bumps.concat(extra) : this.bumps;
        for (const b of all) {
            const dx = (d.x - b.c.x) / b.k[0], dy = (d.y - b.c.y) / b.k[1], dz = (d.z - b.c.z) / b.k[2];
            r += b.a * Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * b.s * b.s));
        }
        return r;
    }

    pos(d, extra, out = new THREE.Vector3()) {
        const r = this.radius(d, extra);
        out.copy(d).multiplyScalar(r);
        if (this.taper) {
            const t = this.taper(d);
            out.x *= t;
            out.z *= t;
        }
        out.x *= this.scale[0];
        out.y *= this.scale[1];
        out.z *= this.scale[2];
        return out;
    }

    // Point and outward normal at a direction (finite differences on the deformed surface).
    surface(dirArr, extra) {
        const d = Array.isArray(dirArr) ? dirv(dirArr) : dirArr.clone().normalize();
        const p = this.pos(d, extra);
        const t1 = new THREE.Vector3().crossVectors(d, Math.abs(d.y) < 0.9 ? v3(0, 1, 0) : v3(1, 0, 0)).normalize();
        const t2 = new THREE.Vector3().crossVectors(d, t1).normalize();
        const e = 0.002;
        const p1 = this.pos(d.clone().addScaledVector(t1, e).normalize(), extra);
        const p2 = this.pos(d.clone().addScaledVector(t2, e).normalize(), extra);
        const n = new THREE.Vector3().crossVectors(p1.sub(p), p2.sub(p)).normalize();
        if (n.dot(d) < 0) n.negate();
        return { p, n, d };
    }

    /*
        Geometry with morph targets. morphs is { name: (d, p) => Vector3 displacement | bumps[] }.
        Normals are computed from the analytic surface so there is no seam on the sphere.
    */
    geometry(morphs = {}, segW = 104, segH = 80) {
        const g = new THREE.SphereGeometry(1, segW, segH);
        const pos = g.attributes.position;
        const nor = g.attributes.normal;
        const dirs = [];
        const d = new THREE.Vector3();
        const basePos = new Float32Array(pos.count * 3);
        const baseNor = new Float32Array(pos.count * 3);

        const evalSurface = (extra, disp, outP, outN) => {
            for (let i = 0; i < pos.count; i++) {
                const dd = dirs[i];
                const s = this.surface(dd, extra);
                if (disp) {
                    // Normal of a displaced surface: displace two neighbours too.
                    const t1 = new THREE.Vector3().crossVectors(dd, Math.abs(dd.y) < 0.9 ? v3(0, 1, 0) : v3(1, 0, 0)).normalize();
                    const t2 = new THREE.Vector3().crossVectors(dd, t1).normalize();
                    const e = 0.004;
                    const q0 = s.p.clone().add(disp(dd, s.p));
                    const d1 = dd.clone().addScaledVector(t1, e).normalize();
                    const d2 = dd.clone().addScaledVector(t2, e).normalize();
                    const p1 = this.pos(d1, extra); p1.add(disp(d1, p1));
                    const p2 = this.pos(d2, extra); p2.add(disp(d2, p2));
                    const n = new THREE.Vector3().crossVectors(p1.sub(q0), p2.sub(q0)).normalize();
                    if (n.dot(dd) < 0) n.negate();
                    outP.set([q0.x, q0.y, q0.z], i * 3);
                    outN.set([n.x, n.y, n.z], i * 3);
                } else {
                    outP.set([s.p.x, s.p.y, s.p.z], i * 3);
                    outN.set([s.n.x, s.n.y, s.n.z], i * 3);
                }
            }
        };

        for (let i = 0; i < pos.count; i++) {
            d.fromBufferAttribute(pos, i).normalize();
            dirs.push(d.clone());
        }
        evalSurface(null, null, basePos, baseNor);
        pos.array.set(basePos);
        nor.array.set(baseNor);

        const names = Object.keys(morphs);
        if (names.length) {
            g.morphAttributes.position = [];
            g.morphAttributes.normal = [];
            g.morphTargetsRelative = true;
            for (const name of names) {
                const m = morphs[name];
                const p = new Float32Array(pos.count * 3);
                const n = new Float32Array(pos.count * 3);
                if (Array.isArray(m)) {
                    const extra = m.map((b) => ({ c: dirv(b.c), a: b.a, s: b.s ?? 0.3, k: b.k ?? [1, 1, 1] }));
                    evalSurface(extra, null, p, n);
                } else {
                    evalSurface(null, m, p, n);
                }
                for (let i = 0; i < p.length; i++) {
                    p[i] -= basePos[i];
                    n[i] -= baseNor[i];
                }
                g.morphAttributes.position.push(new THREE.BufferAttribute(p, 3));
                g.morphAttributes.normal.push(new THREE.BufferAttribute(n, 3));
            }
        }
        g.userData.morphNames = names;
        return g;
    }
}

/* Orient an object so its local +Z points along `forward` and +Y is as close as possible to `up`. */
function orient(obj, forward, up = v3(0, 1, 0)) {
    const z = forward.clone().normalize();
    const x = new THREE.Vector3().crossVectors(up, z);
    if (x.lengthSq() < 1e-6) x.set(1, 0, 0);
    x.normalize();
    const y = new THREE.Vector3().crossVectors(z, x).normalize();
    obj.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
}

/* ------------------------------------------------------------------ eyes */

const eyeTexCache = new Map();
function eyeTexture(style, iris) {
    const key = style + iris;
    if (eyeTexCache.has(key)) return eyeTexCache.get(key);
    const irisC = rgb(iris);
    const dark = mixc(irisC, [8, 6, 10], 0.75);
    const light = mixc(irisC, [255, 255, 255], 0.25);
    const sclera = [250, 248, 246];
    const R = style === "small" ? 0.55 : 0.82; // iris angular radius (rad)
    const P = style === "small" ? 0.3 : 0.5; // pupil
    const tex = paintSphere((d) => {
        const a = Math.acos(clamp(d.z, -1, 1));
        if (a > R + 0.03) return sclera;
        const ang = Math.atan2(d.y, d.x);
        const t = clamp((a - P) / (R - P), 0, 1);
        let c = mixc(dark, light, Math.pow(t, 0.8));
        c = mixc(c, mixc(c, [255, 255, 255], 0.2), (Math.sin(ang * 23) * 0.5 + 0.5) * 0.35 * t);
        // Lower iris glows lighter (Memoji style), upper part darker.
        c = mixc(c, light, smooth(0.0, -0.6, d.y) * 0.35 * t);
        c = mixc(c, [10, 8, 12], smooth(R - 0.08, R, a)); // limbal ring
        c = mixc(c, [6, 5, 8], 1 - smooth(P - 0.03, P + 0.02, a)); // pupil
        return mixc(c, sclera, smooth(R, R + 0.03, a));
    }, 256, 128, 0.01);
    eyeTexCache.set(key, tex);
    return tex;
}

/*
    An eye: glossy eyeball that turns with the gaze, upper and lower lids that blink and squint,
    and catch lights that stay put (they sit on the eye group, not on the ball).
    style "cute" | "small" (with lids), "void" (ghost / alien: a black glossy shape that squashes to blink).
*/
function makeEye({ r = 0.2, style = "cute", iris = "#3a2416", lidMaterial, open = -0.8, shape = [1, 1, 1], tilt = 0, side = 1 }) {
    const group = new THREE.Group();
    const ballGroup = new THREE.Group();
    group.add(ballGroup);
    const parts = { group, ball: ballGroup, style, open, side };

    if (style === "void") {
        const ball = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), glossMaterial(0x0c0b10, { roughness: 0.12 }));
        ball.scale.set(shape[0], shape[1], shape[2]);
        ballGroup.add(ball);
        ballGroup.rotation.z = tilt;
        parts.voidBall = ball;
        parts.baseScale = shape.slice();
    } else {
        const ball = new THREE.Mesh(
            new THREE.SphereGeometry(r, 48, 32),
            new THREE.MeshPhysicalMaterial({ map: eyeTexture(style, iris), roughness: 0.1, clearcoat: 1, clearcoatRoughness: 0.02 })
        );
        ballGroup.add(ball);
        const upper = new THREE.Mesh(new THREE.SphereGeometry(r * 1.07, 48, 24, 0, Math.PI * 2, 0, Math.PI / 2), lidMaterial);
        const lower = new THREE.Mesh(new THREE.SphereGeometry(r * 1.055, 48, 24, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), lidMaterial);
        upper.material.side = THREE.DoubleSide;
        group.add(upper, lower);
        parts.upper = upper;
        parts.lower = lower;
    }

    // Catch lights: a big soft one up and to the side, a small one below.
    const glintMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.92, toneMapped: false });
    const g1 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.17, 16, 12), glintMat);
    const g2 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.08, 12, 8), glintMat);
    const front = style === "void" ? shape[2] : 1;
    g1.position.set(r * 0.32 * -side, r * 0.38 * (style === "void" ? shape[1] : 1), r * front * 0.93);
    g2.position.set(r * -0.25 * -side, r * -0.3 * (style === "void" ? shape[1] : 1), r * front * 0.97);
    g1.scale.z = g2.scale.z = 0.35;
    group.add(g1, g2);
    parts.glints = [g1, g2];

    parts.update = (blink, squint, wide, lookX, lookY) => {
        if (parts.voidBall) {
            const s = Math.max(0.07, 1 - smooth(0.32, 0.8, blink) * 0.95 - squint * 0.15 + wide * 0.12);
            parts.voidBall.scale.y = parts.baseScale[1] * s;
            for (const g of parts.glints) g.visible = s > 0.4;
            ballGroup.rotation.y = lookX * 0.12;
            ballGroup.rotation.x = -lookY * 0.1;
            return;
        }
        ballGroup.rotation.y = lookX * 0.42;
        ballGroup.rotation.x = -lookY * 0.32;
        // Upper lid: rotation.x > 0 rolls it down over the front of the ball.
        const openA = parts.open - wide * 0.3;
        const closedA = 0.22;
        // Smiling eyes report a partial blink; only a real blink closes the lids.
        const b = smooth(0.32, 0.8, blink);
        parts.upper.rotation.x = lerp(openA + squint * 0.22, closedA, b);
        // Lower lid rises with squints and blinks.
        parts.lower.rotation.x = lerp(0.95 - squint * 0.4, 0.05, b * 0.85);
        const glintVis = 1 - smooth(0.35, 0.6, b);
        for (const g of parts.glints) g.scale.setScalar(Math.max(0.001, glintVis)), (g.scale.z = 0.35 * Math.max(0.001, glintVis));
    };
    return parts;
}

/* ------------------------------------------------------------------ mouth */

/*
    The mouth is a dark oval cavity that grows with jawOpen, a tongue inside it,
    and a closed-mouth line ("smile" arc or a cat-like "w") that fades as the mouth opens.
*/
function makeMouth({ w = 0.2, style = "smile", line = "#3a1d1a", teeth = null, tongue = "#ef7f8f", cavity = "#3c0f16", depth = 0.07 }) {
    const group = new THREE.Group();
    const cav = new THREE.Mesh(
        new THREE.SphereGeometry(1, 40, 24),
        new THREE.MeshStandardMaterial({ color: cavity, roughness: 0.6 })
    );
    cav.position.z = -depth * 0.35;
    group.add(cav);

    const tng = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshStandardMaterial({ color: tongue, roughness: 0.45 }));
    group.add(tng);

    const lineMat = new THREE.MeshStandardMaterial({ color: line, roughness: 0.6 });
    const lineGroup = new THREE.Group();
    const tube = Math.max(0.012, w * 0.07);
    if (style === "w") {
        for (const s of [-1, 1]) {
            const arc = new THREE.Mesh(new THREE.TorusGeometry(w * 0.32, tube, 10, 32, Math.PI * 0.95), lineMat);
            arc.rotation.z = Math.PI + Math.PI * 0.025;
            arc.position.x = s * w * 0.31;
            arc.position.y = w * 0.1;
            lineGroup.add(arc);
        }
        // Little vertical line from the nose (philtrum).
        const ph = new THREE.Mesh(new THREE.CapsuleGeometry(tube, w * 0.32, 6, 10), lineMat);
        ph.position.y = w * 0.32;
        lineGroup.add(ph);
    } else {
        const arc = new THREE.Mesh(new THREE.TorusGeometry(w * 0.85, tube, 10, 40, Math.PI * 0.55), lineMat);
        arc.rotation.z = Math.PI + Math.PI * 0.225;
        arc.position.y = w * 0.75;
        lineGroup.add(arc);
    }
    group.add(lineGroup);

    const parts = { group, cav, tng, lineGroup, teethGroups: [] };

    if (teeth === "buck") {
        const toothMat = new THREE.MeshPhysicalMaterial({ color: 0xfffaf0, roughness: 0.25, clearcoat: 0.6 });
        const tg = new THREE.Group();
        for (const s of [-1, 1]) {
            const t = new THREE.Mesh(new RoundedBoxGeometry(w * 0.36, w * 0.5, w * 0.16, 3, w * 0.07), toothMat);
            t.position.set(s * w * 0.19, -w * 0.18, w * 0.04);
            tg.add(t);
        }
        group.add(tg);
        parts.buck = tg;
    } else if (teeth === "bigbuck") {
        const toothMat = new THREE.MeshPhysicalMaterial({ color: 0xffe9b8, roughness: 0.25, clearcoat: 0.6 });
        const tg = new THREE.Group();
        for (const s of [-1, 1]) {
            const t = new THREE.Mesh(new RoundedBoxGeometry(w * 0.42, w * 0.72, w * 0.18, 3, w * 0.08), toothMat);
            t.position.set(s * w * 0.22, -w * 0.3, w * 0.05);
            tg.add(t);
        }
        group.add(tg);
        parts.buck = tg;
    } else if (teeth === "shark") {
        const toothMat = new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: 0.3, clearcoat: 0.5 });
        const cone = new THREE.ConeGeometry(w * 0.075, w * 0.2, 12);
        for (const row of [1, -1]) {
            const tg = new THREE.Group();
            for (let i = 0; i < 7; i++) {
                const t = new THREE.Mesh(cone, toothMat);
                const x = (i / 6 - 0.5) * 1.5;
                t.position.set(x * w, 0, -Math.abs(x) * w * 0.15);
                t.rotation.z = row > 0 ? Math.PI : 0;
                tg.add(t);
            }
            group.add(tg);
            parts.teethGroups.push({ g: tg, row });
        }
    }

    parts.update = (jaw, smile, pucker, funnel, side) => {
        const open = clamp(smooth(0.1, 0.65, jaw) + funnel * 0.2, 0, 1);
        const width = w * (1 + smile * 0.25 - pucker * 0.45 - funnel * 0.2);
        const height = Math.max(0.0005, w * (0.12 + 0.95 * open) * (style === "o" ? 1.2 : 1));
        cav.scale.set(width, height, depth);
        cav.position.y = -height * 0.35;
        cav.visible = open > 0.03;
        tng.visible = cav.visible;
        tng.scale.set(width * 0.62, height * 0.42, depth * 0.85);
        tng.position.set(0, -height * 0.85, -depth * 0.05);
        const lineVis = 1 - smooth(0.15, 0.4, open);
        lineGroup.visible = lineVis > 0.02;
        lineGroup.scale.set(1 + smile * 0.25 - pucker * 0.4, Math.max(0.2, 1 + smile * 0.4 - pucker * 0.3) * Math.max(0.3, lineVis), 1);
        group.position.x = side * w * 0.25;
        for (const t of parts.teethGroups) {
            t.g.visible = open > 0.1;
            t.g.position.y = cav.position.y + (t.row > 0 ? height * 0.8 : -height * 0.8);
            t.g.scale.setScalar(clamp(open * 2.2, 0.01, 1));
            t.g.scale.x = width / w;
        }
        if (parts.buck) parts.buck.position.y = open > 0.05 ? height * 0.25 : 0;
    };
    return parts;
}

/* ------------------------------------------------------------------ ears with springy physics */

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

function roundEar(r, outerMat, innerMat, flat = 0.42) {
    const pivot = new THREE.Group();
    const outer = new THREE.Mesh(new THREE.SphereGeometry(r, 40, 28), outerMat);
    outer.scale.set(1, 1, flat);
    outer.position.y = r * 0.72;
    const inner = new THREE.Mesh(new THREE.SphereGeometry(r * 0.62, 32, 20), innerMat);
    inner.scale.set(1, 1, flat * 0.6);
    inner.position.set(0, r * 0.66, r * flat * 0.62);
    pivot.add(outer, inner);
    return pivot;
}

function longEar(len, width, outerMat, innerMat) {
    const pivot = new THREE.Group();
    const outer = new THREE.Mesh(new THREE.CapsuleGeometry(width, len, 12, 32), outerMat);
    outer.scale.set(1, 1, 0.45);
    outer.position.y = len / 2 + width * 0.6;
    const inner = new THREE.Mesh(new THREE.CapsuleGeometry(width * 0.58, len * 0.85, 10, 24), innerMat);
    inner.scale.set(1, 1, 0.3);
    inner.position.set(0, len / 2 + width * 0.7, width * 0.26);
    pivot.add(outer, inner);
    // A gentle forward curl: the top half of the ear leans forward.
    pivot.userData.curl = true;
    return pivot;
}

/* ------------------------------------------------------------------ the generic animal */

/*
    spec:
      shape        HeadShape options
      paint(d)     colour of the skin at direction d
      skin         material options
      morphs       extra morph targets (jaw / puff / smile are added by default)
      eyes         { dir, r, style, iris, open, shape, tilt }
      brows        { color, w } or null
      nose         function(ctx) -> Object3D, or null
      mouth        { dir, w, style, teeth, depthIn }
      ears         [{ type: "round" | "long", dir, r | len/width, splay, tilt, outer, inner }]
      extras(ctx)  adds custom parts; may return an update(state) function
*/
function buildStandard(spec) {
    const shape = new HeadShape(spec.shape);
    const root = new THREE.Group();
    const cache = spec._cache || (spec._cache = {});

    if (!cache.tex) cache.tex = paintSphere(spec.paint);
    if (!cache.geo) {
        const morphs = {
            jaw: (d) => {
                const w = smooth(-0.02, -0.5, d.y) * smooth(0.1, 0.75, d.z);
                return v3(0, -0.13 * w, 0.02 * w);
            },
            puff: spec.puff || [
                { c: [0.62, -0.28, 0.72], a: 0.1, s: 0.24 },
                { c: [-0.62, -0.28, 0.72], a: 0.1, s: 0.24 },
            ],
            smile: (d) => {
                const w = Math.exp(-((Math.abs(d.x) - 0.48) ** 2 + (d.y + 0.18) ** 2) / (2 * 0.16 * 0.16)) * smooth(0.2, 0.7, d.z);
                return v3(0, 0.04 * w, 0.02 * w);
            },
        };
        cache.geo = shape.geometry(morphs);
    }
    const skinMat = furMaterial({ map: cache.tex, ...(spec.skin || {}) });
    const head = new THREE.Mesh(cache.geo, skinMat);
    head.updateMorphTargets();
    root.add(head);

    const lidMat = furMaterial({ color: spec.lidColor || spec.skinColor || "#ffffff", ...(spec.skin || {}), map: null });
    const ctx = { shape, root, head, skinMat, lidMat, spec };

    // Eyes
    const eyes = [];
    if (spec.eyes) {
        const e = spec.eyes;
        for (const side of [1, -1]) {
            const s = shape.surface([e.dir[0] * side, e.dir[1], e.dir[2]]);
            const eye = makeEye({
                r: e.r, style: e.style, iris: e.iris, lidMaterial: lidMat, open: e.open ?? -0.8,
                shape: e.shape, tilt: (e.tilt || 0) * side, side,
            });
            const fwd = s.n.clone().multiplyScalar(e.outward ?? 0.35).add(v3(0, 0, 1)).normalize();
            eye.group.position.copy(s.p).addScaledVector(s.n, -(e.sink ?? 0.4) * e.r);
            orient(eye.group, fwd);
            root.add(eye.group);
            eyes.push(eye);
        }
    }

    // Brows
    const brows = [];
    if (spec.brows && spec.eyes) {
        const e = spec.eyes;
        const bm = new THREE.MeshStandardMaterial({ color: spec.brows.color, roughness: 0.8 });
        for (const side of [1, -1]) {
            const s = shape.surface([e.dir[0] * side * 1.02, e.dir[1] + (spec.brows.lift ?? 0.3), e.dir[2]]);
            const bw = spec.brows.w ?? 0.12;
            const brow = new THREE.Mesh(new THREE.CapsuleGeometry(bw * 0.24, bw, 6, 12), bm);
            brow.rotation.z = Math.PI / 2;
            brow.scale.z = 0.6;
            const holder = new THREE.Group();
            holder.position.copy(s.p).addScaledVector(s.n, 0.01);
            orient(holder, s.n);
            holder.add(brow);
            root.add(holder);
            brows.push({ holder, brow, base: holder.position.clone(), n: s.n.clone(), side });
        }
    }

    // Nose
    if (spec.nose) {
        const nose = spec.nose(ctx);
        if (nose) root.add(nose);
    }

    // Mouth
    let mouth = null;
    if (spec.mouth) {
        const m = spec.mouth;
        const s = shape.surface(m.dir);
        mouth = makeMouth({ w: m.w, style: m.style, line: m.line, teeth: m.teeth, cavity: m.cavity, tongue: m.tongue });
        const holder = new THREE.Group();
        holder.position.copy(s.p).addScaledVector(s.n, m.out ?? 0.005);
        orient(holder, s.n.clone().lerp(v3(0, 0, 1), 0.3));
        holder.add(mouth.group);
        root.add(holder);
        mouth.holder = holder;
        mouth.base = holder.position.clone();
    }

    // Ears
    const ears = [];
    for (const ear of spec.ears || []) {
        for (const side of [1, -1]) {
            const s = shape.surface([ear.dir[0] * side, ear.dir[1], ear.dir[2]]);
            const base = new THREE.Group();
            base.position.copy(s.p).addScaledVector(s.n, -(ear.sink ?? 0.08));
            const up = s.n.clone().lerp(v3(0, 1, 0), ear.upright ?? 0.55).normalize();
            // Ear faces forward; its +Y follows `up`.
            const fwd = v3(0, 0, 1);
            const x = new THREE.Vector3().crossVectors(up, fwd).normalize();
            const z = new THREE.Vector3().crossVectors(x, up).normalize();
            base.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, up, z));
            base.rotateZ(-(ear.splay || 0) * side);
            base.rotateX(-(ear.tilt || 0));
            const outerMat = ear.outerMat || furMaterial({ color: ear.outer || spec.skinColor || "#ffffff", ...(spec.skin || {}), map: null });
            const innerMat = ear.innerMat || new THREE.MeshPhysicalMaterial({ color: ear.inner || "#f4b6c2", roughness: 0.7, sheen: 0.5 });
            const pivot = ear.type === "long"
                ? longEar(ear.len, ear.width, outerMat, innerMat)
                : roundEar(ear.r, outerMat, innerMat, ear.flat);
            base.add(pivot);
            root.add(base);
            ears.push({ pivot, spring: new Spring(ear.k ?? 80, ear.c ?? 7), side, floppy: ear.floppy ?? 1 });
        }
    }

    const extraUpdate = spec.extras ? spec.extras(ctx) : null;

    const morphIndex = head.morphTargetDictionary || {};
    const state = { t: 0 };

    function update(bs, dt, motion, time) {
        const g = (n) => bs[n] || 0;
        const jaw = g("jawOpen");
        const smile = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2;
        const pucker = g("mouthPucker");
        const funnel = g("mouthFunnel");
        const puff = g("cheekPuff");
        const mouthSide = g("mouthLeft") - g("mouthRight");

        const infl = head.morphTargetInfluences;
        if (infl) {
            if (morphIndex.jaw !== undefined) infl[morphIndex.jaw] = jaw * 0.9;
            if (morphIndex.puff !== undefined) infl[morphIndex.puff] = clamp(puff * 1.4 + pucker * 0.15, 0, 1.3);
            if (morphIndex.smile !== undefined) infl[morphIndex.smile] = smile;
        }

        for (const eye of eyes) {
            const L = eye.side > 0 ? "Left" : "Right";
            const lookIn = g("eyeLookIn" + L), lookOut = g("eyeLookOut" + L);
            const lookX = eye.side > 0 ? lookOut - lookIn : lookIn - lookOut;
            const lookY = g("eyeLookUp" + L) - g("eyeLookDown" + L);
            eye.update(g("eyeBlink" + L), g("eyeSquint" + L) * 0.45 + smile * 0.1, g("eyeWide" + L), lookX, lookY);
        }

        for (const b of brows) {
            const L = b.side > 0 ? "Left" : "Right";
            const up = g("browInnerUp") * 0.6 + g("browOuterUp" + L) * 0.5;
            const down = g("browDown" + L);
            b.holder.position.copy(b.base).add(v3(0, (up - down * 0.6) * 0.07, 0));
            // Inner end up = worried, inner end down = cross.
            b.brow.rotation.set(0, 0, Math.PI / 2 + b.side * (g("browInnerUp") * 0.35 - down * 0.4));
        }

        if (mouth) {
            mouth.update(jaw, smile, pucker, funnel, mouthSide);
            mouth.holder.position.copy(mouth.base).add(v3(0, -jaw * 0.07, jaw * 0.01));
        }

        // Ear springs: the head's acceleration and turning swing the ears.
        const acc = motion?.accel || v3();
        const ang = motion?.angVel || v3();
        for (const e of ears) {
            const fx = (-acc.x * 0.012 - ang.z * 2.5 + ang.y * 1.2 * e.side) * e.floppy;
            const fy = (acc.y * 0.012 + ang.x * 2.5 + acc.z * 0.006) * e.floppy;
            e.spring.step(dt, fx, fy);
            e.pivot.rotation.z = e.spring.x;
            e.pivot.rotation.x = e.spring.y;
        }

        if (extraUpdate) extraUpdate({ bs, g, dt, time, jaw, smile, motion });
    }

    return {
        object: root,
        update,
        dispose() {
            root.traverse((o) => {
                if (o.isMesh) {
                    if (o.geometry !== cache.geo) o.geometry.dispose();
                    // Materials are cheap and per instance; shared textures stay cached.
                    const ms = Array.isArray(o.material) ? o.material : [o.material];
                    ms.forEach((m) => m.dispose());
                }
            });
        },
    };
}

/* ------------------------------------------------------------------ noses */

function heartNose(color, size, dirArr, opts = {}) {
    return (ctx) => {
        const s = ctx.shape.surface(dirArr);
        const geo = new THREE.SphereGeometry(size, 40, 28);
        const p = geo.attributes.position;
        for (let i = 0; i < p.count; i++) {
            const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
            // Narrow the bottom: a rounded inverted triangle.
            const k = 1 - 0.38 * smooth(0.2 * size, -size, y);
            p.setXYZ(i, x * k * (opts.wide ?? 1.35), y * 0.82, z * 0.72);
        }
        geo.computeVertexNormals();
        const nose = new THREE.Mesh(geo, glossMaterial(color, { roughness: opts.rough ?? 0.25 }));
        nose.position.copy(s.p).addScaledVector(s.n, size * 0.25);
        orient(nose, s.n.clone().lerp(v3(0, 0, 1), 0.5));
        return nose;
    };
}

function nostrils(color, dirArr, gap = 0.06, r = 0.022) {
    return (ctx) => {
        const g = new THREE.Group();
        const m = new THREE.MeshStandardMaterial({ color, roughness: 0.5 });
        for (const side of [1, -1]) {
            const s = ctx.shape.surface([dirArr[0] + gap * side, dirArr[1], dirArr[2]]);
            const n = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), m);
            n.scale.set(1.3, 0.8, 0.4);
            n.position.copy(s.p);
            orient(n, s.n);
            g.add(n);
        }
        return g;
    };
}

/* ------------------------------------------------------------------ painting helpers */

const lonOf = (d) => Math.atan2(d.x, d.z);
const latOf = (d) => Math.asin(clamp(d.y, -1, 1));

function blush(c, d, color = [255, 140, 160], amount = 0.5, pos = [0.56, -0.2, 0.8], size = 0.17) {
    const m = Math.max(blobMask(d, pos, [size, size * 0.75, size]), blobMask(d, [-pos[0], pos[1], pos[2]], [size, size * 0.75, size]));
    return mixc(c, color, m * amount);
}

/* ------------------------------------------------------------------ the zoo */

const SPECS = {};

SPECS.bear = () => {
    const fur = rgb("#a96d3b"), muzzle = rgb("#ecd0aa"), dark = rgb("#8a5530");
    return {
        shape: {
            scale: [1.03, 0.98, 0.97],
            bumps: [
                { c: [0, -0.36, 0.93], a: 0.17, s: 0.26, k: [1.3, 1, 1] },
                { c: [0.55, -0.3, 0.75], a: 0.05, s: 0.25 },
                { c: [-0.55, -0.3, 0.75], a: 0.05, s: 0.25 },
            ],
        },
        paint: (d) => {
            let c = mixc(fur, dark, smooth(0.2, 0.95, d.y) * 0.35);
            c = mixc(c, muzzle, blobMask(d, [0, -0.37, 0.93], [0.36, 0.27, 0.4], 0.3));
            return blush(c, d, [240, 130, 120], 0.35);
        },
        skinColor: "#a96d3b",
        eyes: { dir: [0.33, 0.1, 0.94], r: 0.155, iris: "#3a2213", open: -0.77 },
        brows: { color: "#6b3f1e", w: 0.1, lift: 0.27 },
        nose: heartNose("#2b1a14", 0.1, [0, -0.2, 1]),
        mouth: { dir: [0, -0.43, 0.92], w: 0.13, style: "w", line: "#4a2a1c" },
        ears: [{ type: "round", dir: [0.62, 0.74, 0.1], r: 0.24, inner: "#e8bf93", upright: 0.6, splay: 0.1, k: 140, c: 12, floppy: 0.4 }],
    };
};

SPECS.bunny = () => {
    const fur = rgb("#f6f1ec"), shade = rgb("#e7ddd6"), pink = rgb("#f7b3c2");
    return {
        shape: {
            scale: [0.98, 1.0, 0.95],
            bumps: [
                { c: [0.42, -0.36, 0.84], a: 0.1, s: 0.22 },
                { c: [-0.42, -0.36, 0.84], a: 0.1, s: 0.22 },
                { c: [0, -0.3, 0.95], a: 0.06, s: 0.2 },
            ],
        },
        paint: (d) => {
            let c = mixc(fur, shade, smooth(0.0, -0.9, d.z) * 0.6);
            c = mixc(c, [255, 255, 255], blobMask(d, [0, -0.36, 0.92], [0.4, 0.25, 0.4]) * 0.5);
            return blush(c, d, pink, 0.65, [0.58, -0.18, 0.79], 0.18);
        },
        skin: { sheen: 0.9, sheenColor: "#ffffff" },
        skinColor: "#f6f1ec",
        eyes: { dir: [0.34, 0.06, 0.94], r: 0.165, iris: "#2a1a1f", open: -0.72 },
        nose: heartNose("#f48ea6", 0.06, [0, -0.2, 1], { rough: 0.35 }),
        mouth: { dir: [0, -0.34, 0.95], w: 0.11, style: "w", line: "#a35a6a", teeth: "buck" },
        ears: [{ type: "long", dir: [0.3, 0.95, 0.0], len: 0.85, width: 0.17, inner: "#f7b3c2", upright: 0.85, splay: 0.22, tilt: 0.1, k: 45, c: 4, floppy: 1.4 }],
    };
};

SPECS.hamster = () => {
    const fur = rgb("#eaa45a"), white = rgb("#fff7ec"), dark = rgb("#c7803a");
    return {
        shape: {
            scale: [1.08, 0.96, 0.98],
            bumps: [
                { c: [0.62, -0.32, 0.68], a: 0.14, s: 0.27 },
                { c: [-0.62, -0.32, 0.68], a: 0.14, s: 0.27 },
                { c: [0, -0.3, 0.95], a: 0.07, s: 0.2 },
            ],
        },
        puff: [
            { c: [0.7, -0.35, 0.62], a: 0.2, s: 0.27 },
            { c: [-0.7, -0.35, 0.62], a: 0.2, s: 0.27 },
        ],
        paint: (d) => {
            let c = mixc(fur, dark, smooth(0.4, 1, d.y) * 0.3);
            const face = Math.max(
                blobMask(d, [0, -0.38, 0.85], [0.85, 0.42, 0.75], 0.25),
                blobMask(d, [0, 0.3, 0.95], [0.08, 0.5, 0.3], 0.5) // white blaze
            );
            c = mixc(c, white, face);
            return blush(c, d, [255, 150, 160], 0.5, [0.6, -0.22, 0.76], 0.17);
        },
        skin: { sheen: 0.9 },
        skinColor: "#eaa45a",
        eyes: { dir: [0.36, 0.1, 0.93], r: 0.15, iris: "#1d1416", open: -0.72 },
        brows: null,
        nose: heartNose("#f39aa7", 0.05, [0, -0.2, 1], { rough: 0.3 }),
        mouth: { dir: [0, -0.32, 0.95], w: 0.1, style: "w", line: "#9a5a50", teeth: "buck" },
        ears: [{ type: "round", dir: [0.55, 0.76, 0.2], r: 0.15, inner: "#f6a9b4", upright: 0.5, splay: 0.25, k: 120, c: 10, floppy: 0.5 }],
        extras: (ctx) => tuft(ctx, "#d48c43", [0, 1, 0.12]),
    };
};

SPECS.beaver = () => {
    const fur = rgb("#8d5733"), muzzle = rgb("#d6a274"), dark = rgb("#6e4024");
    return {
        shape: {
            scale: [1.0, 1.0, 0.97],
            bumps: [
                { c: [0, -0.33, 0.92], a: 0.15, s: 0.28, k: [1.35, 1, 1] },
                { c: [0.5, -0.32, 0.78], a: 0.06, s: 0.22 },
                { c: [-0.5, -0.32, 0.78], a: 0.06, s: 0.22 },
            ],
        },
        paint: (d) => {
            let c = mixc(fur, dark, smooth(0.3, 1, d.y) * 0.45);
            c = mixc(c, muzzle, blobMask(d, [0, -0.36, 0.92], [0.46, 0.3, 0.42], 0.3));
            return blush(c, d, [235, 120, 110], 0.3);
        },
        skinColor: "#8d5733",
        eyes: { dir: [0.33, 0.12, 0.94], r: 0.14, iris: "#24150e", open: -0.72 },
        brows: { color: "#4f2c16", w: 0.1, lift: 0.27 },
        nose: heartNose("#2a1710", 0.11, [0, -0.17, 1], { wide: 1.6 }),
        mouth: { dir: [0, -0.33, 0.95], w: 0.13, style: "w", line: "#3d2216", teeth: "bigbuck" },
        ears: [{ type: "round", dir: [0.78, 0.45, 0.12], r: 0.13, inner: "#6a3e22", upright: 0.3, splay: 0.4, k: 150, c: 12, floppy: 0.3 }],
        extras: (ctx) => tuft(ctx, "#6e4024", [0, 1, 0.1]),
    };
};

SPECS.lion = () => {
    const fur = rgb("#f0b54b"), muzzle = rgb("#fbe6bd"), chin = rgb("#fff3d9");
    return {
        shape: {
            scale: [1.0, 0.98, 0.96],
            bumps: [
                { c: [0.13, -0.33, 0.93], a: 0.1, s: 0.17 },
                { c: [-0.13, -0.33, 0.93], a: 0.1, s: 0.17 },
                { c: [0, -0.5, 0.86], a: 0.05, s: 0.17 },
            ],
        },
        paint: (d) => {
            let c = mixc(fur, rgb("#e09a35"), smooth(0.4, 1, d.y) * 0.3);
            c = mixc(c, muzzle, Math.max(blobMask(d, [0.13, -0.34, 0.93], [0.24, 0.2, 0.3]), blobMask(d, [-0.13, -0.34, 0.93], [0.24, 0.2, 0.3])));
            c = mixc(c, chin, blobMask(d, [0, -0.55, 0.83], [0.2, 0.15, 0.3]));
            return blush(c, d, [245, 130, 110], 0.3);
        },
        skinColor: "#f0b54b",
        eyes: { dir: [0.32, 0.1, 0.94], r: 0.15, iris: "#5a3510", open: -0.82 },
        brows: { color: "#b8701e", w: 0.1, lift: 0.27 },
        nose: heartNose("#b8655a", 0.085, [0, -0.17, 1]),
        mouth: { dir: [0, -0.39, 0.93], w: 0.12, style: "w", line: "#7a3d22" },
        ears: [{ type: "round", dir: [0.6, 0.74, 0.06], r: 0.17, inner: "#f6c98a", upright: 0.6, splay: 0.15, k: 150, c: 12, floppy: 0.4 }],
        extras: (ctx) => mane(ctx),
    };
};

SPECS.tiger = () => {
    const fur = rgb("#f38a2b"), white = rgb("#fff8ef"), black = rgb("#2a1a14");
    return {
        shape: {
            scale: [1.03, 0.98, 0.96],
            bumps: [
                { c: [0.13, -0.33, 0.93], a: 0.1, s: 0.17 },
                { c: [-0.13, -0.33, 0.93], a: 0.1, s: 0.17 },
                { c: [0.86, -0.3, 0.42], a: 0.07, s: 0.2 },
                { c: [-0.86, -0.3, 0.42], a: 0.07, s: 0.2 },
            ],
        },
        paint: (d) => {
            let c = fur;
            const lon = lonOf(d), lat = latOf(d), alon = Math.abs(lon);
            // White muzzle, cheeks and chin; white spots above the eyes.
            const wm = Math.max(
                blobMask(d, [0, -0.38, 0.9], [0.42, 0.3, 0.45], 0.3),
                blobMask(d, [0.7, -0.3, 0.6], [0.32, 0.28, 0.35], 0.35),
                blobMask(d, [-0.7, -0.3, 0.6], [0.32, 0.28, 0.35], 0.35),
                blobMask(d, [0.32, 0.36, 0.88], [0.1, 0.07, 0.12], 0.4),
                blobMask(d, [-0.32, 0.36, 0.88], [0.1, 0.07, 0.12], 0.4)
            );
            c = mixc(c, white, wm);
            // Forehead stripes: three short bands narrowing to the sides.
            let s = 0;
            for (const lc of [0.6, 0.76, 0.92]) {
                const half = 0.035 * (1 - alon / 0.42);
                if (half > 0) s = Math.max(s, 1 - smooth(half * 0.6, half, Math.abs(lat - lc - 0.05 * Math.cos(lon * 6))));
            }
            // Side stripes: wavy vertical bands on cheeks and back of the head.
            for (const lc of [1.2, 1.5, 1.82, 2.2, 2.6]) {
                const along = clamp(1 - Math.abs(lat - 0.15) / 0.75, 0, 1);
                const half = 0.07 * along;
                const off = alon - lc - 0.1 * Math.sin(lat * 5 + lc);
                if (half > 0) s = Math.max(s, 1 - smooth(half * 0.6, half, Math.abs(off)));
            }
            c = mixc(c, black, s * (1 - wm * 0.85));
            return blush(c, d, [255, 140, 130], 0.25);
        },
        skinColor: "#f38a2b",
        eyes: { dir: [0.33, 0.1, 0.94], r: 0.15, iris: "#3d2a10", open: -0.84 },
        brows: null,
        nose: heartNose("#f29a9a", 0.08, [0, -0.17, 1]),
        mouth: { dir: [0, -0.39, 0.93], w: 0.12, style: "w", line: "#4a2a1c" },
        ears: [{ type: "round", dir: [0.6, 0.76, 0.06], r: 0.2, inner: "#fff4ea", upright: 0.6, splay: 0.15, k: 140, c: 12, floppy: 0.4 }],
    };
};

SPECS.turtle = () => {
    const skin = rgb("#86c46a"), belly = rgb("#e6efae"), dark = rgb("#5f9e4c");
    return {
        shape: {
            scale: [0.96, 1.05, 0.95],
            bumps: [{ c: [0, -0.3, 0.95], a: 0.06, s: 0.3, k: [1.5, 1, 1] }],
            taper: (d) => 1 - 0.08 * smooth(0, -1, d.y),
        },
        paint: (d) => {
            let c = mixc(skin, dark, smooth(0.3, 1, d.y) * 0.4);
            // Little darker spots on the top of the head.
            const sp = Math.max(
                blobMask(d, [0.25, 0.85, 0.45], [0.1, 0.1, 0.1], 0.5),
                blobMask(d, [-0.3, 0.8, 0.5], [0.08, 0.08, 0.08], 0.5),
                blobMask(d, [0, 0.95, 0.2], [0.09, 0.09, 0.09], 0.5)
            );
            c = mixc(c, dark, sp * 0.6);
            c = mixc(c, belly, blobMask(d, [0, -0.6, 0.75], [0.6, 0.38, 0.6], 0.35));
            return blush(c, d, [255, 145, 150], 0.45);
        },
        skin: { roughness: 0.55, sheen: 0.2, clearcoat: 0.25, normalScale: 0.08 },
        skinColor: "#86c46a",
        eyes: { dir: [0.35, 0.12, 0.93], r: 0.16, iris: "#1f1a12", open: -0.64 },
        nose: nostrils("#3d5e2d", [0, -0.1, 1], 0.05, 0.018),
        mouth: { dir: [0, -0.33, 0.95], w: 0.22, style: "smile", line: "#3b5a2a" },
        ears: [],
        extras: (ctx) => shell(ctx),
    };
};

SPECS.shark = () => {
    const top = rgb("#6b90b8"), deep = rgb("#4f7298"), belly = rgb("#f4f6fa");
    return {
        shape: {
            scale: [1.0, 0.96, 1.06],
            bumps: [{ c: [0, 0.05, 1], a: 0.1, s: 0.32 }],
        },
        paint: (d) => {
            let c = mixc(top, deep, smooth(0.2, 1, d.y) * 0.5);
            const edge = -0.08 - 0.08 * Math.sin(lonOf(d) * 3) * smooth(0.3, 1, Math.abs(lonOf(d)));
            const b = smooth(edge + 0.05, edge - 0.05, d.y) * smooth(-0.6, 0.1, d.z);
            c = mixc(c, belly, b);
            // Gills: three dark slits on each side.
            const lon = Math.abs(lonOf(d)), lat = latOf(d);
            let gill = 0;
            for (const gx of [1.35, 1.5, 1.65]) {
                const along = clamp(1 - Math.abs(lat + 0.05) / 0.28, 0, 1);
                gill = Math.max(gill, (1 - smooth(0.012, 0.025, Math.abs(lon - gx - 0.05 * lat))) * smooth(0, 0.3, along));
            }
            c = mixc(c, rgb("#2f4a66"), gill * 0.8);
            return blush(c, d, [255, 150, 170], 0.4 * (1 - b * 0.3), [0.58, -0.15, 0.79], 0.15);
        },
        skin: { roughness: 0.42, sheen: 0.15, clearcoat: 0.35, normalScale: 0.05 },
        skinColor: "#6b90b8",
        eyes: { dir: [0.42, 0.13, 0.9], r: 0.14, iris: "#14161c", open: -0.67, outward: 0.6 },
        nose: nostrils("#3c5878", [0, 0.06, 1], 0.12, 0.016),
        mouth: { dir: [0, -0.3, 0.96], w: 0.26, style: "smile", line: "#2f4a66", teeth: "shark", cavity: "#5a1520" },
        ears: [],
        extras: (ctx) => fin(ctx),
    };
};

SPECS.gorilla = () => {
    const fur = rgb("#34343b"), face = rgb("#7b6e68"), dark = rgb("#25252b");
    return {
        shape: {
            scale: [1.02, 1.0, 0.95],
            bumps: [
                { c: [0, -0.38, 0.9], a: 0.2, s: 0.3, k: [1.25, 1, 1] },
                { c: [0.3, 0.27, 0.92], a: 0.07, s: 0.15, k: [1.4, 1, 1] },
                { c: [-0.3, 0.27, 0.92], a: 0.07, s: 0.15, k: [1.4, 1, 1] },
                { c: [0, 1, 0.1], a: 0.13, s: 0.22, k: [0.6, 1, 1.4] },
            ],
        },
        paint: (d) => {
            let c = mixc(fur, dark, smooth(0.5, 1, d.y) * 0.4);
            const m = Math.max(
                blobMask(d, [0.3, 0.1, 0.94], [0.25, 0.22, 0.3], 0.3),
                blobMask(d, [-0.3, 0.1, 0.94], [0.25, 0.22, 0.3], 0.3),
                blobMask(d, [0, -0.32, 0.92], [0.45, 0.42, 0.45], 0.25)
            );
            c = mixc(c, face, m);
            return blush(c, d, [200, 120, 120], 0.25 * m);
        },
        skin: { sheen: 0.8, sheenColor: "#9a9aa8" },
        lidColor: "#6a5e58",
        skinColor: "#34343b",
        eyes: { dir: [0.3, 0.1, 0.95], r: 0.13, iris: "#4a2c16", style: "cute", open: -0.68 },
        brows: { color: "#1c1c22", w: 0.13, lift: 0.22 },
        nose: (ctx) => {
            const g = new THREE.Group();
            const s = ctx.shape.surface([0, -0.14, 1]);
            const pad = new THREE.Mesh(new THREE.SphereGeometry(0.13, 40, 24), glossMaterial("#2e2828", { roughness: 0.4 }));
            pad.scale.set(1.35, 0.7, 0.55);
            pad.position.copy(s.p).addScaledVector(s.n, 0.02);
            orient(pad, s.n);
            g.add(pad);
            for (const side of [1, -1]) {
                const n = new THREE.Mesh(new THREE.SphereGeometry(0.04, 16, 12), new THREE.MeshStandardMaterial({ color: 0x0a0808 }));
                n.scale.set(1.2, 0.8, 0.5);
                n.position.set(side * 0.065, -0.02, 0.065);
                pad.add(n);
            }
            return g;
        },
        mouth: { dir: [0, -0.48, 0.88], w: 0.2, style: "smile", line: "#2a2020" },
        ears: [{ type: "round", dir: [0.97, 0.05, 0.05], r: 0.12, inner: "#5c524e", outerMat: new THREE.MeshPhysicalMaterial({ color: "#6a5e58", roughness: 0.7 }), upright: 0.1, splay: 0.0, k: 200, c: 15, floppy: 0.2 }],
    };
};

SPECS.alien = () => {
    const skin = rgb("#93e06c"), light = rgb("#c9f5a9"), dark = rgb("#69b84a");
    return {
        shape: {
            scale: [1.06, 1.12, 0.95],
            bumps: [],
            taper: (d) => (d.y < 0 ? 1 - 0.36 * Math.pow(-d.y, 1.4) : 1 + 0.06 * d.y),
        },
        paint: (d) => {
            let c = mixc(skin, dark, smooth(0.3, 1, d.y) * 0.35);
            c = mixc(c, light, blobMask(d, [0, -0.5, 0.85], [0.4, 0.3, 0.4], 0.5) * 0.6);
            return blush(c, d, [255, 160, 190], 0.4, [0.55, -0.3, 0.75], 0.14);
        },
        skin: { roughness: 0.35, sheen: 0.3, clearcoat: 0.5, iridescence: 0.45, normalScale: 0.04 },
        skinColor: "#93e06c",
        eyes: { dir: [0.38, 0.06, 0.92], r: 0.2, style: "void", shape: [1.25, 0.72, 0.42], tilt: 0.42, sink: 0.05, outward: 0.5 },
        nose: nostrils("#3f7a2c", [0, -0.2, 1], 0.035, 0.012),
        mouth: { dir: [0, -0.45, 0.9], w: 0.09, style: "smile", line: "#2f5a22" },
        ears: [],
        extras: (ctx) => antennae(ctx),
    };
};

/* ------------------------------------------------------------------ extras */

function tuft(ctx, color, dirArr) {
    const s = ctx.shape.surface(dirArr);
    const g = new THREE.Group();
    const mat = furMaterial({ color });
    for (const [rz, sc] of [[0.35, 0.8], [0, 1], [-0.45, 0.7]]) {
        const h = new THREE.Mesh(new THREE.SphereGeometry(0.07, 20, 14), mat);
        h.scale.set(0.8 * sc, 2.4 * sc, 0.8 * sc);
        h.position.y = 0.12 * sc;
        const piv = new THREE.Group();
        piv.rotation.z = rz;
        piv.add(h);
        g.add(piv);
    }
    g.position.copy(s.p).addScaledVector(s.n, -0.03);
    orient(g, v3(0, 0, 1), s.n);
    ctx.root.add(g);
    const sp = new Spring(70, 6);
    return ({ dt, motion }) => {
        const a = motion?.accel || v3();
        sp.step(dt, -a.x * 0.01, a.z * 0.004);
        g.rotation.z = sp.x;
    };
}

function mane(ctx) {
    const geos = [];
    const c1 = new THREE.Color("#c06a22"), c2 = new THREE.Color("#d9822d"), c3 = new THREE.Color("#a8561a");
    const add = (x, y, z, r, col, sx = 1, sy = 1, sz = 1) => {
        const g = new THREE.SphereGeometry(r, 24, 16);
        g.scale(sx, sy, sz);
        g.translate(x, y, z);
        const n = g.attributes.position.count;
        const colors = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) colors.set([col.r, col.g, col.b], i * 3);
        g.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geos.push(g);
    };
    const N = 22;
    for (let i = 0; i < N; i++) {
        const a = (i / N) * Math.PI * 2;
        const R = 1.0 + 0.04 * Math.sin(i * 2.3);
        add(Math.cos(a) * R, Math.sin(a) * R * 1.02 - 0.05, -0.12, 0.34, i % 2 ? c1 : c2, 1, 1, 0.8);
        const a2 = a + Math.PI / N;
        add(Math.cos(a2) * 1.2, Math.sin(a2) * 1.18 - 0.05, -0.38, 0.33, c3, 1, 1, 0.8);
    }
    add(0, 0, -0.65, 1.05, c3, 1, 1, 0.55);
    const geo = mergeGeometries(geos);
    const mat = furMaterial({ vertexColors: true, roughness: 0.85, sheen: 0.9, sheenColor: "#ffd08a" });
    mat.vertexColors = true;
    const m = new THREE.Mesh(geo, mat);
    ctx.root.add(m);
    const sp = new Spring(60, 6);
    return ({ dt, motion }) => {
        const a = motion?.accel || v3();
        sp.step(dt, -a.x * 0.002, a.y * 0.002);
        m.position.x = sp.x * 0.1;
        m.position.y = sp.y * 0.1;
    };
}

let shellTex = null;
function shell(ctx) {
    if (!shellTex) {
        // Seeds of the scutes on the top of the shell.
        const seeds = [v3(0, 1, 0)];
        const ring = (n, lat, off) => {
            for (let i = 0; i < n; i++) {
                const a = (i / n) * Math.PI * 2 + off;
                seeds.push(v3(Math.cos(a) * Math.cos(lat), Math.sin(lat), Math.sin(a) * Math.cos(lat)));
            }
        };
        ring(6, 0.95, 0);
        ring(11, 0.4, 0.3);
        const base = rgb("#7a9a3c"), alt = rgb("#8fb04a"), line = rgb("#4c6a24"), rim = rgb("#e2dc84"), belly = rgb("#f1e6a0");
        shellTex = paintSphere((d) => {
            if (d.y < -0.02) return mixc(rim, belly, smooth(-0.05, -0.3, d.y));
            let d1 = 9, d2 = 9, idx = 0;
            seeds.forEach((sd, i) => {
                const dist = 1 - d.dot(sd);
                if (dist < d1) { d2 = d1; d1 = dist; idx = i; } else if (dist < d2) d2 = dist;
            });
            let c = idx % 2 ? base : alt;
            c = mixc(c, mixc(c, [255, 240, 180], 0.3), smooth(0.06, 0, d1));
            c = mixc(c, line, 1 - smooth(0.006, 0.02, d2 - d1));
            return mixc(c, rim, smooth(0.08, 0.0, d.y));
        }, 512, 256, 0.03);
    }
    const geo = new THREE.SphereGeometry(1.35, 64, 40);
    const body = new THREE.Mesh(geo, furMaterial({ map: shellTex, roughness: 0.5, sheen: 0.1, clearcoat: 0.45, normalScale: 0.08 }));
    body.scale.set(1, 0.52, 1.1);
    body.position.set(0, -0.72, -1.25);
    body.rotation.x = 0.28;
    ctx.root.add(body);
    return null;
}

function fin(ctx) {
    const sh = new THREE.Shape();
    sh.moveTo(-0.42, 0);
    sh.bezierCurveTo(-0.3, 0.25, -0.05, 0.55, 0.22, 0.62);
    sh.bezierCurveTo(0.12, 0.4, 0.15, 0.15, 0.32, 0);
    sh.lineTo(-0.42, 0);
    const geo = new THREE.ExtrudeGeometry(sh, { depth: 0.04, bevelEnabled: true, bevelThickness: 0.04, bevelSize: 0.035, bevelSegments: 5, curveSegments: 32 });
    geo.translate(0, 0, -0.02);
    const m = new THREE.Mesh(geo, furMaterial({ color: "#5d82ab", roughness: 0.42, sheen: 0.1, clearcoat: 0.35, normalScale: 0.05 }));
    const s = ctx.shape.surface([0, 0.95, -0.2]);
    const holder = new THREE.Group();
    holder.position.copy(s.p).addScaledVector(s.n, -0.1);
    holder.rotation.y = -Math.PI / 2; // fin runs front to back
    holder.add(m);
    ctx.root.add(holder);
    const sp = new Spring(60, 5);
    return ({ dt, motion }) => {
        const a = motion?.accel || v3();
        const w = motion?.angVel || v3();
        sp.step(dt, -a.x * 0.008 + w.y * 1.5, 0);
        holder.rotation.x = sp.x;
    };
}

function antennae(ctx) {
    const parts = [];
    for (const side of [1, -1]) {
        const s = ctx.shape.surface([0.32 * side, 0.95, 0.05]);
        const base = new THREE.Group();
        base.position.copy(s.p).addScaledVector(s.n, -0.02);
        orient(base, v3(0, 0, 1), s.n.clone().lerp(v3(0, 1, 0), 0.5));
        const pivot = new THREE.Group();
        const curve = new THREE.QuadraticBezierCurve3(v3(0, 0, 0), v3(0.05 * side, 0.25, 0.05), v3(0.12 * side, 0.45, 0));
        const stalk = new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.025, 10), furMaterial({ color: "#7fcf5a", roughness: 0.4, clearcoat: 0.5, normalScale: 0.03 }));
        const ball = new THREE.Mesh(
            new THREE.SphereGeometry(0.075, 24, 16),
            glossMaterial("#fff27a", { emissive: "#ffe14d", emissiveIntensity: 0.9 })
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

let ghostCache = null;
function buildGhost() {
    const root = new THREE.Group();
    // Profile: a dome on top flowing into a flared sheet with a hem.
    const pts = [];
    const N = 40;
    for (let i = 0; i <= N; i++) {
        const t = i / N;
        if (t < 0.5) {
            const a = (t / 0.5) * (Math.PI / 2);
            pts.push(new THREE.Vector2(Math.sin(a) * 1.0 + 0.0001, Math.cos(a) * 1.05));
        } else {
            const u = (t - 0.5) / 0.5;
            pts.push(new THREE.Vector2(1.0 + 0.16 * u * u, -u * 1.75));
        }
    }
    const geo = new THREE.LatheGeometry(pts, 96);
    const base = geo.attributes.position.array.slice();
    const mat = new THREE.MeshPhysicalMaterial({
        color: "#fbfbff", roughness: 0.55, sheen: 0.6, sheenColor: new THREE.Color("#dfe6ff"), sheenRoughness: 0.5,
        side: THREE.DoubleSide, normalMap: furNormal(), normalScale: new THREE.Vector2(0.06, 0.06),
    });
    const body = new THREE.Mesh(geo, mat);
    root.add(body);

    const eyes = [];
    for (const side of [1, -1]) {
        const eye = makeEye({ r: 0.15, style: "void", shape: [0.78, 1.15, 0.38], side });
        const d = dirv([0.28 * side, 0.12, 0.95]);
        eye.group.position.copy(d).multiplyScalar(1.0);
        orient(eye.group, d.clone().lerp(v3(0, 0, 1), 0.6));
        root.add(eye.group);
        eyes.push(eye);
    }
    for (const side of [1, -1]) {
        const b = new THREE.Mesh(new THREE.SphereGeometry(0.12, 24, 16), new THREE.MeshBasicMaterial({ color: "#ffb3c6", transparent: true, opacity: 0.55 }));
        const d = dirv([0.5 * side, -0.15, 0.85]);
        b.position.copy(d).multiplyScalar(0.99);
        b.scale.set(1, 0.6, 0.25);
        orient(b, d);
        root.add(b);
    }
    const mouth = makeMouth({ w: 0.09, style: "o", line: "#1a1820", cavity: "#14121a", tongue: "#f08aa0", depth: 0.05 });
    const md = dirv([0, -0.25, 0.97]);
    const mh = new THREE.Group();
    mh.position.copy(md).multiplyScalar(1.0);
    orient(mh, md);
    mh.add(mouth.group);
    root.add(mh);

    const pos = geo.attributes.position;
    function update(bs, dt, motion, time) {
        const g = (n) => bs[n] || 0;
        eyes[0].update(g("eyeBlinkLeft"), g("eyeSquintLeft"), g("eyeWideLeft"), 0, 0);
        eyes[1].update(g("eyeBlinkRight"), g("eyeSquintRight"), g("eyeWideRight"), 0, 0);
        mouth.update(g("jawOpen"), (g("mouthSmileLeft") + g("mouthSmileRight")) / 2, g("mouthPucker"), g("mouthFunnel"), 0);
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

const specCache = {};

/*
    Build a rig. The returned object is in face space: head centre and size match the
    canonical MediaPipe face (centimetres), so it can be put straight under the face's pose matrix.
*/
export function buildAnimal(id) {
    let rig;
    if (id === "ghost") {
        rig = buildGhost();
    } else {
        const make = SPECS[id] || SPECS.bear;
        if (!specCache[id]) specCache[id] = make();
        rig = buildStandard(specCache[id]);
    }
    const holder = new THREE.Group();
    // Head centre in canonical face space and its size (cm). See README for how this was measured.
    holder.position.set(0, -0.6, -1.2);
    holder.scale.setScalar(id === "ghost" ? 11.6 : 11.3);
    holder.add(rig.object);
    return { ...rig, object: holder, id };
}

