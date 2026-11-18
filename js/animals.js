/*
    animals.js builds the cute 3D animal heads ("memoji") procedurally with three.js.

    Every head is made in a unit space (head radius about 1) and is driven every frame by the
    52 ARKit style blendshapes that MediaPipe Face Landmarker returns (eyeBlinkLeft, jawOpen, ...).
    Nothing is downloaded: shapes are deformed high resolution spheres, and the fur, stripes,
    muzzles and blush are painted into textures from a function of the surface direction.

    Every head has real fur (shell texturing), eyes that follow the gaze and blink, brows, lids and a
    mouth that make expressions (smile, sad, angry, surprised, kiss), and springy ears.

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
function paintSphere(fn, W = 640, H = 320, grain = 0.035, colorSpace = THREE.SRGBColorSpace) {
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
    tex.colorSpace = colorSpace;
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
    constructor({ scale = [1, 1, 1], bumps = [], taper = null, crown = 0.17 } = {}) {
        this.scale = scale;
        this.crown = crown; // how much taller the top of the head is (covers the hair above a person's eyes)
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
        if (out.y > 0) out.y *= 1 + this.crown * out.y;
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

/* ------------------------------------------------------------------ fur (shell texturing) */

let furNoiseTex = null;
let whiteTex = null;

function white() {
    if (!whiteTex) {
        whiteTex = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
        whiteTex.needsUpdate = true;
    }
    return whiteTex;
}

// One random height per strand.
function furNoise() {
    if (furNoiseTex) return furNoiseTex;
    const N = 256;
    const data = new Uint8Array(N * N * 4);
    for (let i = 0; i < N * N; i++) {
        const v = Math.pow(hash2(i % N, Math.floor(i / N) + 17.3), 0.8) * 255;
        data.set([v, v, v, 255], i * 4);
    }
    furNoiseTex = new THREE.DataTexture(data, N, N);
    furNoiseTex.wrapS = furNoiseTex.wrapT = THREE.RepeatWrapping;
    furNoiseTex.magFilter = THREE.NearestFilter;
    furNoiseTex.minFilter = THREE.LinearMipmapLinearFilter;
    furNoiseTex.generateMipmaps = true;
    furNoiseTex.needsUpdate = true;
    return furNoiseTex;
}

/*
    Real fur: the mesh is drawn again in `layers` shells, each pushed out a little along the normal.
    A shell keeps only the strands taller than its height, so strands get thinner towards the tips,
    the silhouette goes fuzzy and the roots are darker. lenMap (grey texture) says how long the fur is
    where: short on the muzzle, none on noses and around the mouth. Shells share the mesh's geometry
    and morph targets, so the fur moves with the face.
*/
function addFur(mesh, { length = 0.04, layers = 10, lenMap = null, repeat = [3, 1.6], gravity = 0.4 } = {}) {
    const base = mesh.material;
    const shells = [];
    for (let i = 1; i <= layers; i++) {
        const h = i / layers;
        const m = new THREE.MeshStandardMaterial({
            map: base.map || white(),
            color: base.color ? base.color.clone() : new THREE.Color(0xffffff),
            vertexColors: !!base.vertexColors,
            roughness: 0.92,
            metalness: 0,
        });
        m.onBeforeCompile = (shader) => {
            shader.uniforms.furOffset = { value: length * h };
            shader.uniforms.furGravity = { value: gravity };
            shader.uniforms.furLevel = { value: h };
            shader.uniforms.furNoise = { value: furNoise() };
            shader.uniforms.furLen = { value: lenMap || white() };
            shader.uniforms.furRepeat = { value: new THREE.Vector2(repeat[0], repeat[1]) };
            shader.vertexShader = "uniform float furOffset;\nuniform float furGravity;\n" + shader.vertexShader.replace(
                "#include <displacementmap_vertex>",
                "#include <displacementmap_vertex>\n\ttransformed += normalize( objectNormal ) * furOffset;\n\ttransformed.y -= furOffset * furOffset * furGravity * 8.0;"
            );
            shader.fragmentShader = "uniform float furLevel;\nuniform sampler2D furNoise;\nuniform sampler2D furLen;\nuniform vec2 furRepeat;\n" + shader.fragmentShader.replace(
                "#include <map_fragment>",
                "#include <map_fragment>\n\tfloat fStrand = texture2D( furNoise, vMapUv * furRepeat ).r;\n\tfloat fLen = texture2D( furLen, vMapUv ).r;\n\tif ( fStrand * fLen < furLevel ) discard;\n\tdiffuseColor.rgb *= mix( 0.6, 1.1, furLevel );"
            );
        };
        m.customProgramCacheKey = () => "memoji-fur-shell";
        const s = new THREE.Mesh(mesh.geometry, m);
        s.morphTargetInfluences = mesh.morphTargetInfluences;
        s.morphTargetDictionary = mesh.morphTargetDictionary;
        s.renderOrder = 1 + i;
        s.userData.furShell = true;
        mesh.add(s);
        shells.push(s);
    }
    return shells;
}

/* ------------------------------------------------------------------ eyes */

const eyeTexCache = new Map();
function eyeTexture(style, iris) {
    const key = style + iris;
    if (eyeTexCache.has(key)) return eyeTexCache.get(key);
    const irisC = rgb(iris);
    const dark = mixc(irisC, [8, 6, 10], 0.7);
    const light = mixc(irisC, [255, 255, 255], 0.3);
    const sclera = [250, 248, 246];
    const R = style === "small" ? 0.55 : 0.8; // iris angular radius (rad)
    const P = style === "small" ? 0.3 : 0.44; // pupil
    const tex = paintSphere((d) => {
        const a = Math.acos(clamp(d.z, -1, 1));
        if (a > R + 0.04) return mixc(sclera, [230, 222, 220], smooth(1.2, 1.6, a));
        const ang = Math.atan2(d.y, d.x);
        const t = clamp((a - P) / (R - P), 0, 1);
        // Radial fibres, a light inner ring and a dark limbal ring.
        const fib = (Math.sin(ang * 31) * 0.5 + 0.5) * 0.6 + (Math.sin(ang * 13 + 1.3) * 0.5 + 0.5) * 0.4;
        let c = mixc(dark, light, Math.pow(t, 0.7) * 0.85);
        c = mixc(c, mixc(c, [255, 250, 240], 0.35), fib * 0.45 * Math.sin(t * Math.PI));
        c = mixc(c, mixc(irisC, [255, 230, 160], 0.35), (1 - smooth(0, 0.25, t)) * 0.55); // warm ring at the pupil
        c = mixc(c, light, smooth(0.0, -0.7, d.y) * 0.4 * t); // lower iris glows
        c = mixc(c, mixc(c, [0, 0, 0], 0.5), smooth(0.2, 0.75, d.y) * 0.5); // lid shadow on top
        c = mixc(c, [12, 9, 14], smooth(R - 0.12, R, a)); // limbal ring
        c = mixc(c, [5, 4, 7], 1 - smooth(P - 0.035, P + 0.015, a)); // pupil
        return mixc(c, sclera, smooth(R, R + 0.04, a));
    }, 384, 192, 0.008);
    eyeTexCache.set(key, tex);
    return tex;
}

/*
    An eye: a glossy ball that turns with the gaze, an upper lid (with a dark lid line, and lashes if
    asked) and a lower lid that blink and squint, and catch lights that stay put.
    The lids sit in a pivot that tilts: inner corner down for angry, outer corner down for sad.
    style "cute" | "small" (with lids), "void" (ghost / alien: a black glossy shape that squashes to blink).
*/
function makeEye({ r = 0.2, style = "cute", iris = "#3a2416", lidMaterial, open = -0.8, shape = [1, 1, 1], tilt = 0, side = 1, lashes = false, liner = "#2b1d18" }) {
    const group = new THREE.Group();
    group.userData.isEye = true;
    const ballGroup = new THREE.Group();
    group.add(ballGroup);
    const parts = { group, ball: ballGroup, style, open, side, tilt };

    if (style === "void") {
        const ball = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), glossMaterial(0x0c0b10, { roughness: 0.1 }));
        ball.scale.set(shape[0], shape[1], shape[2]);
        ballGroup.add(ball);
        ballGroup.rotation.z = tilt;
        parts.voidBall = ball;
        parts.baseScale = shape.slice();
    } else {
        const ball = new THREE.Mesh(
            new THREE.SphereGeometry(r, 48, 32),
            new THREE.MeshPhysicalMaterial({ map: eyeTexture(style, iris), roughness: 0.08, clearcoat: 1, clearcoatRoughness: 0.02 })
        );
        ballGroup.add(ball);
        const lidPivot = new THREE.Group();
        group.add(lidPivot);
        const upper = new THREE.Mesh(new THREE.SphereGeometry(r * 1.07, 48, 24, 0, Math.PI * 2, 0, Math.PI / 2), lidMaterial);
        const lower = new THREE.Mesh(new THREE.SphereGeometry(r * 1.055, 48, 24, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), lidMaterial);
        upper.material.side = THREE.DoubleSide;
        // Lid line: a dark rim along the edge of the upper lid gives the eye its shape.
        const lineMat = new THREE.MeshStandardMaterial({ color: liner, roughness: 0.6 });
        const rim = new THREE.Mesh(new THREE.TorusGeometry(r * 1.07, r * 0.055, 8, 64), lineMat);
        rim.rotation.x = Math.PI / 2;
        upper.add(rim);
        if (lashes) {
            const lashGeo = new THREE.ConeGeometry(r * 0.07, r * 0.42, 8);
            lashGeo.translate(0, r * 0.21, 0);
            for (const phi of [0.25, 0.5, 0.78]) {
                const lash = new THREE.Mesh(lashGeo, lineMat);
                const R = r * 1.07;
                lash.position.set(side * R * Math.cos(phi), 0, R * Math.sin(phi));
                lash.rotation.z = -side * (0.9 + phi * 0.4);
                lash.rotation.x = 0.5;
                upper.add(lash);
            }
        }
        lidPivot.add(upper, lower);
        parts.lidPivot = lidPivot;
        parts.upper = upper;
        parts.lower = lower;
    }

    // Catch lights: a big soft one up and to the side, a small one below.
    const glintMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.92, toneMapped: false });
    const g1 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.17, 16, 12), glintMat);
    const g2 = new THREE.Mesh(new THREE.SphereGeometry(r * 0.08, 12, 8), glintMat);
    const front = style === "void" ? shape[2] : 1;
    const sy = style === "void" ? shape[1] : 1;
    g1.position.set(r * 0.32 * -side, r * 0.38 * sy, r * front * 0.93);
    g2.position.set(r * -0.25 * -side, r * -0.3 * sy, r * front * 0.97);
    g1.scale.z = g2.scale.z = 0.35;
    ballGroup.add(g1, g2);
    parts.glints = [g1, g2];

    // blink, squint, wide 0..1; look -1..1; angry, sad 0..1.
    parts.update = ({ blink, squint, wide, lookX, lookY, angry, sad }) => {
        const tiltZ = side * (angry * 0.42 - sad * 0.38);
        if (parts.voidBall) {
            const s = Math.max(0.07, 1 - blink * 0.95 - squint * 0.2 + wide * 0.15);
            parts.voidBall.scale.y = parts.baseScale[1] * s;
            for (const g of parts.glints) g.visible = s > 0.35;
            ballGroup.rotation.set(-lookY * 0.12, lookX * 0.14, parts.tilt + tiltZ);
            return;
        }
        ballGroup.rotation.y = lookX * 0.55;
        ballGroup.rotation.x = -lookY * 0.4;
        parts.lidPivot.rotation.z = tiltZ;
        // Upper lid: rotation.x > 0 rolls it down over the ball. Angry and sad lower it a little.
        const openA = parts.open - wide * 0.32 + angry * 0.32 + sad * 0.18;
        const b = clamp(blink, 0, 1);
        parts.upper.rotation.x = lerp(openA + squint * 0.25, 0.24, b);
        // Lower lid rises with squints, smiles ("happy eyes") and blinks.
        parts.lower.rotation.x = lerp(0.95 - squint * 0.55, 0.05, b * 0.85);
        const vis = Math.max(0.001, 1 - smooth(0.4, 0.65, b));
        for (const g of parts.glints) {
            g.scale.set(vis, vis, 0.35 * vis);
        }
    };
    return parts;
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
    inner.position.set(0, r * 0.66, r * flat * 0.72);
    pivot.add(outer, inner);
    pivot.userData.outer = outer;
    return pivot;
}

function longEar(len, width, outerMat, innerMat) {
    const pivot = new THREE.Group();
    const outer = new THREE.Mesh(new THREE.CapsuleGeometry(width, len, 12, 32), outerMat);
    outer.scale.set(1, 1, 0.45);
    outer.position.y = len / 2 + width * 0.6;
    const inner = new THREE.Mesh(new THREE.CapsuleGeometry(width * 0.58, len * 0.85, 10, 24), innerMat);
    inner.scale.set(1, 1, 0.3);
    inner.position.set(0, len / 2 + width * 0.7, width * 0.3);
    pivot.add(outer, inner);
    pivot.userData.outer = outer;
    return pivot;
}

/* ------------------------------------------------------------------ the generic animal */

/*
    spec:
      shape        HeadShape options
      paint(d)     colour of the skin at direction d
      skin         material options
      fur          { length, layers, len(d) } or null for smooth skin
      eyes         { dir, r, style, iris, open, shape, tilt, lashes, liner }
      brows        { color, w, lift } or null
      nose         function(ctx) -> Object3D, or null
      mouth        { dir, w, style, line, teeth }
      ears         [{ type: "round" | "long", dir, r | len/width, splay, tilt, inner, fur }]
      whiskers     { color, len } or null
      neck         false to leave it out
      extras(ctx)  adds custom parts; may return an update(state) function
*/
function srgbColor(c) {
    return new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);
}

function buildStandard(spec, opts = {}) {
    const shape = new HeadShape(spec.shape);
    const root = new THREE.Group();
    const cache = spec._cache || (spec._cache = {});

    if (!cache.tex) cache.tex = paintSphere(spec.paint);
    if (!cache.geo) {
        const morphs = {
            jaw: (d) => {
                const w = smooth(-0.05, -0.55, d.y) * smooth(0.1, 0.75, d.z);
                return v3(0, -0.11 * w, 0);
            },
            puff: spec.puff || [
                { c: [0.6, -0.3, 0.72], a: 0.1, s: 0.24 },
                { c: [-0.6, -0.3, 0.72], a: 0.1, s: 0.24 },
            ],
            smile: (d) => {
                const w = Math.exp(-((Math.abs(d.x) - 0.46) ** 2 + (d.y + 0.2) ** 2) / (2 * 0.16 * 0.16)) * smooth(0.2, 0.7, d.z);
                return v3(0, 0.045 * w, 0.025 * w);
            },
        };
        cache.geo = shape.geometry(morphs);
    }
    // Fur length: shorter around the eyes, none on the nose and mouth.
    if (spec.fur && !cache.furLen) {
        const e = spec.eyes, m = spec.mouth;
        cache.furLen = paintSphere((d) => {
            let l = spec.fur.len ? spec.fur.len(d) : 1;
            if (e) {
                const em = Math.max(blobMask(d, [e.dir[0], e.dir[1], e.dir[2]], [e.r * 1.1, e.r * 1.1, e.r * 1.1], 0.5),
                    blobMask(d, [-e.dir[0], e.dir[1], e.dir[2]], [e.r * 1.1, e.r * 1.1, e.r * 1.1], 0.5));
                l *= 1 - em * 0.8;
            }
            if (m) l *= 1 - blobMask(d, m.dir, [m.w * 1.6, m.w * 1.1, m.w * 1.6], 0.5) * 0.9;
            l *= 1 - blobMask(d, [0, -0.2, 1], [0.12, 0.09, 0.12], 0.5) * 0.95; // nose
            const v = clamp(l, 0, 1) * 255;
            return [v, v, v];
        }, 256, 128, 0, THREE.NoColorSpace);
    }
    const skinMat = furMaterial({ map: cache.tex, ...(spec.skin || {}) });
    const head = new THREE.Mesh(cache.geo, skinMat);
    head.updateMorphTargets();
    root.add(head);
    const furLayers = opts.furLayers ?? spec.fur?.layers ?? 10;
    if (spec.fur && furLayers > 0) {
        addFur(head, { length: spec.fur.length, layers: furLayers, lenMap: cache.furLen, gravity: spec.fur.gravity ?? 0.4 });
    }

    const lidMat = furMaterial({ color: spec.lidColor || spec.skinColor || "#ffffff", ...(spec.skin || {}), map: null });
    const ctx = { shape, root, head, skinMat, lidMat, spec, furLayers };

    // Eyes
    const eyes = [];
    if (spec.eyes) {
        const e = spec.eyes;
        for (const side of [1, -1]) {
            const s = shape.surface([e.dir[0] * side, e.dir[1], e.dir[2]]);
            const eye = makeEye({
                r: e.r, style: e.style, iris: e.iris, lidMaterial: lidMat, open: e.open ?? -0.8,
                shape: e.shape, tilt: (e.tilt || 0) * side, side, lashes: e.lashes, liner: e.liner,
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
        const bm = furMaterial({ color: spec.brows.color, roughness: 0.85, sheen: 0.5, normalScale: 0.3 });
        for (const side of [1, -1]) {
            const s = shape.surface([e.dir[0] * side * 1.04, e.dir[1] + (spec.brows.lift ?? e.r * 1.7), e.dir[2]]);
            const bw = spec.brows.w ?? 0.13;
            const brow = new THREE.Mesh(new THREE.CapsuleGeometry(bw * 0.24, bw, 8, 16), bm);
            brow.scale.z = 0.55;
            const holder = new THREE.Group();
            holder.position.copy(s.p).addScaledVector(s.n, (spec.fur?.length ?? 0) * 0.6 + 0.01);
            orient(holder, s.n);
            holder.add(brow);
            root.add(holder);
            brows.push({ holder, brow, base: holder.position.clone(), side });
        }
    }

    // Nose
    let nose = null;
    if (spec.nose) {
        nose = spec.nose(ctx);
        if (nose) {
            root.add(nose);
            nose.userData.base = nose.position.clone();
        }
    }

    // Mouth
    let mouth = null;
    if (spec.mouth) {
        const m = spec.mouth;
        const s = shape.surface(m.dir);
        mouth = makeMouth({ w: m.w, style: m.style, line: m.line, teeth: m.teeth, cavity: m.cavity, tongue: m.tongue, curv: m.curv });
        const holder = new THREE.Group();
        holder.position.copy(s.p).addScaledVector(s.n, m.out ?? 0.008);
        orient(holder, s.n.clone().lerp(v3(0, 0, 1), 0.3));
        holder.add(mouth.group);
        root.add(holder);
        mouth.holder = holder;
        mouth.base = holder.position.clone();
    }

    // Whiskers
    if (spec.whiskers) {
        const wm = new THREE.MeshStandardMaterial({ color: spec.whiskers.color, roughness: 0.5 });
        const len = spec.whiskers.len ?? 0.4;
        for (const side of [1, -1]) {
            const s = shape.surface([0.2 * side, -0.32, 0.94]);
            for (let k = 0; k < 3; k++) {
                const a = s.p.clone().addScaledVector(s.n, 0.02).add(v3(0, (1 - k) * 0.035, 0));
                const b = a.clone().add(v3(side * len * 0.55, (1 - k) * 0.05 + 0.03, 0.02));
                const c = a.clone().add(v3(side * len, (1 - k) * 0.11 - 0.02, -0.1));
                const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.QuadraticBezierCurve3(a, b, c), 14, 0.0055, 5), wm);
                root.add(tube);
            }
        }
    }

    // Neck: covers the real neck under the head.
    if (spec.neck !== false) {
        const nc = srgbColor(spec.paint(dirv([0, -0.75, -0.4])));
        const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.34, 0.5, 48, 1, true), furMaterial({ color: nc, side: THREE.DoubleSide }));
        neck.position.set(0, -0.86, -0.2);
        neck.userData.neck = true;
        if (spec.fur && furLayers > 0) addFur(neck, { length: spec.fur.length * 0.7, layers: Math.max(3, Math.round(furLayers * 0.6)), repeat: [6, 2] });
        root.add(neck);
    }

    // Ears
    const ears = [];
    for (const ear of spec.ears || []) {
        for (const side of [1, -1]) {
            const s = shape.surface([ear.dir[0] * side, ear.dir[1], ear.dir[2]]);
            const base = new THREE.Group();
            base.position.copy(s.p).addScaledVector(s.n, -(ear.sink ?? 0.08));
            const up = s.n.clone().lerp(v3(0, 1, 0), ear.upright ?? 0.55).normalize();
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
            if (spec.fur && furLayers > 0 && !ear.noFur) addFur(pivot.userData.outer, { length: ear.fur ?? spec.fur.length * 0.5, layers: Math.max(3, Math.round(furLayers * 0.6)), repeat: [2, 2] });
            base.add(pivot);
            root.add(base);
            ears.push({ pivot, spring: new Spring(ear.k ?? 80, ear.c ?? 7), side, floppy: ear.floppy ?? 1 });
        }
    }

    const extraUpdate = spec.extras ? spec.extras(ctx) : null;
    const morphIndex = head.morphTargetDictionary || {};

    function update(bs, dt, motion, time) {
        const e = readExpression(bs);
        const infl = head.morphTargetInfluences;
        if (infl) {
            if (morphIndex.jaw !== undefined) infl[morphIndex.jaw] = e.jaw * 0.9;
            if (morphIndex.puff !== undefined) infl[morphIndex.puff] = clamp(e.puff * 1.5 + e.pucker * 0.15, 0, 1.3);
            if (morphIndex.smile !== undefined) infl[morphIndex.smile] = e.smile;
        }

        for (const eye of eyes) {
            const L = eye.side > 0;
            eye.update({
                blink: L ? e.blinkL : e.blinkR,
                squint: (L ? e.squintL : e.squintR) * 0.4 + e.smile * 0.35,
                wide: (L ? e.wideL : e.wideR) + e.surprise * 0.4,
                lookX: L ? e.lookXL : e.lookXR,
                lookY: L ? e.lookYL : e.lookYR,
                angry: e.angry, sad: e.sad,
            });
        }

        for (const b of brows) {
            const L = b.side > 0;
            const up = e.surprise * 0.9 + e.sad * 0.3 + (L ? e.browUpL : e.browUpR) * 0.4 - e.angry * 0.55 - (L ? e.browDownL : e.browDownR) * 0.3;
            b.holder.position.copy(b.base).add(v3(0, clamp(up, -0.7, 1) * 0.085, 0));
            // Positive turns the outer end up: angry. Negative lifts the inner end: sad.
            b.brow.rotation.set(0, 0, Math.PI / 2 + b.side * (e.angry * 0.55 - e.sad * 0.5));
        }

        if (nose) nose.position.copy(nose.userData.base).add(v3(0, e.sneer * 0.03 + e.smile * 0.008, 0));

        if (mouth) {
            mouth.update(e);
            mouth.holder.position.copy(mouth.base).add(v3(0, -e.jaw * 0.06, e.jaw * 0.01));
        }

        // Ear springs: the head's acceleration and turning swing the ears.
        const acc = motion?.accel || v3();
        const ang = motion?.angVel || v3();
        for (const ear of ears) {
            const fx = (-acc.x * 0.012 - ang.z * 2.5 + ang.y * 1.2 * ear.side) * ear.floppy;
            const fy = (acc.y * 0.012 + ang.x * 2.5 + acc.z * 0.006) * ear.floppy;
            ear.spring.step(dt, fx, fy);
            // Ears droop a little when sad and perk up when surprised.
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
                    if (o.geometry !== cache.geo && !o.userData.furShell) o.geometry.dispose();
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
        nose.position.copy(s.p).addScaledVector(s.n, size * 0.25 + (ctx.spec.fur?.length ?? 0) * 0.3);
        orient(nose, s.n.clone().lerp(v3(0, 0, 1), 0.5));
        // Nostrils: two small dark dimples on the underside.
        const nm = new THREE.MeshStandardMaterial({ color: new THREE.Color(color).multiplyScalar(0.35), roughness: 0.6 });
        for (const side of [1, -1]) {
            const n = new THREE.Mesh(new THREE.SphereGeometry(size * 0.2, 12, 8), nm);
            n.scale.set(1.3, 0.7, 0.5);
            n.position.set(side * size * 0.42 * (opts.wide ?? 1.35) * 0.75, -size * 0.32, size * 0.5);
            nose.add(n);
        }
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

// Whisker dots on the two muzzle lobes.
function whiskerDots(c, d, color = [90, 60, 45]) {
    let m = 0;
    for (const s of [1, -1]) {
        for (const [x, y] of [[0.13, -0.3], [0.19, -0.34], [0.15, -0.38], [0.22, -0.3]]) {
            m = Math.max(m, blobMask(d, [x * s, y, 0.94], [0.016, 0.016, 0.03], 0.5));
        }
    }
    return mixc(c, color, m * 0.8);
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
        fur: { length: 0.045, len: (d) => 1 - blobMask(d, [0, -0.37, 0.93], [0.36, 0.27, 0.4], 0.3) * 0.6 },
        eyes: { dir: [0.27, 0.1, 0.94], r: 0.155, iris: "#4a2a14", open: -0.77 },
        brows: { color: "#6b3f1e", w: 0.12 },
        nose: heartNose("#2b1a14", 0.1, [0, -0.2, 1]),
        mouth: { dir: [0, -0.42, 0.92], w: 0.13, style: "w", line: "#4a2a1c" },
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
        fur: { length: 0.04 },
        eyes: { dir: [0.28, 0.08, 0.94], r: 0.165, iris: "#3a2228", open: -0.72, lashes: true },
        brows: { color: "#d9c9c0", w: 0.1 },
        nose: heartNose("#f48ea6", 0.06, [0, -0.2, 1], { rough: 0.35 }),
        mouth: { dir: [0, -0.36, 0.94], w: 0.11, style: "w", line: "#a35a6a", teeth: "buck" },
        whiskers: { color: "#cbbfb8", len: 0.36 },
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
        fur: { length: 0.05 },
        eyes: { dir: [0.3, 0.1, 0.93], r: 0.15, iris: "#2a1a1a", open: -0.72 },
        brows: { color: "#b56e2e", w: 0.1 },
        nose: heartNose("#f39aa7", 0.05, [0, -0.2, 1], { rough: 0.3 }),
        mouth: { dir: [0, -0.33, 0.95], w: 0.1, style: "w", line: "#9a5a50", teeth: "buck" },
        whiskers: { color: "#8a6a55", len: 0.3 },
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
        fur: { length: 0.04, len: (d) => 1 - blobMask(d, [0, -0.36, 0.92], [0.46, 0.3, 0.42], 0.3) * 0.55 },
        eyes: { dir: [0.27, 0.12, 0.94], r: 0.14, iris: "#2e1a10", open: -0.72 },
        brows: { color: "#4f2c16", w: 0.12 },
        nose: heartNose("#2a1710", 0.11, [0, -0.17, 1], { wide: 1.6 }),
        mouth: { dir: [0, -0.34, 0.95], w: 0.13, style: "w", line: "#3d2216", teeth: "bigbuck" },
        whiskers: { color: "#3a2416", len: 0.32 },
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
            c = whiskerDots(c, d, [150, 95, 50]);
            return blush(c, d, [245, 130, 110], 0.3);
        },
        skinColor: "#f0b54b",
        fur: { length: 0.035 },
        eyes: { dir: [0.27, 0.1, 0.94], r: 0.15, iris: "#6a3c10", open: -0.82 },
        brows: { color: "#b8701e", w: 0.12 },
        nose: heartNose("#b8655a", 0.085, [0, -0.17, 1]),
        mouth: { dir: [0, -0.4, 0.93], w: 0.12, style: "w", line: "#7a3d22", teeth: "fangs" },
        whiskers: { color: "#7a5030", len: 0.4 },
        ears: [{ type: "round", dir: [0.6, 0.74, 0.06], r: 0.17, inner: "#f6c98a", upright: 0.6, splay: 0.15, k: 150, c: 12, floppy: 0.4 }],
        extras: (ctx) => mane(ctx),
    };
};

SPECS.tiger = () => {
    const fur = rgb("#f38a2b"), white = rgb("#fff8ef"), black = rgb("#2a1a14");
    const whiteMask = (d) => Math.max(
        blobMask(d, [0, -0.38, 0.9], [0.42, 0.3, 0.45], 0.3),
        blobMask(d, [0.7, -0.3, 0.6], [0.32, 0.28, 0.35], 0.35),
        blobMask(d, [-0.7, -0.3, 0.6], [0.32, 0.28, 0.35], 0.35),
        blobMask(d, [0.3, 0.36, 0.88], [0.1, 0.07, 0.12], 0.4),
        blobMask(d, [-0.3, 0.36, 0.88], [0.1, 0.07, 0.12], 0.4)
    );
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
            const wm = whiteMask(d);
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
            c = whiskerDots(c, d, [70, 50, 40]);
            return blush(c, d, [255, 140, 130], 0.25);
        },
        skinColor: "#f38a2b",
        fur: { length: 0.032, len: (d) => 1 - blobMask(d, [0, -0.38, 0.9], [0.42, 0.3, 0.45], 0.3) * 0.5 },
        eyes: { dir: [0.27, 0.1, 0.94], r: 0.15, iris: "#5a3c10", open: -0.84 },
        brows: { color: "#2a1a14", w: 0.11 },
        nose: heartNose("#f29a9a", 0.08, [0, -0.17, 1]),
        mouth: { dir: [0, -0.4, 0.93], w: 0.12, style: "w", line: "#4a2a1c", teeth: "fangs" },
        whiskers: { color: "#3a2a22", len: 0.42 },
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
            const sp = Math.max(
                blobMask(d, [0.25, 0.85, 0.45], [0.1, 0.1, 0.1], 0.5),
                blobMask(d, [-0.3, 0.8, 0.5], [0.08, 0.08, 0.08], 0.5),
                blobMask(d, [0, 0.95, 0.2], [0.09, 0.09, 0.09], 0.5),
                blobMask(d, [0.55, 0.55, 0.6], [0.06, 0.06, 0.06], 0.5),
                blobMask(d, [-0.6, 0.5, 0.6], [0.07, 0.07, 0.07], 0.5)
            );
            c = mixc(c, dark, sp * 0.6);
            c = mixc(c, belly, blobMask(d, [0, -0.6, 0.75], [0.6, 0.38, 0.6], 0.35));
            return blush(c, d, [255, 145, 150], 0.45);
        },
        skin: { roughness: 0.5, sheen: 0.2, clearcoat: 0.3, normalScale: 0.08 },
        skinColor: "#86c46a",
        fur: null,
        neck: false,
        eyes: { dir: [0.3, 0.12, 0.93], r: 0.16, iris: "#2a2214", open: -0.64 },
        brows: { color: "#4f7f3c", w: 0.1 },
        nose: nostrils("#3d5e2d", [0, -0.1, 1], 0.05, 0.018),
        mouth: { dir: [0, -0.33, 0.95], w: 0.22, style: "smile", line: "#3b5a2a", curv: 0.6 },
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
            const lon = Math.abs(lonOf(d)), lat = latOf(d);
            let gill = 0;
            for (const gx of [1.35, 1.5, 1.65]) {
                const along = clamp(1 - Math.abs(lat + 0.05) / 0.28, 0, 1);
                gill = Math.max(gill, (1 - smooth(0.012, 0.025, Math.abs(lon - gx - 0.05 * lat))) * smooth(0, 0.3, along));
            }
            c = mixc(c, rgb("#2f4a66"), gill * 0.8);
            return blush(c, d, [255, 150, 170], 0.4 * (1 - b * 0.3), [0.58, -0.15, 0.79], 0.15);
        },
        skin: { roughness: 0.4, sheen: 0.15, clearcoat: 0.4, normalScale: 0.05 },
        skinColor: "#6b90b8",
        fur: null,
        eyes: { dir: [0.4, 0.13, 0.9], r: 0.14, iris: "#1a1c24", open: -0.67, outward: 0.6 },
        brows: { color: "#3c5878", w: 0.11 },
        nose: nostrils("#3c5878", [0, 0.06, 1], 0.12, 0.016),
        mouth: { dir: [0, -0.32, 0.96], w: 0.26, style: "smile", line: "#2f4a66", teeth: "shark", cavity: "#5a1520", curv: 0.7 },
        ears: [],
        extras: (ctx) => fin(ctx),
    };
};

SPECS.gorilla = () => {
    const fur = rgb("#34343b"), face = rgb("#7b6e68"), dark = rgb("#25252b");
    const faceMask = (d) => Math.max(
        blobMask(d, [0.3, 0.1, 0.94], [0.25, 0.22, 0.3], 0.3),
        blobMask(d, [-0.3, 0.1, 0.94], [0.25, 0.22, 0.3], 0.3),
        blobMask(d, [0, -0.32, 0.92], [0.45, 0.42, 0.45], 0.25)
    );
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
            const m = faceMask(d);
            c = mixc(c, face, m);
            return blush(c, d, [200, 120, 120], 0.25 * m);
        },
        skin: { sheen: 0.8, sheenColor: "#9a9aa8" },
        lidColor: "#6a5e58",
        skinColor: "#34343b",
        fur: { length: 0.055, gravity: 0.8, len: (d) => 1 - faceMask(d) },
        eyes: { dir: [0.26, 0.1, 0.95], r: 0.13, iris: "#5a3418", style: "cute", open: -0.68 },
        brows: { color: "#1c1c22", w: 0.15, lift: 0.17 },
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
        mouth: { dir: [0, -0.48, 0.88], w: 0.2, style: "smile", line: "#2a2020", teeth: "rows", curv: 0.7 },
        ears: [{ type: "round", dir: [0.97, 0.05, 0.05], r: 0.12, inner: "#5c524e", outerMat: new THREE.MeshPhysicalMaterial({ color: "#6a5e58", roughness: 0.7 }), upright: 0.1, splay: 0.0, k: 200, c: 15, floppy: 0.2, noFur: true }],
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
            // A few freckles on the forehead.
            let f = 0;
            for (const [x, y] of [[0.12, 0.55], [-0.18, 0.6], [0.02, 0.7], [0.25, 0.45], [-0.3, 0.42]]) f = Math.max(f, blobMask(d, [x, y, 0.8], [0.025, 0.025, 0.05], 0.5));
            c = mixc(c, dark, f * 0.5);
            return blush(c, d, [255, 160, 190], 0.4, [0.55, -0.3, 0.75], 0.14);
        },
        skin: { roughness: 0.35, sheen: 0.3, clearcoat: 0.5, iridescence: 0.45, normalScale: 0.04 },
        skinColor: "#93e06c",
        fur: null,
        eyes: { dir: [0.36, 0.06, 0.92], r: 0.2, style: "void", shape: [1.25, 0.72, 0.42], tilt: 0.42, sink: 0.05, outward: 0.5 },
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
    if (ctx.furLayers > 0) addFur(m, { length: 0.07, layers: Math.max(4, Math.round(ctx.furLayers * 0.8)), repeat: [2, 2], gravity: 0.6 });
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
        side: THREE.DoubleSide, normalMap: furNormal(), normalScale: new THREE.Vector2(0.06, 0.06),
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
        eyes[0].update({ blink: e.blinkL, squint: e.squintL * 0.4 + e.smile * 0.3, wide: e.wideL + e.surprise * 0.4, lookX: e.lookXL, lookY: e.lookYL, angry: e.angry, sad: e.sad });
        eyes[1].update({ blink: e.blinkR, squint: e.squintR * 0.4 + e.smile * 0.3, wide: e.wideR + e.surprise * 0.4, lookX: e.lookXR, lookY: e.lookYR, angry: e.angry, sad: e.sad });
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
    Measured from the canonical face model: eyes at y ≈ 2.7, nose tip y ≈ -1.1 / z ≈ 7.5,
    mouth y ≈ -4.3, chin y ≈ -9.4, face width ±7.7. A real head (with hair) reaches to about
    y ≈ 14 and z ≈ -14. A head of radius 13 centred at (0, 1.4, -2.2), with a taller crown, covers all
    of it, and puts the animal's eyes (direction y ≈ 0.1), nose (y ≈ -0.2) and mouth (y ≈ -0.4) on the
    person's own. The engine then slides it so its eyes are exactly in front of the person's eyes.
*/
export const HEAD_CENTER = [0, 1.4, -2.2];
export const HEAD_SCALE = 13.0;

const specCache = {};

/*
    Build a rig, in face space: it can be put straight under the face's pose matrix.
    options.furLayers: fewer fur shells for faces in the background (0 = no fur).
*/
export function buildAnimal(id, options = {}) {
    let rig;
    if (id === "ghost") {
        rig = buildGhost();
    } else {
        const make = SPECS[id] || SPECS.bear;
        if (!specCache[id]) specCache[id] = make();
        rig = buildStandard(specCache[id], options);
    }
    const holder = new THREE.Group();
    holder.position.set(HEAD_CENTER[0], HEAD_CENTER[1], HEAD_CENTER[2]);
    holder.scale.setScalar(id === "ghost" ? HEAD_SCALE * 1.02 : HEAD_SCALE);
    holder.add(rig.object);
    return { ...rig, object: holder, id };
}
