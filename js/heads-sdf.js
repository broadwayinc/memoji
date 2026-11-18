/*
    heads-sdf.js: the cute animal heads, sculpted as signed distance fields, and the mesher that turns
    them into triangles. Pure math, no three.js, so it also runs in a Web Worker (heads-worker.js).

    Every head is a few soft shapes blended together (smooth union, smooth subtraction): a bean shaped
    head with full cheeks, a muzzle that bulges out, cheek tufts, a mane, a fin, a brow ridge, gill
    grooves... That is where the detail comes from, like clay, not from millions of polygons.
    The colour of each point (pastel colour blocks: muzzle, inner ears, stripes, blush) is a function
    of its position, written into vertex colours.

    Unit space: head radius about 1, +Z is the front, +Y is up. +X is the animal's left.
*/

/* ------------------------------------------------------------------ math */

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const mix = (a, b, t) => a + (b - a) * t;
export const smooth = (e0, e1, x) => {
    const t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
};

export function sdEll(x, y, z, cx, cy, cz, rx, ry, rz) {
    const px = (x - cx) / rx, py = (y - cy) / ry, pz = (z - cz) / rz;
    const k0 = Math.sqrt(px * px + py * py + pz * pz);
    const qx = px / rx, qy = py / ry, qz = pz / rz;
    const k1 = Math.sqrt(qx * qx + qy * qy + qz * qz);
    return k1 > 1e-9 ? (k0 * (k0 - 1)) / k1 : -Math.min(rx, ry, rz);
}

export function sdSphere(x, y, z, cx, cy, cz, r) {
    const dx = x - cx, dy = y - cy, dz = z - cz;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) - r;
}

// A capsule whose radius goes from r1 (at a) to r2 (at b).
export function sdCone(x, y, z, ax, ay, az, bx, by, bz, r1, r2) {
    const pax = x - ax, pay = y - ay, paz = z - az;
    const bax = bx - ax, bay = by - ay, baz = bz - az;
    const h = clamp((pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz), 0, 1);
    const dx = pax - bax * h, dy = pay - bay * h, dz = paz - baz * h;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) - mix(r1, r2, h);
}

export const smin = (a, b, k) => {
    const h = Math.max(k - Math.abs(a - b), 0) / k;
    return Math.min(a, b) - h * h * k * 0.25;
};
export const smax = (a, b, k) => -smin(-a, -b, k);

// "#rrggbb" to linear RGB (vertex colours are linear).
export function col(hex) {
    const n = parseInt(hex.slice(1), 16);
    const f = (c) => Math.pow(c / 255, 2.2);
    return [f((n >> 16) & 255), f((n >> 8) & 255), f(n & 255)];
}
export const mixc = (a, b, t) => [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)];
// 1 inside a region (d < 0), 0 outside, with a soft (anti-aliased) edge.
const inside = (d, w = 0.012) => smooth(w, -w, d);
// The surface of a blended-in part (muzzle, cheek pad): a little past the part's own boundary.
const onPart = (d, m = 0.04, w = 0.012) => smooth(w, -w, d - m);

/* ------------------------------------------------------------------ mesher (surface nets) */

const EDGES = [];
for (let b = 0; b < 8; b++) for (const bit of [1, 2, 4]) if (!(b & bit)) EDGES.push([b, b | bit]);

function gradient(sdf, x, y, z, e = 0.0015) {
    const a = sdf(x + e, y - e, z - e), b = sdf(x - e, y - e, z + e), c = sdf(x - e, y + e, z - e), d = sdf(x + e, y + e, z + e);
    let gx = a - b - c + d, gy = -a - b + c + d, gz = -a + b - c + d;
    const l = Math.sqrt(gx * gx + gy * gy + gz * gz) || 1;
    return [gx / l, gy / l, gz / l];
}

/*
    Surface nets: one vertex per grid cell the surface passes through (the average of where it crosses
    the cell's edges, then pulled exactly onto the surface), one quad per grid edge it crosses.
    Normals come from the field itself, so the shading is perfectly smooth at any polygon count.
*/
export function meshSDF(sdf, color, bmin, bmax, cell, morphs = null) {
    const nx = Math.ceil((bmax[0] - bmin[0]) / cell) + 1;
    const ny = Math.ceil((bmax[1] - bmin[1]) / cell) + 1;
    const nz = Math.ceil((bmax[2] - bmin[2]) / cell) + 1;
    const vals = new Float32Array(nx * ny * nz);
    let q = 0;
    for (let k = 0; k < nz; k++) {
        const z = bmin[2] + k * cell;
        for (let j = 0; j < ny; j++) {
            const y = bmin[1] + j * cell;
            for (let i = 0; i < nx; i++) vals[q++] = sdf(bmin[0] + i * cell, y, z);
        }
    }
    const V = (i, j, k) => vals[i + nx * (j + ny * k)];
    const cx = nx - 1, cy = ny - 1, cz = nz - 1;
    const cellVert = new Int32Array(cx * cy * cz).fill(-1);
    const pos = [];
    const cv = new Float32Array(8);
    for (let k = 0; k < cz; k++) {
        for (let j = 0; j < cy; j++) {
            for (let i = 0; i < cx; i++) {
                let mask = 0;
                for (let b = 0; b < 8; b++) {
                    const v = V(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
                    cv[b] = v;
                    if (v < 0) mask |= 1 << b;
                }
                if (mask === 0 || mask === 255) continue;
                let sx = 0, sy = 0, sz = 0, n = 0;
                for (const [a, b] of EDGES) {
                    const va = cv[a], vb = cv[b];
                    if ((va < 0) === (vb < 0)) continue;
                    const t = va / (va - vb);
                    sx += (a & 1) + t * ((b & 1) - (a & 1));
                    sy += ((a >> 1) & 1) + t * (((b >> 1) & 1) - ((a >> 1) & 1));
                    sz += ((a >> 2) & 1) + t * (((b >> 2) & 1) - ((a >> 2) & 1));
                    n++;
                }
                cellVert[i + cx * (j + cy * k)] = pos.length / 3;
                pos.push(bmin[0] + (i + sx / n) * cell, bmin[1] + (j + sy / n) * cell, bmin[2] + (k + sz / n) * cell);
            }
        }
    }
    const C = (i, j, k) => (i < 0 || j < 0 || k < 0 || i >= cx || j >= cy || k >= cz ? -1 : cellVert[i + cx * (j + cy * k)]);
    const idx = [];
    const quad = (a, b, c, d) => {
        if (a < 0 || b < 0 || c < 0 || d < 0) return;
        idx.push(a, b, c, a, c, d);
    };
    for (let k = 0; k < nz; k++) {
        for (let j = 0; j < ny; j++) {
            for (let i = 0; i < nx; i++) {
                const v0 = V(i, j, k) < 0;
                if (i + 1 < nx && v0 !== V(i + 1, j, k) < 0) quad(C(i, j - 1, k - 1), C(i, j, k - 1), C(i, j, k), C(i, j - 1, k));
                if (j + 1 < ny && v0 !== V(i, j + 1, k) < 0) quad(C(i - 1, j, k - 1), C(i - 1, j, k), C(i, j, k), C(i, j, k - 1));
                if (k + 1 < nz && v0 !== V(i, j, k + 1) < 0) quad(C(i - 1, j - 1, k), C(i, j - 1, k), C(i, j, k), C(i - 1, j, k));
            }
        }
    }
    // Pull every vertex onto the surface, and take its normal and colour.
    const count = pos.length / 3;
    const P = new Float32Array(pos), N = new Float32Array(count * 3), K = new Float32Array(count * 3);
    for (let v = 0; v < count; v++) {
        let x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
        for (let it = 0; it < 2; it++) {
            const d = sdf(x, y, z);
            const g = gradient(sdf, x, y, z);
            x -= g[0] * d; y -= g[1] * d; z -= g[2] * d;
        }
        const g = gradient(sdf, x, y, z);
        P.set([x, y, z], v * 3);
        N.set(g, v * 3);
        K.set(color(x, y, z, g[0], g[1], g[2]), v * 3);
    }
    // Wind every triangle so its front faces outwards.
    const I = new Uint32Array(idx);
    for (let t = 0; t < I.length; t += 3) {
        const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
        const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
        const wx = P[c] - P[a], wy = P[c + 1] - P[a + 1], wz = P[c + 2] - P[a + 2];
        const fx = uy * wz - uz * wy, fy = uz * wx - ux * wz, fz = ux * wy - uy * wx;
        if (fx * (N[a] + N[b] + N[c]) + fy * (N[a + 1] + N[b + 1] + N[c + 1]) + fz * (N[a + 2] + N[b + 2] + N[c + 2]) < 0) {
            const s = I[t + 1];
            I[t + 1] = I[t + 2];
            I[t + 2] = s;
        }
    }
    const out = { position: P, normal: N, color: K, index: I, morphs: {} };
    if (morphs) {
        for (const [name, fn] of Object.entries(morphs)) {
            const D = new Float32Array(count * 3);
            for (let v = 0; v < count; v++) {
                const d = fn(P[v * 3], P[v * 3 + 1], P[v * 3 + 2], N[v * 3], N[v * 3 + 1], N[v * 3 + 2]);
                D[v * 3] = d[0]; D[v * 3 + 1] = d[1]; D[v * 3 + 2] = d[2];
            }
            out.morphs[name] = D;
        }
    }
    return out;
}

/*
    The point where the ray from `from` going along `dir` first meets the surface (sphere tracing from
    outside), and the normal there. Used to place eyes, noses, mouths and ears exactly on the skin.
*/
export function surfacePoint(sdf, dir, from = [0, 0, 0]) {
    const l = Math.hypot(dir[0], dir[1], dir[2]);
    const d = [dir[0] / l, dir[1] / l, dir[2] / l];
    // Start outside and walk inwards.
    let t = 3;
    let x = from[0] + d[0] * t, y = from[1] + d[1] * t, z = from[2] + d[2] * t;
    for (let i = 0; i < 200; i++) {
        const s = sdf(x, y, z);
        if (Math.abs(s) < 1e-4) break;
        t -= s * 0.9;
        x = from[0] + d[0] * t; y = from[1] + d[1] * t; z = from[2] + d[2] * t;
    }
    return { p: [x, y, z], n: gradient(sdf, x, y, z) };
}

/* ------------------------------------------------------------------ shared shapes */

// A wide, soft bean: the head, with fuller cheeks low at the front.
function bean(x, y, z, w = 1.1, h = 0.92, d = 0.97, jowl = 1) {
    const d0 = sdEll(x, y, z, 0, 0.02, 0, w, h, d);
    const j = sdEll(Math.abs(x), y, z, 0.42 * w, -0.33 * h, 0.28, 0.6 * w * jowl, 0.52 * h, 0.6 * d);
    return smin(d0, j, 0.25);
}

// Shared morph targets of a head: jaw drop, cheek puff, smile (cheeks lift).
function headMorphs(cheek = [0.58, -0.3, 0.55]) {
    return {
        jaw: (x, y, z) => {
            const w = smooth(-0.12, -0.6, y) * smooth(0.25, 0.8, z) * (1 - smooth(0.35, 0.75, Math.abs(x)));
            return [0, -0.1 * w, 0];
        },
        puff: (x, y, z, nx, ny, nz) => {
            const dx = Math.abs(x) - cheek[0], dy = y - cheek[1], dz = z - cheek[2];
            const w = Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * 0.24 * 0.24));
            return [nx * 0.09 * w, ny * 0.09 * w, nz * 0.09 * w];
        },
        smile: (x, y, z) => {
            const dx = Math.abs(x) - 0.42, dy = y + 0.25;
            const w = Math.exp(-(dx * dx + dy * dy) / (2 * 0.16 * 0.16)) * smooth(0.3, 0.8, z);
            return [0, 0.035 * w, 0.02 * w];
        },
    };
}

/* ------------------------------------------------------------------ ears */

/*
    A round ear (a soft disc with its inside scooped out) or a long ear (bunny), built in its own space:
    its base at the origin, pointing up (+Y), its inside facing +Z.
*/
function roundEar(r, outer, inner, opts = {}) {
    const cy = r * 0.85;
    const sdf = (x, y, z) => {
        const d = sdEll(x, y, z, 0, cy, 0, r, r * (opts.tall ?? 0.95), r * 0.42);
        const cup = sdEll(x, y, z, 0, cy - r * 0.06, r * 0.32, r * 0.62, r * 0.58 * (opts.tall ?? 0.95), r * 0.3);
        return smax(d, -cup, r * 0.12);
    };
    const back = opts.back ? col(opts.back) : null;
    const O = col(outer), I = col(inner);
    const color = (x, y, z) => {
        const inCup = inside(sdEll(x, y, z, 0, cy - r * 0.06, r * 0.32, r * 0.7, r * 0.66 * (opts.tall ?? 0.95), r * 0.4), r * 0.04);
        let c = mixc(O, I, inCup * smooth(-r * 0.05, r * 0.08, z));
        if (back) c = mixc(c, back, smooth(-r * 0.05, -r * 0.25, z) * smooth(cy - r * 0.3, cy + r * 0.2, y));
        return c;
    };
    const m = r * 1.3;
    return { sdf, color, bounds: [[-m, -0.2 * r, -m * 0.6], [m, cy + m, m * 0.6]], cell: r / 15 };
}

function longEar(len, w, outer, inner) {
    const cy = len * 0.55;
    const sdf = (x, y, z) => {
        const d = sdEll(x, y, z, 0, cy, 0, w, len * 0.55, w * 0.5);
        const cup = sdEll(x, y, z, 0, cy + len * 0.03, w * 0.4, w * 0.58, len * 0.46, w * 0.32);
        return smax(d, -cup, w * 0.15);
    };
    const O = col(outer), I = col(inner);
    const color = (x, y, z) => mixc(O, I, inside(sdEll(x, y, z, 0, cy + len * 0.03, w * 0.4, w * 0.66, len * 0.5, w * 0.45), w * 0.06) * smooth(-0.01, 0.04, z));
    return { sdf, color, bounds: [[-w * 1.3, -0.1, -w], [w * 1.3, len * 1.2, w]], cell: w / 13 };
}

/*
    A big koala ear: a round disc with a fluffy, scalloped rim and a soft white inside.
*/
function fluffyEar(r, outer, inner) {
    const cy = r * 0.85;
    const N = 11;
    const tufts = [];
    for (let i = 0; i < N; i++) {
        const a = Math.PI * (0.05 + (0.9 * i) / (N - 1)) - Math.PI * 0.4; // around the outer edge
        tufts.push([Math.cos(a) * r * 0.86, cy + Math.sin(a) * r * 0.86]);
    }
    const sdf = (x, y, z) => {
        let d = sdEll(x, y, z, 0, cy, 0, r, r * 0.95, r * 0.36);
        for (const [tx, ty] of tufts) d = smin(d, sdSphere(Math.abs(x), y, z, tx, ty, r * 0.12, r * 0.2), r * 0.1);
        const cup = sdEll(x, y, z, 0, cy - r * 0.04, r * 0.3, r * 0.6, r * 0.56, r * 0.24);
        return smax(d, -cup, r * 0.1);
    };
    const O = col(outer), I = col(inner);
    const color = (x, y, z) => {
        // Grey disc, a white fluffy inside, and a lighter fluffy rim.
        const rr = Math.hypot(x, (y - cy) / 0.95);
        const front = smooth(-r * 0.02, r * 0.12, z);
        const centre = inside(rr - r * 0.66, r * 0.06) * front;
        const rim = smooth(r * 0.82, r * 0.95, rr) * front;
        return mixc(mixc(O, I, rim * 0.45), I, centre);
    };
    const m = r * 1.35;
    return { sdf, color, bounds: [[-m, -0.25 * r, -m * 0.5], [m, cy + m, m * 0.55]], cell: r / 16 };
}

/* ------------------------------------------------------------------ the zoo */

/*
    Each animal: head { sdf, color, bounds, cell, morphs } and an optional ear, plus where its features go
    (directions from the head centre, found on the surface at build time) and their colours.
*/
export const HEADS = {};
const H = HEADS;

H.bear = (() => {
    const fur = col("#b98466"), muzzle = col("#f8e7d5"), cheek = col("#e9a996");
    const mz = [0, -0.34, 0.7, 0.44, 0.31, 0.38];
    const sdf = (x, y, z) => smin(bean(x, y, z), sdEll(x, y, z, ...mz), 0.14);
    const color = (x, y, z) => {
        let c = mixc(fur, muzzle, onPart(sdEll(x, y, z, ...mz)));
        return mixc(c, cheek, inside(sdSphere(Math.abs(x), y, z, 0.62, -0.14, 0.72, 0.14), 0.04) * 0.5);
    };
    return {
        head: { sdf, color, bounds: [[-1.4, -1.2, -1.2], [1.4, 1.2, 1.3]], cell: 0.03, morphs: headMorphs() },
        ear: roundEar(0.3, "#b98466", "#f0c2b0"),
        ears: { dir: [0.62, 0.74, 0.02], upright: 0.5, splay: 0.12, sink: 0.12, k: 140, c: 12, floppy: 0.4 },
        eyes: { dir: [0.3, -0.03, 0.94], r: 0.09 },
        brows: { color: "#9c7560", lift: 0.2 },
        nose: { dir: [0, -0.2, 1], color: "#5a3f37", size: 0.12, wide: 1.4 },
        mouth: { dir: [0, -0.43, 0.9], w: 0.13, style: "w", line: "#6d4e43" },
    };
})();

H.bunny = (() => {
    const fur = col("#f7eef0"), pad = col("#ffffff"), blush = col("#f6b8c8");
    const sdf = (x, y, z) => {
        let d = bean(x, y, z, 1.0, 0.95, 0.95, 0.9);
        d = smin(d, sdEll(Math.abs(x), y, z, 0.13, -0.33, 0.8, 0.17, 0.13, 0.15), 0.06);
        return d;
    };
    const color = (x, y, z) => {
        let c = mixc(fur, pad, onPart(sdEll(Math.abs(x), y, z, 0.13, -0.33, 0.8, 0.17, 0.13, 0.15), 0.03));
        return mixc(c, blush, inside(sdSphere(Math.abs(x), y, z, 0.56, -0.16, 0.74, 0.15), 0.05) * 0.85);
    };
    return {
        head: { sdf, color, bounds: [[-1.3, -1.2, -1.2], [1.3, 1.2, 1.3]], cell: 0.03, morphs: headMorphs() },
        ear: longEar(0.95, 0.22, "#fdf7f8", "#f7c2cf"),
        ears: { dir: [0.3, 0.92, 0.0], upright: 0.85, splay: 0.18, tilt: 0.08, sink: 0.12, k: 45, c: 4, floppy: 1.4 },
        eyes: { dir: [0.3, -0.05, 0.94], r: 0.085 },
        brows: { color: "#e2cfd2", lift: 0.19 },
        nose: { dir: [0, -0.22, 1], color: "#f39ab1", size: 0.065 },
        mouth: { dir: [0, -0.42, 0.92], w: 0.1, style: "w", line: "#c47b8c", teeth: "buck" },
    };
})();

H.hamster = (() => {
    const fur = col("#f3bd86"), white = col("#fff8ef"), blush = col("#f7b4b8");
    const sdf = (x, y, z) => {
        let d = bean(x, y, z, 1.15, 0.9, 0.95, 1.05);
        d = smin(d, sdEll(Math.abs(x), y, z, 0.6, -0.3, 0.42, 0.42, 0.36, 0.42), 0.15);
        d = smin(d, sdEll(x, y, z, 0, -0.33, 0.78, 0.2, 0.16, 0.16), 0.08);
        d = smin(d, sdCone(x, y, z, 0, 0.82, 0.08, 0.04, 1.06, 0.12, 0.07, 0.02), 0.08); // tuft
        return d;
    };
    const color = (x, y, z) => {
        const face = Math.max(inside(sdEll(x, y, z, 0, -0.42, 0.55, 1.0, 0.5, 0.75), 0.02), inside(Math.abs(x) - 0.06 - 0.04 * smooth(0.3, 0.9, y), 0.015) * smooth(0.0, 0.1, y) * smooth(0.3, 0.6, z));
        let c = mixc(fur, white, face);
        return mixc(c, blush, inside(sdSphere(Math.abs(x), y, z, 0.62, -0.18, 0.72, 0.14), 0.05) * 0.8);
    };
    return {
        head: { sdf, color, bounds: [[-1.5, -1.2, -1.2], [1.5, 1.25, 1.3]], cell: 0.03, morphs: headMorphs([0.62, -0.3, 0.5]) },
        ear: roundEar(0.2, "#f3bd86", "#f7b5bf"),
        ears: { dir: [0.55, 0.78, 0.15], upright: 0.5, splay: 0.25, sink: 0.08, k: 120, c: 10, floppy: 0.5 },
        eyes: { dir: [0.3, -0.03, 0.93], r: 0.085 },
        brows: { color: "#c99a70", lift: 0.19 },
        nose: { dir: [0, -0.24, 1], color: "#f2a2ae", size: 0.055 },
        mouth: { dir: [0, -0.4, 0.92], w: 0.1, style: "w", line: "#b07a6c", teeth: "buck" },
        whiskers: { color: "#b8957c", len: 0.16 },
    };
})();

H.beaver = (() => {
    const fur = col("#9c7055"), muzzle = col("#f1dcc3"), dark = col("#7f5a44");
    const mz = [0, -0.33, 0.7, 0.52, 0.31, 0.39];
    const sdf = (x, y, z) => {
        let d = smin(bean(x, y, z, 1.05, 0.95, 0.95), sdEll(x, y, z, ...mz), 0.13);
        d = smin(d, sdCone(x, y, z, 0, 0.86, 0.1, 0.0, 1.08, 0.18, 0.08, 0.02), 0.08);
        return d;
    };
    const color = (x, y, z) => mixc(mixc(fur, dark, smooth(0.75, 1.0, y) * 0.8), muzzle, onPart(sdEll(x, y, z, ...mz)));
    return {
        head: { sdf, color, bounds: [[-1.4, -1.2, -1.2], [1.4, 1.25, 1.3]], cell: 0.03, morphs: headMorphs() },
        ear: roundEar(0.17, "#9c7055", "#7f5a44"),
        ears: { dir: [0.86, 0.42, 0.05], upright: 0.3, splay: 0.35, sink: 0.08, k: 150, c: 12, floppy: 0.3 },
        eyes: { dir: [0.3, 0.01, 0.94], r: 0.085 },
        brows: { color: "#8a6a52", lift: 0.19 },
        nose: { dir: [0, -0.18, 1], color: "#5b4036", size: 0.12, wide: 1.6 },
        mouth: { dir: [0, -0.4, 0.92], w: 0.13, style: "w", line: "#6f5244", teeth: "bigbuck" },
    };
})();

H.lion = (() => {
    const face = col("#fde39a"), mane = col("#f6b375"), maneDark = col("#ee9f5f"), muzzle = col("#fff5e0");
    const N = 14;
    const ring = [];
    for (let i = 0; i < N; i++) {
        const a = (i / N) * Math.PI * 2 + 0.11;
        ring.push([Math.cos(a) * 1.0, Math.sin(a) * 0.96 - 0.02]);
    }
    const maneSdf = (x, y, z) => {
        // Scalloped mane: a ring of soft balls behind the face, and a cushion at the back.
        let m = sdEll(x, y, z, 0, -0.02, -0.32, 1.05, 1.0, 0.62);
        for (const [rx, ry] of ring) m = smin(m, sdSphere(x, y, z, rx, ry, -0.12, 0.3), 0.12);
        return m;
    };
    const sdf = (x, y, z) => {
        let d = bean(x, y, z, 0.95, 0.88, 0.92);
        d = smin(d, maneSdf(x, y, z), 0.1);
        d = smin(d, sdEll(Math.abs(x), y, z, 0.12, -0.33, 0.78, 0.17, 0.14, 0.15), 0.06);
        d = smin(d, sdEll(x, y, z, 0, -0.5, 0.72, 0.14, 0.08, 0.1), 0.05);
        return d;
    };
    const color = (x, y, z) => {
        // Face where the face is the nearer shape, mane where the mane is: the colour follows the sculpt.
        const f = smooth(0.015, -0.015, bean(x, y, z, 0.95, 0.88, 0.92) - maneSdf(x, y, z));
        let c = mixc(mixc(mane, maneDark, smooth(-0.2, -0.6, z) * 0.6), face, f);
        return mixc(c, muzzle, Math.max(onPart(sdEll(Math.abs(x), y, z, 0.12, -0.33, 0.78, 0.17, 0.14, 0.15), 0.035), onPart(sdEll(x, y, z, 0, -0.5, 0.72, 0.14, 0.08, 0.1), 0.03)));
    };
    return {
        head: { sdf, color, bounds: [[-1.5, -1.45, -1.2], [1.5, 1.45, 1.3]], cell: 0.032, morphs: headMorphs() },
        ear: roundEar(0.19, "#fde39a", "#f7c98d"),
        ears: { dir: [0.58, 0.72, 0.1], upright: 0.55, splay: 0.15, sink: 0.12, k: 150, c: 12, floppy: 0.4 },
        eyes: { dir: [0.3, -0.02, 0.94], r: 0.085 },
        brows: { color: "#d9a35c", lift: 0.18 },
        nose: { dir: [0, -0.2, 1], color: "#c98777", size: 0.09 },
        mouth: { dir: [0, -0.43, 0.9], w: 0.12, style: "w", line: "#9b6a50", teeth: "fangs" },
        whiskers: { color: "#a77e60", len: 0.16 },
    };
})();

H.tiger = (() => {
    const fur = col("#f8b57a"), white = col("#fff8ef"), stripe = col("#86624f");
    const sdf = (x, y, z) => {
        const ax = Math.abs(x);
        let d = bean(x, y, z, 1.06, 0.92, 0.95);
        // Cheek tufts: two soft points out to each side.
        d = smin(d, sdCone(ax, y, z, 0.78, -0.28, 0.42, 1.12, -0.36, 0.28, 0.2, 0.04), 0.1);
        d = smin(d, sdCone(ax, y, z, 0.7, -0.45, 0.42, 0.98, -0.6, 0.3, 0.15, 0.03), 0.08);
        d = smin(d, sdEll(ax, y, z, 0.12, -0.33, 0.78, 0.17, 0.14, 0.15), 0.06);
        return d;
    };
    const color = (x, y, z) => {
        const ax = Math.abs(x);
        let c = fur;
        const w = Math.max(
            onPart(sdEll(ax, y, z, 0.12, -0.33, 0.78, 0.17, 0.14, 0.15), 0.05),
            inside(sdEll(ax, y, z, 0.8, -0.38, 0.4, 0.38, 0.24, 0.4), 0.015),
            inside(sdSphere(ax, y, z, 0.3, 0.24, 0.86, 0.065), 0.012) // spots over the eyes
        );
        c = mixc(c, white, w);
        // Stripes: three on the forehead, three on each side.
        let s = 0;
        for (const [yc, half] of [[0.58, 0.26], [0.72, 0.2], [0.84, 0.13]]) {
            s = Math.max(s, inside(Math.abs(y - yc - 0.04 * ax) - 0.03, 0.01) * inside(ax - half, 0.02) * smooth(0.0, 0.3, z));
        }
        for (const yc of [0.25, 0.05, -0.15]) {
            s = Math.max(s, inside(Math.abs(y - yc + 0.25 * (ax - 0.85)) - 0.035, 0.01) * smooth(0.78, 0.9, ax) * (1 - w));
        }
        return mixc(c, stripe, s);
    };
    return {
        head: { sdf, color, bounds: [[-1.5, -1.2, -1.2], [1.5, 1.2, 1.3]], cell: 0.03, morphs: headMorphs() },
        ear: roundEar(0.25, "#f8b57a", "#fff3e6", { back: "#86624f" }),
        ears: { dir: [0.6, 0.74, 0.05], upright: 0.55, splay: 0.14, sink: 0.12, k: 140, c: 12, floppy: 0.4 },
        eyes: { dir: [0.3, -0.01, 0.94], r: 0.085 },
        brows: { color: "#86624f", lift: 0.18 },
        nose: { dir: [0, -0.2, 1], color: "#f19ea2", size: 0.085 },
        mouth: { dir: [0, -0.43, 0.9], w: 0.12, style: "w", line: "#86624f", teeth: "fangs" },
        whiskers: { color: "#86624f", len: 0.17 },
    };
})();

H.koala = (() => {
    const fur = col("#8c8aa0"), chin = col("#dddce6"), blush = col("#eab3c2");
    const sdf = (x, y, z) => {
        let d = bean(x, y, z, 1.1, 0.9, 0.95, 1.05);
        // A soft rise under the big nose, and fuller cheeks.
        d = smin(d, sdEll(x, y, z, 0, -0.3, 0.7, 0.32, 0.26, 0.3), 0.12);
        return d;
    };
    const color = (x, y, z) => {
        let c = mixc(fur, chin, inside(sdEll(x, y, z, 0, -0.6, 0.6, 0.55, 0.3, 0.5), 0.03));
        return mixc(c, blush, inside(sdSphere(Math.abs(x), y, z, 0.6, -0.2, 0.72, 0.13), 0.04) * 0.55);
    };
    return {
        head: { sdf, color, bounds: [[-1.45, -1.15, -1.2], [1.45, 1.15, 1.3]], cell: 0.03, morphs: headMorphs() },
        ear: fluffyEar(0.44, "#8c8aa0", "#f7f5f9"),
        ears: { dir: [0.86, 0.5, 0.0], upright: 0.35, splay: 0.45, sink: 0.14, k: 110, c: 9, floppy: 0.6 },
        eyes: { dir: [0.3, 0.02, 0.94], r: 0.08 },
        brows: { color: "#6f6d84", lift: 0.18 },
        nose: { dir: [0, -0.14, 1], color: "#4d4857", size: 0.15, shape: "oval" },
        mouth: { dir: [0, -0.5, 0.88], w: 0.11, style: "smile", line: "#6e6a7a", curv: 0.8 },
    };
})();

H.monkey = (() => {
    const fur = col("#cf9566"), face = col("#f7dcc4"), blush = col("#f2b4a6"), dark = col("#b67c52");
    // The face: a heart shaped mask (two eye lobes and a muzzle), slightly raised from the head.
    const mask = (x, y, z) => smin(sdEll(Math.abs(x), y, z, 0.26, 0.06, 0.62, 0.34, 0.32, 0.4), sdEll(x, y, z, 0, -0.34, 0.6, 0.52, 0.38, 0.42), 0.12);
    const sdf = (x, y, z) => {
        let d = bean(x, y, z, 1.0, 0.95, 0.95);
        d = smin(d, mask(x, y, z), 0.1);
        // A little tuft of hair on top.
        d = smin(d, sdCone(x, y, z, 0, 0.84, 0.12, 0.1, 1.12, 0.22, 0.1, 0.025), 0.1);
        d = smin(d, sdCone(x, y, z, 0.02, 0.86, 0.1, -0.12, 1.06, 0.16, 0.07, 0.02), 0.08);
        return d;
    };
    const color = (x, y, z) => {
        let c = mixc(mixc(fur, dark, smooth(0.7, 1.05, y) * 0.6), face, onPart(mask(x, y, z), 0.035));
        return mixc(c, blush, inside(sdSphere(Math.abs(x), y, z, 0.44, -0.22, 0.86, 0.11), 0.04) * 0.6);
    };
    return {
        head: { sdf, color, bounds: [[-1.3, -1.2, -1.2], [1.3, 1.3, 1.3]], cell: 0.03, morphs: headMorphs([0.5, -0.32, 0.6]) },
        ear: roundEar(0.27, "#cf9566", "#f7dcc4"),
        ears: { dir: [1, 0.06, 0.02], upright: 0.08, splay: 0, sink: 0.1, k: 160, c: 12, floppy: 0.35 },
        eyes: { dir: [0.27, 0.02, 0.96], r: 0.085 },
        brows: { color: "#a8714a", lift: 0.18 },
        nostrils: { dir: [0, -0.22, 1], color: "#b98a6e", gap: 0.045, r: 0.02 },
        mouth: { dir: [0, -0.42, 0.92], w: 0.17, style: "smile", line: "#9a6a4c", curv: 0.6 },
    };
})();

H.gorilla = (() => {
    const fur = col("#a8a3bb"), face = col("#ead9cf"), dark = col("#8f89a5");
    const faceRegion = (x, y, z) => smin(smin(sdEll(Math.abs(x), y, z, 0.3, 0.04, 0.86, 0.28, 0.25, 0.3), sdEll(x, y, z, 0, -0.36, 0.7, 0.55, 0.42, 0.4), 0.08), sdEll(x, y, z, 0, 0.0, 0.92, 0.2, 0.2, 0.2), 0.06);
    const sdf = (x, y, z) => {
        let d = bean(x, y, z, 1.02, 1.0, 0.95);
        d = smin(d, sdEll(x, y, z, 0, 0.88, 0.05, 0.32, 0.26, 0.45), 0.22); // crest
        d = smin(d, sdCone(x, y, z, -0.44, 0.22, 0.8, 0.44, 0.22, 0.8, 0.09, 0.09), 0.08); // brow ridge
        d = smin(d, sdEll(x, y, z, 0, -0.36, 0.66, 0.5, 0.38, 0.38), 0.12); // muzzle
        return d;
    };
    const color = (x, y, z) => mixc(mixc(fur, dark, smooth(0.6, 1.0, y) * 0.6), face, inside(faceRegion(x, y, z), 0.02));
    return {
        head: { sdf, color, bounds: [[-1.35, -1.2, -1.2], [1.35, 1.35, 1.3]], cell: 0.03, morphs: headMorphs() },
        ear: roundEar(0.15, "#ead9cf", "#d6c0b4"),
        ears: { dir: [0.98, 0.02, 0.02], upright: 0.1, splay: 0, sink: 0.06, k: 200, c: 15, floppy: 0.2 },
        eyes: { dir: [0.28, 0.01, 0.95], r: 0.08 },
        brows: { color: "#6f6985", lift: 0.16 },
        nose: { dir: [0, -0.14, 1], color: "#6a6170", size: 0.14, wide: 1.7, nostrils: true },
        mouth: { dir: [0, -0.5, 0.86], w: 0.2, style: "smile", line: "#7a6870", teeth: "rows", curv: 0.6 },
    };
})();

H.alien = (() => {
    const skin = col("#c2f2d0"), cheek = col("#ddf9e5"), freckle = col("#a2dcb5"), blush = col("#f8c9da");
    const sdf = (x, y, z) => smin(sdEll(x, y, z, 0, 0.14, 0, 1.04, 0.96, 0.92), sdEll(x, y, z, 0, -0.52, 0.2, 0.46, 0.4, 0.55), 0.38);
    const color = (x, y, z) => {
        let c = mixc(skin, cheek, inside(sdEll(x, y, z, 0, -0.5, 0.5, 0.45, 0.32, 0.4), 0.05) * 0.7);
        let f = 0;
        for (const [fx, fy] of [[0.12, 0.6], [-0.18, 0.66], [0.02, 0.76], [0.26, 0.5], [-0.3, 0.48]]) f = Math.max(f, inside(sdSphere(x, y, z, fx, fy, 0.72, 0.03), 0.01));
        c = mixc(c, freckle, f * smooth(0.3, 0.6, z));
        return mixc(c, blush, inside(sdSphere(Math.abs(x), y, z, 0.5, -0.28, 0.66, 0.11), 0.04) * 0.7);
    };
    return {
        head: { sdf, color, bounds: [[-1.3, -1.1, -1.2], [1.3, 1.3, 1.25]], cell: 0.03, morphs: headMorphs([0.45, -0.4, 0.5]) },
        eyes: { dir: [0.36, 0.01, 0.92], r: 0.2, style: "void", shape: [1.25, 0.72, 0.42], tilt: 0.42, sink: 0.05 },
        nostrils: { dir: [0, -0.2, 1], color: "#8fc8a2", gap: 0.035, r: 0.012 },
        mouth: { dir: [0, -0.46, 0.9], w: 0.09, style: "smile", line: "#6fa886" },
        antennae: true,
    };
})();

// Mesh data for one part ("head" or "ear") of an animal.
export function meshPart(id, part) {
    const spec = HEADS[id];
    const p = spec?.[part];
    if (!p) return null;
    return meshSDF(p.sdf, p.color, p.bounds[0], p.bounds[1], p.cell, p.morphs || null);
}
