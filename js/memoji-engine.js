/*
    memoji-engine.js turns faces into 3D animal memoji, live from the camera or on a still photo.

    - MediaPipe Face Landmarker (WebAssembly + WebGL) finds up to 6 faces and returns, for each,
      478 landmarks, 52 blendshapes (eyeBlinkLeft, jawOpen, ...) and a 4x4 pose matrix.
    - three.js (WebGL) renders an animal head under every face's pose matrix.
      The camera uses MediaPipe's own projection (vertical field of view 63 degrees),
      so a head drawn with the pose matrix lands exactly on the face in the picture.
    - The closest face (the biggest one) wears the user's spirit animal, everyone else a random one.
    - Lighting adapts to the camera: the brightness and colour of the frame set the ambient light
      and exposure, the shading on the main face (left / right cheek, forehead / chin) sets the
      direction of the key light, and a blurred copy of the frame is the environment map the
      glossy eyes and noses reflect.
*/
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { clone as skeletonClone } from "three/addons/utils/SkeletonUtils.js";
import { ANIMALS, buildAnimal } from "./animals.js";

export { ANIMALS };

const MP_VERSION = "0.10.21";
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const MAX_FACES = 6;
const THREE_LIBS = "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/libs/";

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/* ------------------------------------------------------------------ face landmarker */

let landmarkerPromise = null;

/*
    Loads the Face Landmarker once. It tries the GPU delegate first (WebGL) and falls back to the CPU
    (WebAssembly with SIMD) when the GPU one cannot start, for example on a browser without WebGL2.
*/
export function loadFaceLandmarker() {
    if (!landmarkerPromise) {
        landmarkerPromise = (async () => {
            const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
            const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
            const options = (delegate) => ({
                baseOptions: { modelAssetPath: MODEL_URL, delegate },
                runningMode: "VIDEO",
                numFaces: MAX_FACES,
                outputFaceBlendshapes: true,
                outputFacialTransformationMatrixes: true,
                minFaceDetectionConfidence: 0.45,
                minFacePresenceConfidence: 0.45,
                minTrackingConfidence: 0.45,
            });
            let lm;
            try {
                lm = await FaceLandmarker.createFromOptions(fileset, options("GPU"));
            } catch (err) {
                console.warn("GPU face tracking unavailable, using CPU.", err);
                lm = await FaceLandmarker.createFromOptions(fileset, options("CPU"));
            }
            lm.__mode = "VIDEO";
            return lm;
        })();
        landmarkerPromise.catch(() => {
            landmarkerPromise = null;
        });
    }
    return landmarkerPromise;
}

async function setRunningMode(lm, mode) {
    if (lm.__mode !== mode) {
        await lm.setOptions({ runningMode: mode });
        lm.__mode = mode;
    }
}

/* ------------------------------------------------------------------ custom models (glTF / GLB / VRM) */

const customModels = new Map(); // id -> { name, template, fit }

/*
    MediaPipe's blendshape names follow Apple ARKit. A model's morph targets are matched to them
    loosely: case, "_", "-" and spaces are ignored and Left / Right / _L / _R are the same.
*/
function normName(s) {
    let n = String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
    n = n.replace(/left$/, "l").replace(/right$/, "r");
    const alias = { blinkl: "eyeblinkl", blinkr: "eyeblinkr", mouthopen: "jawopen", aa: "jawopen", a: "jawopen", smilel: "mouthsmilel", smiler: "mouthsmiler" };
    return alias[n] || n;
}

export const DEFAULT_FIT = { scale: 1, y: 0, z: 0, ry: 0 };

/*
    Register a custom 3D model from the bytes of a .glb, .gltf (with embedded buffers) or .vrm file.
    The model is centred and scaled to the size of a head; fit fine-tunes it (scale, up/down,
    forward/back, turn).
*/
let gltfLoader = null;
// glTF loader that also reads Draco and Meshopt compressed meshes and KTX2 (Basis) textures.
function getGltfLoader() {
    if (!gltfLoader) {
        gltfLoader = new GLTFLoader();
        const draco = new DRACOLoader();
        draco.setDecoderPath(THREE_LIBS + "draco/gltf/");
        gltfLoader.setDRACOLoader(draco);
        const ktx2 = new KTX2Loader();
        ktx2.setTranscoderPath(THREE_LIBS + "basis/");
        ktx2.detectSupport(thumbRenderer().renderer);
        gltfLoader.setKTX2Loader(ktx2);
        gltfLoader.setMeshoptDecoder(MeshoptDecoder);
    }
    return gltfLoader;
}

export async function registerCustomModel(id, arrayBuffer, { name = "My model", fit = {} } = {}) {
    const loader = getGltfLoader();
    const gltf = await loader.parseAsync(arrayBuffer, "");
    const root = gltf.scene || gltf.scenes?.[0];
    if (!root) throw new Error("The file has no 3D scene.");
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    if (box.isEmpty()) throw new Error("The model is empty.");
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const wrapper = new THREE.Group();
    root.position.sub(center);
    wrapper.add(root);
    // A head is about 2.1 units wide in animal space (radius ~1).
    wrapper.scale.setScalar(2.2 / Math.max(size.x, size.y * 0.85, size.z, 1e-6));
    let morphCount = 0;
    root.traverse((o) => {
        if (o.isMesh && o.morphTargetDictionary) morphCount += Object.keys(o.morphTargetDictionary).length;
        if (o.isMesh) o.frustumCulled = false;
    });
    customModels.set(id, { name, template: wrapper, fit: { ...DEFAULT_FIT, ...fit }, morphCount });
    return { id, name, morphCount };
}

export function unregisterCustomModel(id) {
    customModels.delete(id);
}

export function customModelInfo(id) {
    const m = customModels.get(id);
    return m ? { id, name: m.name, fit: { ...m.fit }, morphCount: m.morphCount } : null;
}

export function setCustomFit(id, fit) {
    const m = customModels.get(id);
    if (m) Object.assign(m.fit, fit);
}

function buildCustom(id) {
    const entry = customModels.get(id);
    if (!entry) return buildAnimal("ghost");
    const inst = skeletonClone(entry.template);
    const fitG = new THREE.Group();
    fitG.add(inst);
    const holder = new THREE.Group();
    holder.position.set(0, -0.6, -1.2);
    holder.scale.setScalar(11.3);
    holder.add(fitG);

    const targets = new Map(); // normalized name -> [[mesh, index]]
    inst.traverse((o) => {
        if (o.isMesh && o.morphTargetDictionary && o.morphTargetInfluences) {
            for (const [name, idx] of Object.entries(o.morphTargetDictionary)) {
                const n = normName(name);
                if (!targets.has(n)) targets.set(n, []);
                targets.get(n).push([o, idx]);
            }
        }
    });
    const applyFit = () => {
        const f = entry.fit;
        fitG.scale.setScalar(f.scale);
        fitG.position.set(0, f.y, f.z);
        fitG.rotation.y = f.ry;
    };
    applyFit();
    return {
        id,
        object: holder,
        custom: true,
        update(bs) {
            applyFit();
            for (const [name, value] of Object.entries(bs)) {
                const list = targets.get(normName(name));
                if (list) for (const [mesh, idx] of list) mesh.morphTargetInfluences[idx] = value;
            }
        },
        dispose() {
            // Geometry and materials are shared with the template; nothing to free per instance.
        },
    };
}

export function buildRig(id) {
    if (String(id).startsWith("custom:")) return buildCustom(id);
    return buildAnimal(id);
}

/* ------------------------------------------------------------------ thumbnails */

let thumb = null;

function thumbRenderer() {
    if (!thumb) {
        const canvas = document.createElement("canvas");
        const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.NeutralToneMapping;
        renderer.setClearColor(0x000000, 0);
        const scene = new THREE.Scene();
        const pmrem = new THREE.PMREMGenerator(renderer);
        scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        scene.environmentIntensity = 0.8;
        scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7a70, 1.3));
        const key = new THREE.DirectionalLight(0xffffff, 2.2);
        key.position.set(0.6, 1, 1.2);
        scene.add(key);
        const cam = new THREE.PerspectiveCamera(28, 1, 1, 1000);
        thumb = { renderer, scene, cam };
    }
    return thumb;
}

/* Renders a head facing the viewer and returns it as a PNG data URL. */
export function renderThumbnail(id, size = 160) {
    const { renderer, scene, cam } = thumbRenderer();
    renderer.setSize(size, size, false);
    const rig = buildRig(id);
    rig.update({ mouthSmileLeft: 0.35, mouthSmileRight: 0.35 }, 1 / 60, null, 0.6);
    const pivot = new THREE.Group();
    pivot.add(rig.object);
    pivot.rotation.y = -0.18;
    pivot.rotation.x = 0.05;
    scene.add(pivot);
    pivot.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(pivot);
    const c = box.getCenter(new THREE.Vector3());
    const s = box.getSize(new THREE.Vector3());
    const r = Math.max(s.x, s.y) / 2;
    const dist = (r / Math.tan(THREE.MathUtils.degToRad(cam.fov / 2))) * 1.08;
    cam.position.set(c.x, c.y, c.z + dist + s.z / 2);
    cam.lookAt(c);
    cam.near = dist * 0.2;
    cam.far = dist * 4;
    cam.updateProjectionMatrix();
    renderer.render(scene, cam);
    const url = renderer.domElement.toDataURL("image/png");
    scene.remove(pivot);
    rig.dispose();
    return url;
}

/* ------------------------------------------------------------------ stage */

let nextTrackId = 1;

export class MemojiStage {
    /*
        viewport: an element (position relative, overflow hidden) that the picture fills.
        onFaces(count): called when the number of faces changes.
    */
    constructor(viewport, { onFaces } = {}) {
        this.viewport = viewport;
        this.onFaces = onFaces || (() => { });
        this.inner = document.createElement("div");
        this.inner.className = "stage-inner";
        this.video = document.createElement("video");
        this.video.playsInline = true;
        this.video.muted = true;
        this.video.setAttribute("playsinline", "");
        this.video.setAttribute("muted", "");
        this.still = document.createElement("canvas");
        this.still.hidden = true;
        this.glCanvas = document.createElement("canvas");
        this.glCanvas.className = "stage-gl";
        for (const el of [this.video, this.still, this.glCanvas]) {
            el.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
        }
        this.inner.append(this.video, this.still, this.glCanvas);
        this.inner.style.cssText = "position:absolute;left:0;top:0;transform-origin:center";
        viewport.append(this.inner);

        const renderer = new THREE.WebGLRenderer({ canvas: this.glCanvas, alpha: true, antialias: true, preserveDrawingBuffer: true, powerPreference: "high-performance" });
        renderer.setPixelRatio(1);
        renderer.setClearColor(0x000000, 0);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.NeutralToneMapping;
        this.renderer = renderer;

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(63, 1, 1, 10000); // MediaPipe's projection
        this.pmrem = new THREE.PMREMGenerator(renderer);
        this.studioEnv = this.pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        this.scene.environment = this.studioEnv;
        this.scene.environmentIntensity = 0.85;

        this.hemi = new THREE.HemisphereLight(0xffffff, 0x8a7a70, 1.2);
        this.key = new THREE.DirectionalLight(0xffffff, 2.2);
        this.key.position.set(40, 60, 40);
        this.keyTarget = new THREE.Object3D();
        this.scene.add(this.keyTarget);
        this.key.target = this.keyTarget;
        this.fill = new THREE.DirectionalLight(0xffffff, 0.45);
        this.fill.position.set(-50, 10, 30);
        this.scene.add(this.hemi, this.key, this.fill);

        this.tracks = [];
        this.primary = null;
        this.spirit = "bunny";
        this.mirror = true;
        this.autoLight = true;
        this.mode = "idle"; // "live" | "still" | "idle"
        this.fit = "cover";
        this.facing = "user";
        this.stream = null;
        this.raf = 0;
        this.lastVideoTime = -1;
        this.frameNo = 0;
        this.lastEnvUpdate = 0;
        this.envTarget = null;
        this.light = this.defaultLight();
        this.lightGoal = this.defaultLight();
        this.sample = document.createElement("canvas");
        this.sampleCtx = this.sample.getContext("2d", { willReadFrequently: true });
        this.envCanvas = document.createElement("canvas");
        this.envCanvas.width = 64;
        this.envCanvas.height = 32;
        this.envSmall = document.createElement("canvas");
        this.envSmall.width = 16;
        this.envSmall.height = 8;
        this.lastFaceCount = -1;
        this.clock = new THREE.Clock();
        this.time = 0;

        this.resizeObserver = new ResizeObserver(() => this.layout());
        this.resizeObserver.observe(viewport);
    }

    defaultLight() {
        return {
            sky: new THREE.Color(0xffffff), ground: new THREE.Color(0x8a7a70), hemi: 1.2,
            keyDir: new THREE.Vector3(0.45, 0.75, 0.6).normalize(), keyColor: new THREE.Color(0xffffff), keyInt: 2.2,
            fillInt: 0.45, exposure: 1.0,
        };
    }

    /* ---------------------------------------------------------- source & layout */

    sourceSize() {
        if (this.mode === "live") return [this.video.videoWidth || 0, this.video.videoHeight || 0];
        if (this.mode === "still") return [this.still.width, this.still.height];
        return [0, 0];
    }

    source() {
        return this.mode === "still" ? this.still : this.video;
    }

    layout() {
        const [sw, sh] = this.sourceSize();
        const vw = this.viewport.clientWidth, vh = this.viewport.clientHeight;
        if (!sw || !sh || !vw || !vh) return;
        const scale = this.fit === "contain" ? Math.min(vw / sw, vh / sh) : Math.max(vw / sw, vh / sh);
        const iw = sw * scale, ih = sh * scale;
        this.inner.style.width = iw + "px";
        this.inner.style.height = ih + "px";
        this.inner.style.left = (vw - iw) / 2 + "px";
        this.inner.style.top = (vh - ih) / 2 + "px";
        this.inner.style.transform = this.mirror && this.mode === "live" ? "scaleX(-1)" : "none";
        if (this.glCanvas.width !== sw || this.glCanvas.height !== sh) {
            this.renderer.setSize(sw, sh, false);
        }
        this.camera.aspect = sw / sh;
        this.camera.updateProjectionMatrix();
        this.view = { scale, iw, ih, vw, vh, sw, sh };
    }

    // The part of the source picture that is visible in the viewport, in source pixels.
    visibleRect() {
        const v = this.view;
        if (!v) return null;
        if (this.fit === "contain") return { x: 0, y: 0, w: v.sw, h: v.sh };
        const w = Math.min(v.sw, v.vw / v.scale), h = Math.min(v.sh, v.vh / v.scale);
        return { x: (v.sw - w) / 2, y: (v.sh - h) / 2, w, h };
    }

    /* ---------------------------------------------------------- camera */

    async startCamera(facing = this.facing) {
        this.stopCamera();
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error("This browser cannot open the camera here. Camera access needs HTTPS (or localhost).");
        }
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        this.stream = stream;
        this.facing = facing;
        this.mirror = facing === "user";
        this.video.srcObject = stream;
        await new Promise((res) => {
            if (this.video.readyState >= 1) res();
            else this.video.onloadedmetadata = () => res();
        });
        await this.video.play().catch(() => { });
        this.mode = "live";
        this.fit = "cover";
        this.video.hidden = false;
        this.still.hidden = true;
        this.clearTracks();
        this.layout();
        this.startLoop();
        this.landmarker = await loadFaceLandmarker();
        await setRunningMode(this.landmarker, "VIDEO");
        this.lastVideoTime = -1;
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach((t) => t.stop());
            this.stream = null;
        }
        this.video.srcObject = null;
    }

    async flipCamera() {
        return this.startCamera(this.facing === "user" ? "environment" : "user");
    }

    /* ---------------------------------------------------------- still photos */

    /*
        Shows a photo (File, Blob or ImageBitmap source) and turns its faces into memoji.
        Resolves to the number of faces found.
    */
    async showPhoto(file) {
        this.stopCamera();
        const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
        const max = 1600;
        const k = Math.min(1, max / Math.max(bmp.width, bmp.height));
        this.still.width = Math.round(bmp.width * k);
        this.still.height = Math.round(bmp.height * k);
        this.still.getContext("2d").drawImage(bmp, 0, 0, this.still.width, this.still.height);
        bmp.close?.();
        this.mode = "still";
        this.fit = "contain";
        this.mirror = false;
        this.video.hidden = true;
        this.still.hidden = false;
        this.clearTracks();
        this.layout();
        this.startLoop();
        this.landmarker = await loadFaceLandmarker();
        await setRunningMode(this.landmarker, "IMAGE");
        const res = this.detectWithTiles(this.landmarker, this.still);
        this.applyResult(res, true);
        this.estimateLight(true);
        return this.tracks.length;
    }

    /*
        The face detector is made for faces near the camera, so small faces in a group photo are missed.
        After the whole picture, overlapping tiles are searched too. A tile is the same camera with a
        narrower view, so a face found in it is moved back into the whole picture: its rotation stays,
        its distance grows by the zoom factor and its position is projected again from where it is.
    */
    detectWithTiles(lm, canvas) {
        const W = canvas.width, H = canvas.height;
        const out = { faceLandmarks: [], facialTransformationMatrixes: [], faceBlendshapes: [] };
        const boxes = [];
        const add = (res, tx, ty, tw, th) => {
            (res?.faceLandmarks || []).forEach((marks, i) => {
                const data = res.facialTransformationMatrixes?.[i]?.data;
                if (!data) return;
                const full = marks.map((p) => ({ x: (p.x * tw + tx) / W, y: (p.y * th + ty) / H, z: p.z }));
                let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
                for (const p of full) {
                    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x);
                    y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
                }
                const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
                if (boxes.some((b) => cx > b[0] && cx < b[2] && cy > b[1] && cy < b[3])) return; // found already
                // Faces cut by the tile edge are left to the tile that holds them whole.
                if (tw < W && (x0 * W < tx + 2 || x1 * W > tx + tw - 2 || y0 * H < ty + 2 || y1 * H > ty + th - 2)) return;
                boxes.push([x0, y0, x1, y1]);
                let matrix = data;
                if (tw !== W || th !== H) {
                    const m = new THREE.Matrix4().fromArray(data);
                    const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
                    m.decompose(pos, quat, scl);
                    const tanH = Math.tan(THREE.MathUtils.degToRad(63 / 2));
                    const fc = th / 2 / tanH;
                    const u = tx + tw / 2 + fc * (pos.x / -pos.z);
                    const v = ty + th / 2 - fc * (pos.y / -pos.z);
                    const z = pos.z / (th / H);
                    const f = H / 2 / tanH;
                    const fixed = new THREE.Vector3(((u - W / 2) * -z) / f, (-(v - H / 2) * -z) / f, z);
                    matrix = new THREE.Matrix4().compose(fixed, quat, scl).toArray();
                }
                out.faceLandmarks.push(full);
                out.facialTransformationMatrixes.push({ data: matrix });
                out.faceBlendshapes.push(res.faceBlendshapes?.[i] || { categories: [] });
            });
        };
        add(lm.detect(canvas), 0, 0, W, H);
        if (out.faceLandmarks.length >= MAX_FACES) return out;
        const tile = document.createElement("canvas");
        for (const frac of [0.5, 0.34]) {
            const tw = Math.round(W * frac), th = Math.round(H * frac);
            if (Math.min(tw, th) < 160) continue;
            tile.width = tw;
            tile.height = th;
            const tctx = tile.getContext("2d");
            const steps = Math.round(1 / frac) * 2 - 1;
            for (let j = 0; j < steps; j++) {
                for (let i = 0; i < steps; i++) {
                    if (out.faceLandmarks.length >= MAX_FACES) return out;
                    const tx = Math.round(((W - tw) * i) / (steps - 1));
                    const ty = Math.round(((H - th) * j) / (steps - 1));
                    tctx.drawImage(canvas, tx, ty, tw, th, 0, 0, tw, th);
                    add(lm.detect(tile), tx, ty, tw, th);
                }
            }
        }
        return out;
    }

    /* ---------------------------------------------------------- loop */

    startLoop() {
        if (this.raf) return;
        this.clock.getDelta();
        const tick = (now) => {
            this.raf = requestAnimationFrame(tick);
            try {
                this.frame(now);
            } catch (err) {
                console.error(err);
            }
        };
        this.raf = requestAnimationFrame(tick);
    }

    stopLoop() {
        cancelAnimationFrame(this.raf);
        this.raf = 0;
    }

    frame(now) {
        const dt = Math.min(0.1, this.clock.getDelta() || 1 / 60);
        this.time += dt;
        this.frameNo++;
        if (this.mode === "live" && this.landmarker && this.landmarker.__mode === "VIDEO" && this.video.readyState >= 2) {
            if (this.video.videoWidth && (!this.view || this.view.sw !== this.video.videoWidth || this.view.sh !== this.video.videoHeight)) this.layout();
            if (this.video.currentTime !== this.lastVideoTime) {
                this.lastVideoTime = this.video.currentTime;
                const res = this.landmarker.detectForVideo(this.video, now);
                this.applyResult(res, false);
            }
            if (this.frameNo % 3 === 0) this.estimateLight(false);
        }
        this.dropLostTracks(now);
        this.updateRigs(dt);
        this.updateLights(this.mode === "still" ? 1 : 0.08);
        this.renderer.render(this.scene, this.camera);
    }

    renderNow() {
        this.updateRigs(1 / 60);
        this.renderer.render(this.scene, this.camera);
    }

    /* ---------------------------------------------------------- tracking */

    clearTracks() {
        for (const t of this.tracks) this.disposeTrack(t);
        this.tracks = [];
        this.primary = null;
    }

    disposeTrack(t) {
        if (t.rig) {
            t.anchor.remove(t.rig.object);
            t.rig.dispose();
        }
        this.scene.remove(t.anchor);
    }

    randomAnimal(exclude = []) {
        const pool = ANIMALS.map((a) => a.id).filter((id) => id !== this.spirit && !exclude.includes(id));
        const list = pool.length ? pool : ANIMALS.map((a) => a.id);
        return list[Math.floor(Math.random() * list.length)];
    }

    applyResult(res, instant) {
        const now = performance.now();
        const faces = [];
        (res?.faceLandmarks || []).forEach((lm, i) => {
            const mat = res.facialTransformationMatrixes?.[i]?.data;
            if (!mat) return;
            let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
            for (const p of lm) {
                if (p.x < x0) x0 = p.x;
                if (p.x > x1) x1 = p.x;
                if (p.y < y0) y0 = p.y;
                if (p.y > y1) y1 = p.y;
            }
            const bs = {};
            for (const c of res.faceBlendshapes?.[i]?.categories || []) bs[c.categoryName] = c.score;
            faces.push({ lm, mat, bs, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0, area: (x1 - x0) * (y1 - y0) });
        });

        // Match faces to existing tracks by nearest centre.
        const free = new Set(this.tracks);
        const pairs = [];
        for (const f of faces) {
            for (const t of this.tracks) {
                const d = Math.hypot(f.cx - t.cx, f.cy - t.cy);
                if (d < Math.max(f.w, t.w) * 0.9) pairs.push([d, f, t]);
            }
        }
        pairs.sort((a, b) => a[0] - b[0]);
        const matched = new Set();
        for (const [, f, t] of pairs) {
            if (matched.has(f) || !free.has(t)) continue;
            matched.add(f);
            free.delete(t);
            this.feedTrack(t, f, now, instant);
        }
        for (const f of faces) {
            if (matched.has(f)) continue;
            const t = this.newTrack();
            this.feedTrack(t, f, now, true);
        }

        // The primary (closest = biggest) face wears the spirit animal. A little hysteresis keeps it from flickering.
        const alive = this.tracks.filter((t) => t.lastSeen === now);
        if (alive.length) {
            const best = alive.reduce((a, b) => (b.area > a.area ? b : a));
            if (!this.primary || !this.tracks.includes(this.primary) || this.primary.lastSeen !== now || (best !== this.primary && best.area > this.primary.area * 1.3)) {
                this.primary = best;
            }
        }
        if (faces.length !== this.lastFaceCount) {
            this.lastFaceCount = faces.length;
            this.onFaces(faces.length);
        }
    }

    newTrack() {
        const anchor = new THREE.Group();
        anchor.matrixAutoUpdate = false;
        this.scene.add(anchor);
        const used = this.tracks.map((t) => t.animal);
        const t = {
            id: nextTrackId++, anchor, rig: null, rigId: null,
            animal: this.randomAnimal(used),
            pos: new THREE.Vector3(), quat: new THREE.Quaternion(), scl: new THREE.Vector3(1, 1, 1),
            tPos: new THREE.Vector3(), tQuat: new THREE.Quaternion(), tScl: new THREE.Vector3(1, 1, 1),
            bs: {}, tBs: {}, lm: null, cx: 0, cy: 0, w: 0, area: 0, lastSeen: 0,
            vel: new THREE.Vector3(), prevPos: null, accel: new THREE.Vector3(), angVel: new THREE.Vector3(), prevQuat: null,
            fresh: true,
        };
        this.tracks.push(t);
        return t;
    }

    feedTrack(t, f, now, instant) {
        const m = new THREE.Matrix4().fromArray(f.mat);
        m.decompose(t.tPos, t.tQuat, t.tScl);
        t.tBs = f.bs;
        t.lm = f.lm;
        t.cx = f.cx;
        t.cy = f.cy;
        t.w = f.w;
        t.area = f.area;
        t.lastSeen = now;
        if (instant || t.fresh) {
            t.pos.copy(t.tPos);
            t.quat.copy(t.tQuat);
            t.scl.copy(t.tScl);
            t.bs = { ...f.bs };
            t.fresh = false;
        }
    }

    dropLostTracks(now) {
        if (this.mode !== "live") return;
        const keep = [];
        for (const t of this.tracks) {
            if (now - t.lastSeen > 450) this.disposeTrack(t);
            else keep.push(t);
        }
        if (keep.length !== this.tracks.length) {
            this.tracks = keep;
            if (!keep.includes(this.primary)) this.primary = null;
        }
    }

    // Others get new random animals.
    reshuffle() {
        const used = [];
        for (const t of this.tracks) {
            if (t === this.primary) continue;
            t.animal = this.randomAnimal(used.concat(t.animal));
            used.push(t.animal);
        }
    }

    setSpirit(id) {
        this.spirit = id;
        for (const t of this.tracks) {
            if (t !== this.primary && t.animal === id) t.animal = this.randomAnimal([id]);
        }
    }

    animalsInPicture() {
        return this.tracks.map((t) => t.rigId).filter(Boolean);
    }

    updateRigs(dt) {
        const a = 1 - Math.exp(-dt * 20);
        const aBs = 1 - Math.exp(-dt * 26);
        for (const t of this.tracks) {
            const want = t === this.primary ? this.spirit : t.animal;
            if (t.rigId !== want) {
                if (t.rig) {
                    t.anchor.remove(t.rig.object);
                    t.rig.dispose();
                }
                t.rig = buildRig(want);
                t.rigId = want;
                t.anchor.add(t.rig.object);
            }
            const still = this.mode === "still";
            t.pos.lerp(t.tPos, still ? 1 : a);
            t.quat.slerp(t.tQuat, still ? 1 : a);
            t.scl.lerp(t.tScl, still ? 1 : a);
            for (const k in t.tBs) t.bs[k] = (t.bs[k] ?? 0) + (t.tBs[k] - (t.bs[k] ?? 0)) * (still ? 1 : aBs);
            t.anchor.matrix.compose(t.pos, t.quat, t.scl);
            t.anchor.matrixWorldNeedsUpdate = true;

            // Head motion in the head's own frame, for the springy ears.
            if (t.prevPos && dt > 0) {
                const v = t.pos.clone().sub(t.prevPos).divideScalar(dt);
                const acc = v.clone().sub(t.vel).divideScalar(dt);
                t.vel.copy(v);
                const inv = t.quat.clone().invert();
                t.accel.lerp(acc.applyQuaternion(inv).clampLength(0, 3000), 0.5);
                const dq = t.quat.clone().multiply(t.prevQuat.clone().invert());
                const ang = 2 * Math.acos(clamp(dq.w, -1, 1));
                const s = Math.sqrt(Math.max(1e-8, 1 - dq.w * dq.w));
                const axis = new THREE.Vector3(dq.x / s, dq.y / s, dq.z / s);
                const w = ang > 1e-4 ? axis.multiplyScalar((dq.w < 0 ? -1 : 1) * ang / dt) : new THREE.Vector3();
                t.angVel.lerp(w.applyQuaternion(inv).clampLength(0, 20), 0.5);
            }
            t.prevPos = t.pos.clone();
            t.prevQuat = t.quat.clone();
            t.rig.update(t.bs, dt, { accel: t.accel, angVel: t.angVel }, this.time + t.id);
        }
    }

    /* ---------------------------------------------------------- light estimation */

    estimateLight(immediate) {
        if (!this.autoLight) {
            this.lightGoal = this.defaultLight();
            this.setEnv(this.studioEnv);
            return;
        }
        const src = this.source();
        const [sw, sh] = this.sourceSize();
        if (!sw || !sh) return;
        const W = 96, H = Math.max(8, Math.round((96 * sh) / sw));
        if (this.sample.width !== W || this.sample.height !== H) {
            this.sample.width = W;
            this.sample.height = H;
        }
        const ctx = this.sampleCtx;
        ctx.drawImage(src, 0, 0, W, H);
        const px = ctx.getImageData(0, 0, W, H).data;
        const lin = (c) => Math.pow(c / 255, 2.2);
        const acc = (y0, y1) => {
            let r = 0, g = 0, b = 0, n = 0;
            for (let y = y0; y < y1; y++) {
                for (let x = 0; x < W; x += 2) {
                    const i = (y * W + x) * 4;
                    r += lin(px[i]); g += lin(px[i + 1]); b += lin(px[i + 2]); n++;
                }
            }
            return [r / n, g / n, b / n];
        };
        const all = acc(0, H), top = acc(0, Math.floor(H / 3)), bottom = acc(Math.floor((2 * H) / 3), H);
        const luma = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        const tint = (c, keep) => {
            const m = Math.max(c[0], c[1], c[2], 1e-4);
            const col = new THREE.Color(c[0] / m, c[1] / m, c[2] / m);
            return new THREE.Color(1, 1, 1).lerp(col, keep);
        };

        const goal = this.defaultLight();
        const L = luma(all);
        goal.sky = tint(top, 0.35);
        goal.ground = tint(bottom, 0.45).multiplyScalar(0.7);
        goal.hemi = 0.55 + clamp(L * 4, 0, 1.5);
        goal.keyColor = tint(all, 0.25);

        // The shading on the main face tells where the light comes from.
        const t = this.primary && this.tracks.includes(this.primary) ? this.primary : null;
        let faceL = L;
        if (t?.lm) {
            const patch = (idx) => {
                const p = t.lm[idx];
                if (!p) return 0;
                const cx = Math.round(p.x * W), cy = Math.round(p.y * H);
                let s = 0, n = 0;
                for (let y = cy - 1; y <= cy + 1; y++) {
                    for (let x = cx - 1; x <= cx + 1; x++) {
                        if (x < 0 || y < 0 || x >= W || y >= H) continue;
                        const i = (y * W + x) * 4;
                        s += luma([lin(px[i]), lin(px[i + 1]), lin(px[i + 2])]);
                        n++;
                    }
                }
                return n ? s / n : 0;
            };
            const left = patch(280), right = patch(50), fore = patch(151), chin = patch(199), nose = patch(4);
            faceL = (left + right + fore + chin + nose) / 5;
            const lx = clamp(((left - right) / (left + right + 1e-4)) * 3.2, -1.6, 1.6);
            const ly = clamp(((fore - chin) / (fore + chin + 1e-4)) * 2.5 + 0.45, -0.8, 1.6);
            const local = new THREE.Vector3(lx, ly, 1).normalize();
            goal.keyDir = local.applyQuaternion(t.quat).normalize();
            const contrast = clamp(Math.abs(lx) * 0.5 + Math.abs(ly - 0.45) * 0.5, 0, 1);
            goal.keyInt = 1.4 + contrast * 2.4;
            goal.fillInt = 0.55 - contrast * 0.35;
        }
        // Exposure follows the brightness of the face (or the frame): a dim room gives a dimmer memoji.
        goal.exposure = clamp(0.5 + Math.sqrt(faceL) * 1.25, 0.55, 1.3);
        this.lightGoal = goal;
        if (immediate) this.updateLights(1);

        const nowMs = performance.now();
        if (immediate || nowMs - this.lastEnvUpdate > 1200) {
            this.lastEnvUpdate = nowMs;
            this.updateEnvFromFrame(src);
        }
    }

    // Blurred copy of the frame, mirrored around, as an equirectangular environment.
    updateEnvFromFrame(src) {
        const s = this.envSmall.getContext("2d");
        s.drawImage(src, 0, 0, 8, 8);
        s.save();
        s.translate(16, 0);
        s.scale(-1, 1);
        s.drawImage(src, 0, 0, 8, 8);
        s.restore();
        const e = this.envCanvas.getContext("2d");
        e.imageSmoothingEnabled = true;
        e.drawImage(this.envSmall, 0, 0, 64, 32);
        const tex = new THREE.CanvasTexture(this.envCanvas);
        tex.mapping = THREE.EquirectangularReflectionMapping;
        tex.colorSpace = THREE.SRGBColorSpace;
        const rt = this.pmrem.fromEquirectangular(tex);
        tex.dispose();
        if (this.envTarget) this.envTarget.dispose();
        this.envTarget = rt;
        this.setEnv(rt.texture);
    }

    setEnv(tex) {
        this.scene.environment = tex;
    }

    setAutoLight(on) {
        this.autoLight = on;
        if (!on) {
            this.lightGoal = this.defaultLight();
            this.setEnv(this.studioEnv);
        } else {
            this.lastEnvUpdate = 0;
            this.estimateLight(this.mode === "still");
        }
    }

    updateLights(k) {
        const l = this.light, g = this.lightGoal;
        l.sky.lerp(g.sky, k);
        l.ground.lerp(g.ground, k);
        l.hemi += (g.hemi - l.hemi) * k;
        l.keyDir.lerp(g.keyDir, k).normalize();
        l.keyColor.lerp(g.keyColor, k);
        l.keyInt += (g.keyInt - l.keyInt) * k;
        l.fillInt += (g.fillInt - l.fillInt) * k;
        l.exposure += (g.exposure - l.exposure) * k;

        this.hemi.color.copy(l.sky);
        this.hemi.groundColor.copy(l.ground);
        this.hemi.intensity = l.hemi;
        const target = this.primary && this.tracks.includes(this.primary) ? this.primary.pos : new THREE.Vector3(0, 0, -50);
        this.keyTarget.position.copy(target);
        this.key.position.copy(target).addScaledVector(l.keyDir, 100);
        this.key.color.copy(l.keyColor);
        this.key.intensity = l.keyInt;
        this.fill.position.copy(target).add(new THREE.Vector3(-l.keyDir.x * 100, 20, 80));
        this.fill.target = this.keyTarget;
        this.fill.intensity = Math.max(0.1, l.fillInt);
        this.renderer.toneMappingExposure = l.exposure;
    }

    /* ---------------------------------------------------------- capture */

    /*
        The picture as it is shown (cropped to the viewport, mirrored like the preview),
        with the memoji drawn on it. Resolves to { blob, width, height, animals }.
    */
    async capture({ maxSide = 1440, quality = 0.9 } = {}) {
        this.renderNow();
        const r = this.visibleRect();
        if (!r) throw new Error("Nothing to capture yet.");
        const k = Math.min(1, maxSide / Math.max(r.w, r.h));
        const out = document.createElement("canvas");
        out.width = Math.round(r.w * k);
        out.height = Math.round(r.h * k);
        const ctx = out.getContext("2d");
        if (this.mirror && this.mode === "live") {
            ctx.translate(out.width, 0);
            ctx.scale(-1, 1);
        }
        ctx.drawImage(this.source(), r.x, r.y, r.w, r.h, 0, 0, out.width, out.height);
        const gx = this.glCanvas.width / this.view.sw, gy = this.glCanvas.height / this.view.sh;
        ctx.drawImage(this.glCanvas, r.x * gx, r.y * gy, r.w * gx, r.h * gy, 0, 0, out.width, out.height);
        const blob = await new Promise((res, rej) => out.toBlob((b) => (b ? res(b) : rej(new Error("Could not encode the photo."))), "image/jpeg", quality));
        return { blob, width: out.width, height: out.height, animals: this.animalsInPicture() };
    }

    // Freeze the current camera frame as a still, so the user can review and re-roll it.
    freeze() {
        if (this.mode !== "live") return;
        const [sw, sh] = this.sourceSize();
        this.still.width = sw;
        this.still.height = sh;
        const ctx = this.still.getContext("2d");
        ctx.drawImage(this.video, 0, 0, sw, sh);
    }

    destroy() {
        this.stopLoop();
        this.stopCamera();
        this.clearTracks();
        this.resizeObserver.disconnect();
        this.renderer.dispose();
    }
}
