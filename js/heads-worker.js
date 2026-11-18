/*
    heads-worker.js: builds the head meshes off the main thread, so the camera never stutters while a
    new animal is being sculpted. Receives { id, part }, answers with the mesh's typed arrays.
*/
import { meshPart } from "./heads-sdf.js";

self.onmessage = (event) => {
    const { id, part, key } = event.data;
    try {
        const data = meshPart(id, part);
        if (!data) {
            self.postMessage({ key, data: null });
            return;
        }
        const transfer = [data.position.buffer, data.normal.buffer, data.color.buffer, data.index.buffer, ...Object.values(data.morphs).map((m) => m.buffer)];
        self.postMessage({ key, data }, transfer);
    } catch (err) {
        self.postMessage({ key, error: String(err?.stack || err) });
    }
};
