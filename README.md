# memoji

An Instagram-like photo app where every face turns into a cute 3D animal. Take a photo (or pick one from your gallery), and the face closest to the camera becomes your **spirit animal** while everyone else in the picture gets a random one. Post it, like, comment, follow people and get notified.

It is the photo gallery of the Skapi HTML template (`templates-skapi/database.html`) with a memoji camera in front of it: plain HTML and JavaScript, no build step, and all backend features on [Skapi](https://www.skapi.com) (project `5g3TffXBwAE3eFEcbG-7Zd6TgdLUaFxseKMT1qveY`).

## Spirit animals

Bunny, bear, hamster, beaver, lion, tiger, turtle, shark, gorilla, alien and a little white sheet ghost. You can also upload your own 3D model (see below).

The heads are made procedurally with three.js, with nothing downloaded: deformed high-resolution spheres, muzzles, cheeks and brow ridges, painted fur, stripes and blush, glossy eyes with lids, brows, mouths with tongues and teeth. Every head follows the 52 ARKit-style blendshapes MediaPipe reports, so they blink, look around, raise their brows, smile, puff their cheeks and open their mouths with you. Ears, antennae, fins, manes and tufts sit on springs and wobble when you move.

## How it works

| Part | What it does |
| --- | --- |
| `js/memoji-engine.js` | Camera, face tracking and rendering. MediaPipe Face Landmarker (WebAssembly; GPU delegate over WebGL with a CPU fallback) finds up to 6 faces with landmarks, blendshapes and a 4×4 pose matrix. three.js renders a head under each pose with MediaPipe's own projection (vertical FOV 63°), so the head lands on the face. |
| `js/animals.js` | The 11 procedural animals and their rigs. |
| `js/app.js` | The social app: feed, posting, likes, comments, follows, notifications, spirit picker, custom models. |
| `index.html`, `app.css` | The app page and its design (light and dark). |
| `login.html`, `signup.html`, `signup-success.html`, `forgot-password.html`, `reset-password.html` | Accounts, from the template, restyled. Guests can try it with **Try it as a guest**. |
| `service.js` | Creates `skapi`, keeps the login gate (same as the template's). |
| `sw.js` | Service worker that shows push notifications. |

Details worth knowing:

- **Closest face.** The biggest face is the closest. It keeps the spirit animal until another face is clearly (30%) bigger, so it does not flicker between two people.
- **Lighting.** With the sun button on, the memoji adapt to the camera. The frame's brightness and colour set the ambient light and exposure. The shading on the main face (left cheek against right, forehead against chin) sets the direction of the key light. A blurred copy of the frame is the environment map that the eyes and noses reflect. Turn it off for studio lighting.
- **Group photos.** For a still photo, the picture is searched again in overlapping tiles, so small faces in the back are found too. A face found in a tile is projected back into the whole picture.
- **Head placement.** The head centre and size were measured from MediaPipe's canonical face model (x ±7.7 cm, y −9.4…8.3 cm, z −2.4…7.6 cm). The head is centred at (0, −0.6, −1.2) cm with a radius of about 11.3 cm, so it covers the whole face.
- **What you see is what you post.** The camera shows a 4:5 crop; the photo is that crop, mirrored like the preview for the front camera. Gallery photos keep their shape.

## Data on Skapi

| Table | Access group | What |
| --- | --- | --- |
| `posts` | `authorized`, or `private` for private posts | `data: { description, spirit, animals }`, file `pic` (JPEG), tags (yours plus the spirit animal). Posted with `upload_to_feed` and `notify_subscribers` (not private ones) and `prevent_multiple_referencing`. |
| `likes` | `public` | A record that references the post. The post's reference count is its number of likes. |
| `comments` | `authorized` | `data: { comment, spirit }`, index `comment.<post record_id>`. "Most commented" orders those index names by total. |
| `spirit_models` | `private` | Your uploaded 3D models: `data: { name, fit }`, file `model`. Only you can read them. |

The views follow the URL hash, as in the template: `#` (recent), `#mostliked`, `#mostcomment`, `#feed` (people you follow), `#myposts`, `#myprivate`, `#tag=<tag>`, plus `#user=<user_id>` and `#post=<record_id>` (used by notifications).

**Your spirit animal** is kept in your profile's `misc` field (only you can read it), so it follows you to other devices. A guest cannot edit a profile, so a guest's choice stays in the browser.

## Following and notifications

- **Follow** is `skapi.subscribe({ user_id, get_feed: true, get_notified: true })`. Their new posts show up under **Following**, and with notifications on, each new post sends you a push.
- **Likes, comments and follows** reach the other user at once with `skapi.postRealtime()` while they have memoji open (a toast, plus the Activity list and badge). When they are not connected, the same call sends them a push notification instead.
- **Push notifications** are turned on per device in **Activity**: `vapidPublicKey()`, the browser's push subscription with `sw.js`, then `subscribeNotification()`.
- The Activity list is kept in the browser (localStorage), per account.

## Your own 3D models

In your profile (or the **+** at the end of the animal strip in the camera), upload a **.glb**, a **.gltf** with embedded data, or a **.vrm**, up to 25 MB. Draco- and Meshopt-compressed meshes and KTX2 textures are supported.

- The model is centred and scaled to the size of a head. Then use the **Fit** sliders in the camera (size, up/down, forward/back, turn); the fit is saved with the model.
- To follow your expressions, give the model **morph targets (shape keys) named like ARKit blendshapes**: `eyeBlinkLeft`, `eyeBlinkRight`, `jawOpen`, `mouthSmileLeft`, `browInnerUp`, `cheekPuff` and the rest of the 52. Names are matched loosely: case, `_` and spaces don't matter, and `Left`/`_L` are the same. A model without morph targets still follows your head.
- Models are private records: other people see your posts, never your model files.

## Running it

Camera access and push notifications need **HTTPS** (or `localhost`).

```sh
npm install
npm run dev
```

Then open `http://localhost:4000/index.html`. Any static server works. To try it on a phone, host it on HTTPS, for example with Skapi's website hosting.

To run it against another Skapi project, open any page with `?pid=<Project ID>`, or change `DEFAULT_PROJECT_ID` in `service.js`.

## Libraries

All from a CDN, pinned to exact versions in `index.html`:

- [three.js](https://threejs.org) 0.170.0 (WebGL rendering, glTF loading)
- [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js) 0.10.21 with the `face_landmarker` model (WebAssembly)
- [skapi-js](https://www.npmjs.com/package/skapi-js) 2.4.2

## Limits

- The live face detector is made for faces within about 2 m of the camera. Faces farther away are found in still photos (tiled search), but not always in the live view.
- Building a new animal the first time takes a fraction of a second. The camera builds them all in the background as soon as it opens.
- Hands and hair in front of a face are not masked; the head is drawn over them.
