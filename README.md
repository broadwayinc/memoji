# memoji

An Instagram-like photo app where every face turns into a cute 3D animal. Take a photo (or pick one from your gallery), and the face closest to the camera becomes your **spirit animal** while everyone else in the picture gets a random one. Post it, like, comment, follow people and get notified.

It is the photo gallery of the Skapi HTML template (`templates-skapi/database.html`) with a memoji camera in front of it: plain HTML and JavaScript, no build step, and all backend features on [Skapi](https://www.skapi.com) (project `5g3TffXBwAE3eFEcbG-7Zd6TgdLUaFxseKMT1qveY`).

## Spirit animals

Bunny, bear, hamster, beaver, lion, tiger, koala, monkey, gorilla, alien and a little white sheet ghost. You can also upload your own 3D model (see below).

The heads are made procedurally, with nothing downloaded, in the style of soft vinyl / clay toy icons: simple pastel colour blocks and detail that comes from the sculpted shape, not from textures or polygon count.

- **Sculpted like clay** (`js/heads-sdf.js`): each head is a few soft shapes blended together: a wide bean-shaped head with full cheeks, plus whichever of these the animal has:
  - a muzzle that bulges out;
  - cheek pouches or cheek tufts;
  - a scalloped mane;
  - a brow ridge and crest;
  - a monkey's raised heart-shaped face mask and hair tuft;
  - cupped ears with their inside modelled, and the koala's big fluffy ears with a scalloped rim.

  The blend is meshed with surface nets into about 40k triangles, with perfectly smooth normals. It runs in a Web Worker (`js/heads-worker.js`), so the camera never stutters.
- **Pastel colour blocks**: muzzle, inner ears, stripes, blush, face mask. They are vertex colours whose edges follow the sculpt.
- **Glossy bead eyes** that slide to **follow your gaze** and **blink** with you. Gaze is read from where your iris sits between the eye corners; blinks from how open each eye is, against your own open eye.
- **Expressions**: **smile** (the eyes close into happy ^ ^ arcs), **laugh**, **sad** (lids and brows tilt, mouth corners go down, ears droop), **angry** (lids and brows slant into a V), **surprised** (bigger eyes, brows up, round mouth), **kiss**, wink, cheek puff and nose scrunch. Brows only appear while you make an expression, so a relaxed face stays clean.
- **Mouths** with tongues and teeth (buck teeth, fangs, rows of teeth), plus whiskers and nostrils.
- **Springy parts**: ears and alien antennae wobble when you move.

MediaPipe's raw blendshapes are small and mixed: a big smile also lowers the brows, and `browInnerUp` rises both when sad and when surprised. `readExpression()` in `js/animals.js` turns them into clean expressions, and a slow per-person baseline makes a resting face read as neutral.

## How it works

| Part | What it does |
| --- | --- |
| `js/memoji-engine.js` | Camera, face tracking and rendering. MediaPipe Face Landmarker (WebAssembly; GPU delegate over WebGL with a CPU fallback) finds up to 6 faces with landmarks, blendshapes and a 4×4 pose matrix. three.js renders a head under each pose with MediaPipe's own projection (vertical FOV 63°), so the head lands on the face. |
| `js/heads-sdf.js` | The 11 sculpted heads (signed distance fields) and the mesher; pure math. |
| `js/heads-worker.js` | Builds the head meshes in a Web Worker. |
| `js/animals.js` | The rigs: eyes, brows, mouth, ears, expressions. |
| `js/app.js` | The social app: feed, posting, likes, comments, follows, notifications, spirit picker, custom models. |
| `index.html`, `app.css` | The app page and its design (light and dark). |
| `login.html`, `signup.html`, `signup-success.html`, `forgot-password.html`, `reset-password.html` | Accounts, from the template, restyled. Guests can try it with **Try it as a guest**. |
| `service.js` | Creates `skapi`, keeps the login gate (same as the template's). |
| `sw.js` | Service worker that shows push notifications. |

Details worth knowing:

- **Closest face.** The biggest face is the closest. It keeps the spirit animal until another face is clearly (30%) bigger, so it does not flicker between two people.
- **Lighting.** With the sun button on, the memoji adapt to the camera. The frame's brightness and colour set the ambient light and exposure. The shading on the main face (left cheek against right, forehead against chin) sets the direction of the key light. A blurred copy of the frame is the environment map that the eyes and noses reflect. Turn it off for studio lighting.
- **Group photos.** For a still photo, the picture is searched again in overlapping tiles, so small faces in the back are found too. A face found in a tile is projected back into the whole picture.
- **Head placement.** The animal replaces the whole head, hair included. MediaPipe's canonical face has its eyes at y ≈ 2.7 cm, the chin at −9.4 and a width of ±7.7, and hair reaches about 13 to 15 cm above the eyes. The animal's eyes are at the middle of its head, and the head has a radius of about 15.6 cm (centred at eye height, 4 cm behind the face), so it reaches about 15 cm above and below your eyes. Then, every frame, the head slides along the camera ray until its eyes sit exactly in front of your irises. On test photos the head covers all 478 face landmarks and the hair, and its eye midpoint lands on the person's.
- **Mirror.** The selfie camera shows the picture like a mirror, and the photo is saved the same way. The mirror button in the camera turns it off to see yourself as others do. The back camera is never mirrored.
- **Smooth, not laggy.** Head tracking uses adaptive smoothing: steady when you hold still, quick when you move.
- **What you see is what you post.** The camera shows a 4:5 crop, and the photo is that crop. Gallery photos keep their shape.

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

In your profile (or the **+** at the end of the animal strip in the camera), upload a **.glb**, a **.gltf** with embedded data, or a **.vrm**, up to 40 MB. Draco- and Meshopt-compressed meshes and KTX2 textures are supported.

- **A head model** is centred and scaled to the size of a head.
- **A full avatar** has neck and head bones: every VRM / VRoid model, Mixamo-style rigs, and so on. It is cut at the neck, so only its head and hair replace yours. Its size comes from the vertices that move with the head bone, so arms in a T-pose don't count.
- **Expressions** come from any of these:
  - the VRM's own expression presets (happy, angry, sad, surprised, blink, blinkLeft/Right, aa, ih, ou, oh), VRM 1.0 and 0.x;
  - morph targets named like **ARKit blendshapes** (`eyeBlinkLeft`, `jawOpen`, `mouthSmileLeft`, `browInnerUp`… all 52);
  - VRoid names (`Fcl_EYE_Close_L`, `Fcl_MTH_A`, `Fcl_BRW_Angry`…);
  - VRChat visemes (`vrc.v_aa`, `vrc.blink_left`).

  Names are matched loosely: prefixes before a dot, case, `_` and spaces don't matter, and `Left`/`_L` are the same.
- **Eye bones** (VRM `leftEye` / `rightEye`, VRoid `J_Adj_L_FaceEye`) follow your gaze.
- **Fit sliders** in the camera (size, up/down, forward/back, turn) fine-tune the fit, and it is saved with the model.
- Models are private records: other people see your posts, never your model files.

Tested with three.js's `facecap.glb` (52 ARKit shapes) and pixiv's `VRM1_Constraint_Twist_Sample.vrm`: cut at the neck, with smile, blink, mouth, angry, sad and gaze all working.

### Where to get free models

| Where | What | Notes |
| --- | --- | --- |
| [VRoid Hub](https://hub.vroid.com/en/models) | Thousands of anime avatars as `.vrm`, full expressions | Free account; download only models whose author allows it (shown on each model page). |
| [VRoid Studio](https://vroid.com/en/studio) | Free app to make your own avatar | Export as VRM; it comes with blink, mouth and emotion expressions. |
| [Sketchfab: ARKit faces](https://sketchfab.com/search?features=downloadable&q=arkit+blendshapes&type=models) | Heads with the 52 ARKit blendshapes | Free account; choose a free/CC model and download **glTF** or **GLB**. |
| [Sketchfab: animal heads](https://sketchfab.com/search?features=downloadable&q=animal+head&type=models) | Static animal heads | They follow your head; no expressions unless they have morph targets. |
| [Quaternius](https://quaternius.com) | CC0 (public domain) low-poly packs, glTF | Characters and animals; full bodies are cut at the neck when they have neck/head bones. |
| [facecap.glb](https://github.com/mrdoob/three.js/raw/r170/examples/models/gltf/facecap.glb) | three.js example face with all 52 ARKit shapes | Good for a first test. |
| [VRM sample avatar](https://github.com/pixiv/three-vrm/raw/dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm) | pixiv's VRM 1.0 sample | Good for a first test of a full avatar. |

Always check each model's license before posting with it.

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
- Sculpting a head takes a fraction of a second. It happens in a Web Worker, for every animal, as soon as the app or the camera opens.
- Hands and hair in front of a face are not masked; the head is drawn over them.
