/*
    sw.js: shows push notifications from Skapi.

    Two kinds arrive:
    - From skapi.postRealtime() with a notification (likes, comments, follows): { title, body }.
    - From records posted with notify_subscribers (a user you follow posted):
      { title, body, type: "record" | "reference", record_id, table, user_id, reference? }.
    Clicking one opens memoji at the post, or focuses an open memoji tab and sends it there.
*/
self.addEventListener("push", (event) => {
    let data = {};
    try {
        data = event.data ? event.data.json() : {};
    } catch (err) {
        data = { body: event.data ? event.data.text() : "" };
    }
    const post = data.type === "reference" ? data.reference : data.type === "record" ? data.record_id : null;
    event.waitUntil(
        self.registration.showNotification(data.title || "memoji", {
            body: data.body || "Something new on memoji",
            icon: "icon.svg",
            badge: "icon.svg",
            data: { open: post ? "#post=" + post : "#activity" },
        })
    );
});

self.addEventListener("notificationclick", (event) => {
    event.notification.close();
    const open = event.notification.data?.open || "";
    const url = new URL("index.html" + (open === "#activity" ? "" : open), self.registration.scope).href;
    event.waitUntil(
        clients.matchAll({ type: "window", includeUncontrolled: true }).then((list) => {
            for (const c of list) {
                if (c.url.startsWith(self.registration.scope)) {
                    c.postMessage({ open: open === "#activity" ? "" : open.slice(1) });
                    return c.focus();
                }
            }
            return clients.openWindow(url);
        })
    );
});
