/*
    service.js is loaded by every page of memoji.
    It creates the `skapi` object the pages use and takes care of two things every page needs:
    the project id, and sending visitors to the login page when a page needs a logged in user.
    (Same structure as service.js of the Skapi HTML template.)
*/

/*
    Project ID

    memoji runs on the Skapi project below. Opening a page with ?pid=<Project ID> uses another project
    instead; the id is then kept in sessionStorage so the other pages use it too.
*/
const DEFAULT_PROJECT_ID = "5g3TffXBwAE3eFEcbG-7Zd6TgdLUaFxseKMT1qveY";
const PROJECT_ID_FROM_URL = new URLSearchParams(location.search).get("pid");
if (PROJECT_ID_FROM_URL) {
    sessionStorage.setItem("skapi:project_id", PROJECT_ID_FROM_URL);
}
const PROJECT_ID = PROJECT_ID_FROM_URL || sessionStorage.getItem("skapi:project_id") || DEFAULT_PROJECT_ID;

/*
    pageUrl(page) returns the full address of another page of the app.
    When another project than the default is used, it is added to the address as ?pid=,
    so a page opened in a new tab still knows its project.
*/
function pageUrl(page) {
    let url = new URL(page, location.href);
    if (PROJECT_ID !== DEFAULT_PROJECT_ID) {
        url.searchParams.set("pid", PROJECT_ID);
    }
    return url.href;
}

function isAppPage(href) {
    return /^[a-z-]+\.html([?#].*)?$/.test(href || "");
}

/*
    Enter pressed while a phone keyboard is still composing a word (an IME, for Korean or Japanese for example)
    confirms the word; it must not submit the form with half of it.
*/
document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.isComposing || event.keyCode === 229) && event.target.matches("input, textarea")) {
        event.preventDefault();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    for (let a of document.querySelectorAll("a[href]")) {
        if (isAppPage(a.getAttribute("href"))) {
            a.href = pageUrl(a.getAttribute("href"));
        }
    }
    for (let form of document.querySelectorAll("form[action]")) {
        if (isAppPage(form.getAttribute("action"))) {
            form.action = pageUrl(form.getAttribute("action"));
        }
    }
});

/*
    Pages that need a logged in user. A visitor who opens one of them while logged out is sent to
    login.html, and login.html brings them back after they log in.
*/
const LOGIN_REQUIRED = ["index.html"];

function currentPage() {
    return location.pathname.split("/").pop() || "index.html";
}

function goToLogin() {
    let to = currentPage() + location.hash;
    location.replace(pageUrl("login.html?to=" + encodeURIComponent(to)));
}

// A page that logs the user out on purpose sets this first, so the user is not bounced while it finishes.
let leavingOnPurpose = false;

/*
    - autoLogin: a user who logged in before is logged in again automatically.
    - eventListener.onLogin: called on load with the restored profile or null, then on every login and logout.
*/
let options = {
    autoLogin: true,
    eventListener: {
        onLogin: (user) => {
            if (!user && !leavingOnPurpose && LOGIN_REQUIRED.includes(currentPage())) {
                goToLogin();
            }
        },
    },
};

let skapi;
try {
    skapi = new Skapi(PROJECT_ID, options);
} catch (err) {
    // A project id that is not valid (a typo in ?pid=) makes the constructor throw. Forget it.
    sessionStorage.removeItem("skapi:project_id");
    alert("The project id is not valid: " + err.message);
    throw err;
}

/*
    userReady resolves once the automatic login has been tried:
    to the user's profile, or to null when nobody is logged in.
*/
const userReady = skapi.getProfile().then((user) => {
    if (!user && LOGIN_REQUIRED.includes(currentPage())) {
        goToLogin();
    }
    return user;
}).catch((err) => {
    console.error(err);
    return null;
});
