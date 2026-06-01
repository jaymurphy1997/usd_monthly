# Collaborative Comments — Setup Guide

The monthly report supports shared, threaded-by-section comments with **@mentions**, backed by
**Cloud Firestore** and **Google sign-in**. If no Firebase config is supplied, comments still work but
fall back to **browser localStorage** (private to each viewer, not shared).

## How it behaves

| Mode | When | Storage | Auth | Sharing |
|------|------|---------|------|---------|
| **Firestore** | `--firebase-config` passed at generation time **and** the report is served from an authorized domain | Cloud Firestore | Google sign-in | Real-time, multi-user |
| **localStorage** | no config, or page opened as a local `file://` | the viewer's browser | none | not shared |

> **Important:** Google sign-in popups do **not** work from `file://`. To use shared comments the HTML must
> be served over **https** from a domain listed in Firebase Auth → *Authorized domains* (Firebase Hosting is
> the simplest option). Opened locally, the report silently uses the localStorage fallback.

## One-time Firebase setup

1. Create a Firebase project at <https://console.firebase.google.com> (or reuse an existing one).
2. **Build → Firestore Database → Create database** (production mode, pick a region).
3. **Build → Authentication → Get started → Sign-in method → Google → Enable.**
4. **Project settings → General → Your apps → Web app (`</>`)** → register an app and copy the
   `firebaseConfig` values into a JSON file shaped like [`firebase_config.example.json`](firebase_config.example.json).
5. **Authentication → Settings → Authorized domains** → add the domain you'll serve the report from
   (e.g. `your-project.web.app`, or your custom host).

## Firestore security rules

Comments live under `report_comments/{reportId}/comments/{commentId}`. Anyone may read; only signed-in users
may create, and only the author may edit/delete their own. Paste into **Firestore → Rules**:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /report_comments/{reportId}/comments/{commentId} {
      allow read: if true;
      allow create: if request.auth != null
                    && request.resource.data.uid == request.auth.uid;
      allow update, delete: if request.auth != null
                    && resource.data.uid == request.auth.uid;
    }
  }
}
```

(To restrict reading to signed-in users only, change `allow read: if true;` to `allow read: if request.auth != null;`.)

## Generating a report with comments enabled

```powershell
& ".venv\Scripts\python.exe" generate_report.py --year 2026 --month 4 `
    --project gtm-nfl5sm4-yjezm --dataset analytics_255022204 `
    --firebase-config firebase_config.json --force
```

The config is embedded into the generated HTML. Omit `--firebase-config` for the localStorage-only build.

## Serving the report (for shared comments)

Any static host on an authorized domain works. Firebase Hosting example:

```powershell
npm install -g firebase-tools
firebase login
firebase init hosting        # set public dir to the folder containing the HTML
firebase deploy --only hosting
```

Then open the hosted URL, click **💬 Comments**, sign in with Google, and post. Comments appear live for all
viewers and per program section (use the in-section **💬 Add comment** button to attach to a program).

## Data model

`report_comments/{reportId}/comments/{commentId}`:

| field | type | notes |
|-------|------|-------|
| `text` | string | comment body |
| `anchor` | string | `general` or `prog-<id>` (which section it's attached to) |
| `who` | string | author display name |
| `email` | string | author email |
| `uid` | string | author Firebase UID (enforced by rules) |
| `mentions` | string[] | `@tokens` parsed from the text |
| `resolved` | bool | resolve/reopen toggle |
| `createdAt` | serverTimestamp | server time |

`reportId` is `usd_online_<year>_<month>` (e.g. `usd_online_2026_04`), so each month's report has its own
comment thread.

## Not included (follow-ups)

- **Email/Slack notifications for @mentions.** Mentions are parsed and stored, but delivering a notification
  requires a Cloud Function (Firestore `onCreate` trigger) — out of scope here. The `mentions[]` array is
  written so a function can be added later without changing the report.
- **A shared user directory / autocomplete for mentions.** Currently you type `@name` or `@email` freeform.
