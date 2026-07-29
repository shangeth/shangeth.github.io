# Analytics & Tracking Reference

Reference doc for anything related to analytics, Umami, and UTM-tagged links on this site. Not a content-authoring doc (see `example_contents/` for that) — this is purely for tracking/measurement.

## What's set up

- **Google Analytics (GA4)** — `src/config/site.ts` (`ANALYTICS.ga4Id`). Standard pageview tracking, wired in `BaseLayout.astro`.
- **Umami** — `src/config/site.ts` (`ANALYTICS.umami.websiteId`). Cookieless, simpler dashboard. Also wired in `BaseLayout.astro`.
- Both scripts are lazy-loaded: they only load after the first `scroll`/`click`/`mousemove`/`touchstart`/`keydown`, or a 5-second fallback timer — a deliberate perf choice, not a bug, so don't expect events to fire instantly on a page that's just sitting open untouched.

## Custom events already wired in (Umami)

| Event name | Where | Extra data | File |
|---|---|---|---|
| `resume-download` | Sidebar "Resume" link (every page) | — | `src/components/layout/LeftSidebar.astro` |
| `social-click` | Sidebar social icons (every page) | `network`: Github / Mail / LinkedIn / Google Scholar / ORCID | `src/components/layout/LeftSidebar.astro` |
| `cta-click` | The two "book a call" links on the homepage bio | `location`: `highlight` or `bio-footer` | `src/content/bio.md` |
| `citation-copy` | Copy button on a code block, but only fires when the block is a citation (`data-language="bibtex"`) | — | `src/layouts/BaseDetail.astro` |

To add a new one: use `data-umami-event="event-name"` (plus `data-umami-event-<key>="value"` for extra data) on any link/button in markdown or `.astro` files. For elements created dynamically in JS, call `window.umami?.track("event-name", { key: "value" })` instead (always with `?.` — Umami may not have loaded yet, e.g. ad blockers).

## UTM parameters — when to use each

| Param | Answers | Use it when |
|---|---|---|
| `utm_source` | Which specific place? | Always. The exact platform: `linkedin`, `x`, `resume`, `github`, `email`. |
| `utm_medium` | What kind of channel? | Always. The category `source` belongs to: `social`, `pdf`, `profile`, `signature`, `offline`. |
| `utm_campaign` | Part of what push? | When the link is tied to a specific effort you want to measure as a whole across multiple sources — e.g. `job-search`, `speechllm-launch`. |
| `utm_content` | Which specific variant? | Only if you have 2+ links with the same source+medium+campaign and want to A/B compare — e.g. two different LinkedIn posts. |
| `utm_term` | Which paid keyword? | Paid search only (Google/Bing ads). Not relevant here — skip. |

**Rule of thumb:** always set `source` + `medium`. Add `campaign` when it's part of a push you want to track as a whole. Only add `content` when posting more than once on the same platform.

## Link template

```
https://shangeth.com/?utm_source=<source>&utm_medium=<medium>&utm_campaign=<campaign>
```

Keep values lowercase and consistent — `linkedin` vs `LinkedIn` vs `Linkedin` shows up as three separate sources in Umami.

### Ready-made links for common spots

| Where | Link |
|---|---|
| LinkedIn post | `https://shangeth.com/?utm_source=linkedin&utm_medium=social` |
| X / Twitter post | `https://shangeth.com/?utm_source=x&utm_medium=social` |
| Resume PDF (link at the top of the resume) | `https://shangeth.com/?utm_source=resume&utm_medium=pdf` |
| Email signature | `https://shangeth.com/?utm_source=email&utm_medium=signature` |
| GitHub profile / README | `https://shangeth.com/?utm_source=github&utm_medium=profile` |
| Conference talk / slides | `https://shangeth.com/?utm_source=talk&utm_medium=offline&utm_campaign=<event-name>` |

Only tag the URL at the point you share it — never change the canonical links inside the site itself (nav, sidebar socials, etc.), just the copy you paste externally.

## Campaign name log

Keep campaign names consistent over time by reusing entries from this list instead of inventing new spellings:

| Campaign | Used for |
|---|---|
| `job-search` | General "open to new roles" push (LinkedIn/X posts, resume, outreach) |

## Where to see results in Umami

- **Raw verification** — "Pages"/"URLs" report shows the full path including the query string, confirming a hit was logged at all.
- **Actual UTM breakdown** — go to **Insights**, add `utm_source` (or `utm_medium`/`utm_campaign`) as the field to group by. The default Pages/Referrers views won't aggregate by UTM automatically; Insights is where that breakdown lives.
- Referrer showing blank is normal for direct address-bar entry and for a lot of real social-app traffic (LinkedIn/X in-app browsers often strip it) — UTM params are the reliable signal, not the referrer header.
