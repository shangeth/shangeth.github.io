import type { SiteConfig, ThemeConfig, SettingsConfig, UmamiAnalyticsConfig, AnalyticsConfig } from "../types";

export const SITE: SiteConfig = {
    website: "https://shangeth.com/",
    author: "Shangeth Rajaa",
    desc: "Shangeth Rajaa, Senior ML Scientist at Anyreach AI (ex-Skit.ai). Voice AI research on full-duplex spoken dialogue, turn-taking, and speech LLMs.",
    title: "Shangeth Rajaa",
    ogImage: "avatar.jpeg",
    postPerPage: 5,
    favicon: "/favicon.svg",
    lang: "en",
};

export const THEME_CONFIG: ThemeConfig = {
    lightAndDark: true,
    themeLight: "light_default",
    themeDark: "dark_notepad",
};

export const SETTINGS: SettingsConfig = {
    showTagsInNavbar: false,
    showRSSInFooter: true,
    addDevToolsInProduction: false,
};

const umami: UmamiAnalyticsConfig = {
    websiteId: "f98471af-a3c9-483d-8af6-59cc81d51877",
    src: "https://cloud.umami.is/script.js",
}

export const ANALYTICS: AnalyticsConfig = {
    ga4Id: "G-733WY72RFD",
    umami: umami
};
