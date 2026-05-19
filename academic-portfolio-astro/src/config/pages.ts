import type { PagesConfig } from "../types";

export const PAGES: PagesConfig = {
    home: {
        title: "About",
        subtitle: "",
        isActive: true,
    },
    blog: {
        title: "Blog",
        subtitle: "Writing on Voice AI, speech research, and machine learning.",
        isActive: true,
    },
    publications: {
        title: "Publications",
        subtitle: "Peer-reviewed research at Interspeech, ICASSP, NeurIPS, and PMLR.",
        isActive: true,
    },
    talks: {
        title: "Talks",
        subtitle: "",
        isActive: false,
    },
    projects: {
        title: "Projects",
        subtitle: "",
        isActive: false,
    },
    teaching: {
        title: "Teaching",
        subtitle: "",
        isActive: false,
    },
    tags: {
        title: "Tags",
        subtitle: "",
        isActive: false,
    },
    cv: {
        title: "CV",
        subtitle: "",
        isActive: true,
    },
};
