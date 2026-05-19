import type { SocialLink } from "../types";

export const SOCIALS: SocialLink[] = [
    {
        name: "Github",
        href: "https://github.com/shangeth",
        linkTitle: "Shangeth Rajaa on GitHub",
        isActive: true,
    },
    {
        name: "Mail",
        href: "mailto:shangethrajaa@gmail.com",
        linkTitle: "Email Shangeth",
        isActive: true,
    },
    {
        name: "LinkedIn",
        href: "https://www.linkedin.com/in/shangeth",
        linkTitle: "Shangeth Rajaa on LinkedIn",
        isActive: true,
    },
    {
        name: "Google Scholar",
        href: "https://scholar.google.com/citations?user=apmFPkAAAAAJ",
        linkTitle: "Shangeth Rajaa on Google Scholar",
        isActive: true,
    },
    {
        name: "ORCID",
        href: "https://orcid.org/0009-0003-9819-6506",
        linkTitle: "Shangeth Rajaa on ORCID",
        isActive: true,
    },
];

export const SOCIAL_ICONS: Record<string, string> = {
    Github: "Github",
    Mail: "Mail",
    Linkedin: "LinkedIn",
    "Google Scholar": "GoogleScholar",
    ORCID: "ORCID",
    RSS: "RSS",
};
