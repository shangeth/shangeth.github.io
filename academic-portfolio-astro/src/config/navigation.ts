import type { NavLink } from "../types";

export const NAV_LINKS: NavLink[] = [
    { href: "/", label: "About", isActive: true },
    { href: "/cv", label: "CV", isActive: true },
    { href: "/publications", label: "Publications", isActive: true },
    { href: "/posts", label: "Blog", isActive: true },
    { href: "/courses", label: "Courses", isActive: true, alignRight: true },
];
