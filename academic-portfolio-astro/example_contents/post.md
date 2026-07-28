---
title: "Sample Post Title"
date: "2026-04-27"
description: "A short preview of the post seen in the blog list. This description appears in the listing and should be 1-2 sentences."
author: "Your Name"
tags:
  - "Tag1"
  - "Tag2"
image: "/images/placeholder.svg"
---

# Introduction

Start writing your blog post here. You can add images:

![Sample image](/images/placeholder.svg)

The image above demonstrates how to include images in your posts. Images are useful for:
- Visual illustrations
- Diagrams and charts
- Photos of your work

## A Section Header

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

### Subsection with Code

```python
def hello_world():
    print("Hello, World!")
    return 0
```

## Lists and More

You can use various markdown features:

- **Bold text** for emphasis
- *Italic text* for subtle emphasis
- `inline code` for technical terms
- [Links](https://example.com) for references

## Math Support

This template supports LaTeX math rendering:

Inline math: $E = mc^2$

Block math:

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

## Conclusion

End with a summary or call to action.

## Code / model badges (optional)

If the post links out to a repo or hosted models, add shields.io badges right after the frontmatter (before the first heading), GitHub-README style. Plain markdown, no custom CSS needed — `global.css` has a rule (`.prose img[src*="shields.io"]`) that renders these inline instead of as block figures:

```md
[![GitHub](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/OWNER/REPO)
[![HuggingFace](https://img.shields.io/badge/🤗-MODEL%20NAME-blue.svg)](https://huggingface.co/OWNER/MODEL)
[![Open in Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?logo=googlecolab&color=blue)](https://colab.research.google.com/drive/NOTEBOOK_ID)
```

Default set is GitHub, HuggingFace, Colab — one badge per artifact the post links to (e.g. one HuggingFace badge per model if there are several). Add other shields.io badges (license, stars, etc.) only if the post specifically calls for them.

## Citation (optional)

If the post has a BibTeX entry (e.g. it's tied to a repo/paper), add a "Cite" badge to the badge row linking to a `#citation` anchor, and a `## Citation` section at the very end of the post — after References — with the entry in a fenced ` ```bibtex ` code block (Shiki highlights `bibtex` natively, no fallback needed):

```md
[![Cite this work](https://img.shields.io/badge/Cite-BibTeX-yellow.svg)](#citation)
```

```
## Citation

If you use this work, please cite:

\`\`\`bibtex
@misc{key, author = {...}, title = {{...}}, url = {...}}
\`\`\`
```

## References (optional)

For posts with a bibliography, use a numbered, hanging-indent list via the `.references` class instead of plain paragraphs — this is the standard citation format for this site. Markdown emphasis/links do not parse inside a raw HTML block, so write `<em>` and `<a>` tags directly:

```html
<ol class="references">
<li>Author, A., et al. (2024). <em>Paper Title.</em> Venue. <a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank" rel="noopener noreferrer">arXiv:XXXX.XXXXX</a>.</li>
<li>Another, B. (2023). <em>Another Paper.</em> Conference 2023, 1-10.</li>
</ol>
```

Keep inline citations in the body as direct hyperlinks on the relevant term (e.g. `[HuBERT](https://arxiv.org/abs/2106.07447)`) rather than footnote markers — readers jump straight to the source, and the References list at the bottom stays for full bibliographic completeness.