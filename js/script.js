// toc-menu.js – sticky TOC, smooth highlighting, mobile menu toggle, and dynamic end‑padding
// Author: ChatGPT
// Usage: <script src="toc-menu.js" defer></script>

(function () {
    'use strict';

    /**
     * Insert an invisible spacer after a given element so that
   * short final sections can still trigger IntersectionObserver.
     * @param {HTMLElement|null} element – target element after which to insert spacer
     * @param {string} height – CSS height (e.g. '35vh' or '200px')
     */
    function appendPaddingAfter(element, height = '35vh') {
        if (!element) return;
        const padding = document.createElement('div');
        padding.style.height = height;
        padding.style.pointerEvents = 'none';
        padding.style.visibility = 'hidden';
        element.insertAdjacentElement('afterend', padding);
    }

    window.addEventListener("load", function() {
        const tocWrapper = document.getElementById("toc-wrapper");
        tocWrapper.classList.remove("no-transition");
    })

    document.addEventListener('DOMContentLoaded', () => {
        /* ────────────────────────
         * 1. Hamburger / sidebar menu toggle (optional)
         * ──────────────────────── */
        const toggleBtn = document.getElementById('menu-toggle');
        const menu = document.querySelector('.menu');

        if (toggleBtn && menu) {
            toggleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                menu.classList.toggle('hidden');
            });

            // Hide menu after clicking any link inside it
            menu.querySelectorAll('a').forEach((link) => {
                link.addEventListener('click', () => menu.classList.add('hidden'));
            });

            // Click outside to close
            document.addEventListener('click', (e) => {
                if (!menu.contains(e.target) && e.target !== toggleBtn) {
                    menu.classList.add('hidden');
                }
            });
        }

        /* ────────────────────────
         * 2. Table‑of‑Contents active‑section highlighting
         * ──────────────────────── */
        const links = document.querySelectorAll('#table-of-contents a');
        const headings = Array.from(links).map((link) =>
                                              document.getElementById(decodeURIComponent(link.getAttribute('href').slice(1)))
        );

        const isMobile = window.innerWidth < 600;
        const tocWrapper = document.querySelector('#toc-wrapper');
        const tocToggleBar = document.getElementById('toc-toggle-bar');

        if (isMobile && tocWrapper) {
            tocWrapper.classList.add('collapsed');
        }

        if (tocWrapper && tocToggleBar) {
            tocToggleBar.addEventListener('click', () => {
                tocWrapper.classList.toggle('collapsed');
            });
        }

        const setActiveLink = (el) => {
            links.forEach((l) => l.classList.remove('active'));
            if (el) {
                const idx = headings.indexOf(el);
                if (idx >= 0) links[idx].classList.add('active');
            }
        };

        const observer = new IntersectionObserver(
            (entries) => {
                const visible = entries.filter((e) => e.isIntersecting);
                if (visible.length > 0) {
                    // choose the top‑most visible heading in document order
                    const entry = visible.sort(
                        (a, b) => headings.indexOf(a.target) - headings.indexOf(b.target)
                    )[0];
                    setActiveLink(entry.target);
                }
            },
            {
                rootMargin: '0px 0px -70% 0px', // fire when heading is in top 30% viewport
                threshold: 0,
            }
        );

        headings.forEach((h) => h && observer.observe(h));

        // Highlight immediately on hash jump / page load
        const highlightFromHash = () => {
            const id = decodeURIComponent(location.hash.slice(1));
            const target = document.getElementById(id);
            if (target) setActiveLink(target);
        };

        window.addEventListener('hashchange', highlightFromHash);
        window.addEventListener('load', highlightFromHash);

        /* ────────────────────────
         * 3. Dynamic padding so the final section can trigger IO
         * ──────────────────────── */
        const footnotes = document.getElementById('footnotes');
        if (footnotes) {
            appendPaddingAfter(footnotes);
        } else if (links.length > 0) {
            const lastLink = links[links.length - 1];
            const lastId = decodeURIComponent(lastLink.getAttribute('href').slice(1));
            const lastEl = document.getElementById(lastId);
            appendPaddingAfter(lastEl);
        }
    });
})();
