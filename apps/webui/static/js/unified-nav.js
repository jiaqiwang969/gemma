/**
 * LingKong AI - Unified Navigation Component
 * Injects standard navbar and styles into any page.
 *
 * Works with i18n.js:
 * - Navbar HTML contains English defaults
 * - MutationObserver in i18n.js auto-translates when navbar is injected
 * - Also listens to i18n:ready event as backup
 */

(function () {
    // 1. Styles
    const styles = `
    /* Minimal Navbar */
    .navbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 1000;
        background: rgba(10, 10, 15, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255,255,255,0.1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    .nav-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 60px;
    }

    .navbar .logo {
        display: flex;
        align-items: center;
        gap: 8px;
        text-decoration: none;
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        flex-shrink: 0;
    }

    .navbar .logo-icon { font-size: 1.3rem; }

    .nav-links {
        display: flex;
        gap: 1.5rem;
        list-style: none;
        align-items: center;
        margin: 0;
        padding: 0;
    }

    .nav-links a {
        color: #a0a0b0;
        text-decoration: none;
        font-size: 0.85rem;
        transition: color 0.2s;
        white-space: nowrap;
    }

    .nav-links a:hover { color: #ffffff; }

    /* Language Switch */
    .lang-switch {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-left: 0.8rem;
        padding-left: 0.8rem;
        border-left: 1px solid rgba(255,255,255,0.1);
    }

    .lang-btn {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.1);
        color: #a0a0b0;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .lang-btn:hover {
        border-color: #8b5cf6;
        color: #ffffff;
    }

    .lang-btn.active {
        background: #8b5cf6;
        border-color: #8b5cf6;
        color: white;
    }

    /* Mobile Menu Button */
    .mobile-menu-btn {
        display: none;
        background: transparent;
        border: 1px solid rgba(255,255,255,0.1);
        color: #ffffff;
        font-size: 1.2rem;
        cursor: pointer;
        padding: 8px 12px;
        border-radius: 6px;
        transition: all 0.2s;
    }

    .mobile-menu-btn:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.2);
    }

    /* Mobile Responsive - Tablet */
    @media (max-width: 1024px) {
        .nav-links {
            gap: 1rem;
        }
        .nav-links a {
            font-size: 0.8rem;
        }
    }

    /* Mobile Responsive - Phone */
    @media (max-width: 768px) {
        .nav-container {
            padding: 0 1rem;
        }

        .navbar .logo {
            font-size: 1rem;
        }

        .navbar .logo-icon {
            font-size: 1.2rem;
        }

        .nav-links {
            display: none;
            position: fixed;
            top: 60px;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(10, 10, 15, 0.98);
            flex-direction: column;
            padding: 1.5rem;
            gap: 0;
            overflow-y: auto;
            z-index: 999;
        }

        .nav-links.open {
            display: flex;
        }

        .nav-links li {
            width: 100%;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .nav-links li:last-child {
            border-bottom: none;
        }

        .nav-links a {
            display: block;
            padding: 1rem 0;
            font-size: 1rem;
        }

        .lang-switch {
            margin-left: 0;
            padding-left: 0;
            border-left: none;
            padding-top: 1.5rem;
            margin-top: 0.5rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            justify-content: center;
            gap: 1rem;
        }

        .lang-btn {
            padding: 8px 16px;
            font-size: 0.9rem;
        }

        .mobile-menu-btn {
            display: flex;
            align-items: center;
            justify-content: center;
        }
    }

    /* Very small screens */
    @media (max-width: 360px) {
        .navbar .logo span:not(.logo-icon) {
            display: none;
        }
    }
    `;

    // 2. HTML Template - Contains English defaults as fallback
    const navbarHTML = `
    <div class="nav-container">
        <a href="/" class="logo">
            <span class="logo-icon">🐉</span>
            <span data-i18n="common.brand">LingKong AI</span>
        </a>
        <button class="mobile-menu-btn" aria-label="Toggle menu" aria-expanded="false">☰</button>
        <ul class="nav-links">
            <li><a href="/#features" data-i18n="nav.features">Features</a></li>
            <li><a href="/#how-it-works" data-i18n="nav.installation">Installation</a></li>
            <li><a href="/static/downloads.html" data-i18n="nav.downloads">Downloads</a></li>
            <li><a href="/static/docs.html" data-i18n="nav.apiDocs">API Docs</a></li>
            <li><a href="/static/playground/index.html" data-i18n="nav.playground">Playground</a></li>
            <li><a href="/static/tinybox/index.html" data-i18n="nav.tinybox">TinyBox DIY</a></li>
            <li><a href="https://github.com/jiaqiwang969/gemma" target="_blank">GitHub</a></li>
            <li class="lang-switch">
                <button class="lang-btn" data-lang="en" onclick="switchLang('en')">EN</button>
                <button class="lang-btn" data-lang="zh" onclick="switchLang('zh')">中</button>
            </li>
        </ul>
    </div>
    `;

    // 3. Injection Logic
    function injectNavbar() {
        // Inject Styles
        const styleSheet = document.createElement("style");
        styleSheet.id = "unified-nav-styles";
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);

        // Inject Navbar
        const nav = document.createElement("nav");
        nav.className = "navbar";
        nav.id = "unified-navbar";
        nav.innerHTML = navbarHTML;
        document.body.prepend(nav);

        // Add padding to body to prevent content overlap
        document.body.style.paddingTop = "60px";

        // Setup mobile menu toggle
        const menuBtn = nav.querySelector('.mobile-menu-btn');
        const navLinks = nav.querySelector('.nav-links');

        if (menuBtn && navLinks) {
            menuBtn.addEventListener('click', function() {
                const isOpen = navLinks.classList.toggle('open');
                this.setAttribute('aria-expanded', isOpen);
                this.textContent = isOpen ? '✕' : '☰';
                // Prevent body scroll when menu is open
                document.body.style.overflow = isOpen ? 'hidden' : '';
            });

            // Close menu when clicking a link
            navLinks.querySelectorAll('a').forEach(link => {
                link.addEventListener('click', () => {
                    navLinks.classList.remove('open');
                    menuBtn.setAttribute('aria-expanded', 'false');
                    menuBtn.textContent = '☰';
                    document.body.style.overflow = '';
                });
            });

            // Close menu on escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && navLinks.classList.contains('open')) {
                    navLinks.classList.remove('open');
                    menuBtn.setAttribute('aria-expanded', 'false');
                    menuBtn.textContent = '☰';
                    document.body.style.overflow = '';
                }
            });
        }

        // MutationObserver in i18n.js will auto-translate the navbar
        // But also listen to i18n:ready as backup for edge cases
        window.addEventListener('i18n:ready', () => {
            if (window.i18n && window.i18n.apply) {
                window.i18n.apply(nav);
            }
        });

        // If i18n is already loaded, apply immediately
        if (window.i18n && window.i18n.loaded) {
            window.i18n.apply(nav);
        }
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectNavbar);
    } else {
        injectNavbar();
    }

})();
