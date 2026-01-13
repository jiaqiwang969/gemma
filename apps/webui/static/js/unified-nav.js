/**
 * LingKong AI - Unified Navigation Component
 * Injects standard navbar and styles into any page.
 *
 * Works with i18n.js:
 * - Navbar HTML contains English defaults
 * - MutationObserver in i18n.js auto-translates when navbar is injected
 * - Also listens to i18n:ready event as backup
 *
 * Theme Support:
 * - Dark mode (default) and Light mode
 * - Persists user preference in localStorage
 * - Toggle button in navbar
 */

(function () {
    // 0. Theme CSS Variables
    const themeStyles = `
    /* Dark Theme (Default) - Azure/Sky Aesthetic */
    :root, [data-theme="dark"] {
        --bg-primary: #0a0a0f;
        --bg-secondary: #0f172a; /* Slate 900 */
        --bg-tertiary: #1e293b; /* Slate 800 */
        --bg-card: #111827; /* Gray 900 */
        --bg-code: #0b1120;
        --text-primary: #f8fafc; /* Slate 50 */
        --text-secondary: #cbd5e1; /* Slate 300 */
        --text-muted: #94a3b8; /* Slate 400 */
        
        /* New Brand Colors: Azure/Sky */
        --accent-primary: #3b82f6; /* Blue 500 */
        --accent-secondary: #0ea5e9; /* Sky 500 */
        --accent-green: #22c55e;
        --accent-blue: #3b82f6; 
        --accent-red: #ef4444;
        --accent-gradient: linear-gradient(135deg, #3b82f6, #0ea5e9);
        
        --border-color: rgba(255, 255, 255, 0.1);
        --shadow-color: rgba(0, 0, 0, 0.4);
        --navbar-bg: rgba(15, 23, 42, 0.72); /* Slate 900 with alpha */
        --dropdown-bg: rgba(17, 24, 39, 0.95);
        --bg-input: rgba(30, 41, 59, 0.6);

        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 18px;

        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        --shadow-lg: 0 12px 24px rgba(0,0,0,0.5);

        /* Aliases & Extras */
        --accent-yellow: #fbbc05;
        --bg: var(--bg-primary);
        --text: var(--text-primary);
        --border: var(--border-color);
        --primary: var(--accent-primary);
        --success: var(--accent-green);
        --warning: #f59e0b;
        --error: #ef4444;
        --code-bg: var(--bg-code);
        --code-text: #e2e8f0;
        --left-color: #3b82f6;
        --right-color: #22c55e;
        --gradient-purple: var(--accent-gradient); /* Kept var name for compatibility */
    }

    /* Light Theme - High Contrast & Clean Sky Aesthetic */
    [data-theme="light"] {
        --bg-primary: #ffffff;
        --bg-secondary: #f0f9ff; /* Alice Blue / Very faint sky */
        --bg-tertiary: #e0f2fe; /* Sky 100 */
        --bg-card: #ffffff;
        --bg-code: #f1f5f9;
        
        /* High Contrast Text */
        --text-primary: #020617; /* Slate 950 (Near Black) */
        --text-secondary: #334155; /* Slate 700 (Dark Gray) */
        --text-muted: #475569; /* Slate 600 */
        
        /* New Brand Colors: Azure/Sky in Light Mode */
        --accent-primary: #2563eb; /* Blue 600 (Darker for readability) */
        --accent-secondary: #0284c7; /* Sky 600 */
        --accent-green: #16a34a; 
        --accent-blue: #2563eb; 
        --accent-red: #dc2626; 
        --accent-yellow: #d97706; 
        --accent-gradient: linear-gradient(135deg, #2563eb, #0284c7);
        
        --border-color: #cbd5e1; /* Slate 300 - clearly visible borders */
        --shadow-color: rgba(0, 0, 0, 0.06); 
        --navbar-bg: rgba(255, 255, 255, 0.85);
        --dropdown-bg: rgba(255, 255, 255, 0.95);
        --bg-input: #ffffff;
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 18px;
        
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
        --shadow-lg: 0 12px 24px rgba(0,0,0,0.12);

        /* Aliases & Extras */
        --bg: var(--bg-primary);
        --text: var(--text-primary);
        --border: var(--border-color);
        --primary: var(--accent-primary);
        --success: var(--accent-green);
        --warning: #d97706;
        --error: #dc2626;
        --code-bg: var(--bg-code);
        --code-text: #0f172a;
        --left-color: #2563eb;
        --right-color: #16a34a;
        --gradient-purple: var(--accent-gradient);

        /* Grid Background Pattern - Subtle Blue Tint */
        background-color: var(--bg-primary);
        background-image: linear-gradient(#f0f9ff 1px, transparent 1px), linear-gradient(90deg, #f0f9ff 1px, transparent 1px);
        background-size: 40px 40px;
    }

    /* Apply theme colors to body */
    /* Apply theme colors to body with Apple-style typography */
    body {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        letter-spacing: -0.015em;
        line-height: 1.5;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.025em;
        font-weight: 600;
    }
    `;

    // 1. Styles
    const styles = `
    /* Apple-style Frosted Glass Navbar */
    .navbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 1000;
        background: var(--navbar-bg);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.05); /* Very subtle border */
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    
    [data-theme="dark"] .navbar {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
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
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 700;
        flex-shrink: 0;
        transition: color 0.3s ease;
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

    .nav-links > li {
        position: relative;
    }

    .nav-links a {
        color: var(--text-secondary);
        text-decoration: none;
        font-size: 0.85rem;
        transition: color 0.2s;
        white-space: nowrap;
    }

    .nav-links a:hover { color: var(--text-primary); }

    /* Dropdown Menu */
    .nav-dropdown {
        position: relative;
    }

    .nav-dropdown > a {
        display: flex;
        align-items: center;
        gap: 4px;
        cursor: pointer;
    }

    .nav-dropdown > a::after {
        content: '▾';
        font-size: 0.7rem;
        transition: transform 0.2s;
    }

    .nav-dropdown:hover > a::after {
        transform: rotate(180deg);
    }

    .dropdown-menu {
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--dropdown-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.5rem 0;
        min-width: 180px;
        opacity: 0;
        visibility: hidden;
        transition: all 0.2s;
        margin-top: 10px;
        box-shadow: 0 10px 40px var(--shadow-color);
    }

    .nav-dropdown:hover .dropdown-menu {
        opacity: 1;
        visibility: visible;
        margin-top: 5px;
    }

    .dropdown-menu li {
        list-style: none;
    }

    .dropdown-menu a {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        transition: all 0.2s;
    }

    .dropdown-menu a:hover {
        background: rgba(59, 130, 246, 0.1);
        color: var(--text-primary);
    }

    .dropdown-menu .dropdown-icon {
        font-size: 1rem;
        width: 20px;
        text-align: center;
    }

    .dropdown-menu .dropdown-label {
        flex: 1;
    }

    .dropdown-divider {
        height: 1px;
        background: var(--border-color);
        margin: 0.5rem 0;
    }

    /* Language Switch */
    .lang-switch {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-left: 0.8rem;
        padding-left: 0.8rem;
        border-left: 1px solid var(--border-color);
    }

    .lang-btn {
        background: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .lang-btn:hover {
        border-color: var(--accent-primary);
        color: var(--text-primary);
    }

    .lang-btn.active {
        background: var(--accent-primary);
        border-color: var(--accent-primary);
        color: white;
    }

    /* Theme Toggle Button */
    .theme-toggle {
        background: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.5rem;
    }

    .theme-toggle:hover {
        border-color: var(--accent-primary);
        color: var(--text-primary);
        background: rgba(59, 130, 246, 0.1);
    }

    /* Mobile Menu Button */
    .mobile-menu-btn {
        display: none;
        background: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-size: 1.2rem;
        cursor: pointer;
        padding: 8px 12px;
        border-radius: 6px;
        transition: all 0.2s;
    }

        .mobile-menu-btn:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: var(--accent-primary);
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
            padding: 0 1.25rem;
            height: 72px; /* Increased height for better touch target */
        }

        .navbar .logo {
            font-size: 1.2rem;
        }

        .navbar .logo-icon {
            font-size: 1.5rem;
        }

        .nav-links {
            display: none;
            position: fixed;
        .nav-links {
            display: none;
            position: fixed;
            top: 72px; /* Match new navbar height */
            left: 0;
            right: 0;
            height: calc(100vh - 72px); /* Full screen height minus navbar */
            background: var(--navbar-bg); /* Solid background from theme */
            flex-direction: column;
            padding: 1.5rem 2rem; /* More horizontal padding */
            gap: 0;
            overflow-y: auto;
            z-index: 999;
             /* Stronger backdrop for legibility */
            backdrop-filter: saturate(180%) blur(40px);
            -webkit-backdrop-filter: saturate(180%) blur(40px);
        }

        .nav-links.open {
            display: flex;
            animation: slideDown 0.3s ease-out forwards;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .nav-links.open {
            display: flex;
            animation: slideDown 0.3s ease-out forwards;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .nav-links > li {
            width: 100%;
            border-bottom: 1px solid var(--border-color);
        }

        .nav-links > li:last-child {
            border-bottom: none;
        }

        .nav-links > li > a {
            display: block;
            padding: 1.1rem 0; /* Larger tap area */
            font-size: 1.05rem;
        }

        /* Mobile Dropdown */
        .nav-dropdown > a::after {
            content: '+';
            font-size: 1.2rem;
            margin-left: auto;
            font-weight: 300;
        }

        .nav-dropdown.open > a::after {
            content: '−';
        }

        .dropdown-menu {
            position: static;
            transform: none;
            opacity: 1;
            visibility: visible;
            background: rgba(59, 130, 246, 0.05); /* Blue tint instead of purple */
            border: none;
            border-radius: 0;
            margin: 0;
            padding: 0;
            box-shadow: none;
            display: none;
        }
        
        .dropdown-menu a:hover {
             background: rgba(59, 130, 246, 0.1);
        }

        .nav-dropdown.open .dropdown-menu {
            display: block;
        }

        .dropdown-menu a {
            padding: 0.9rem 1.5rem;
            color: var(--text-secondary);
        }

        .lang-switch {
            margin-left: 0;
            padding-left: 0;
            border-left: none;
            padding-top: 1.5rem;
            margin-top: 0.5rem;
            border-top: 1px solid var(--border-color);
            justify-content: center;
            gap: 1.5rem;
        }

        .lang-btn {
            padding: 10px 20px;
            font-size: 1rem;
        }

        .theme-toggle {
            padding: 10px 20px;
            font-size: 1.25rem;
        }

        .mobile-menu-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 44px;
            height: 44px; /* Proper touch target size */
            font-size: 1.4rem;
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
            <li><a href="/static/downloads.html" data-i18n="nav.downloads">Downloads</a></li>
            <li><a href="/static/docs.html" data-i18n="nav.apiDocs">API Docs</a></li>
            <li><a href="/static/playground/index.html" data-i18n="nav.playground">Playground</a></li>
            <li><a href="/static/tinybox/index.html" data-i18n="nav.tinybox">TinyBox DIY</a></li>
            <li class="nav-dropdown">
                <a href="javascript:void(0)" data-i18n="nav.resources">Resources</a>
                <ul class="dropdown-menu">
                    <li>
                        <a href="/static/evolution/index.html">
                            <span class="dropdown-icon">🧠</span>
                            <span class="dropdown-label" data-i18n="nav.evolution">Evolution</span>
                        </a>
                    </li>
                    <li>
                        <a href="/static/encryption/index.html">
                            <span class="dropdown-icon">🔐</span>
                            <span class="dropdown-label" data-i18n="nav.encryption">Encryption</span>
                        </a>
                    </li>
                    <li>
                        <a href="/static/silent-compute/index.html">
                            <span class="dropdown-icon">🤫</span>
                            <span class="dropdown-label" data-i18n="nav.silent_compute">Silent Compute</span>
                        </a>
                    </li>
                    <li class="dropdown-divider"></li>
                    <li>
                        <a href="/static/pitch.html">
                            <span class="dropdown-icon">📊</span>
                            <span class="dropdown-label" data-i18n="nav.pitch">Business Plan</span>
                        </a>
                    </li>
                </ul>
            </li>
            <li><a href="https://github.com/jiaqiwang969/gemma" target="_blank">GitHub</a></li>
            <li class="lang-switch">
                <button class="theme-toggle" id="themeToggle" title="Toggle theme" aria-label="Toggle dark/light mode">🌙</button>
                <button class="lang-btn" data-lang="en" onclick="switchLang('en')">EN</button>
                <button class="lang-btn" data-lang="zh" onclick="switchLang('zh')">中</button>
            </li>
        </ul>
    </div>
    `;

    // 3. Theme Management
    const THEME_KEY = 'lingkong-theme';

    function getStoredTheme() {
        return localStorage.getItem(THEME_KEY) || 'dark';
    }

    function setStoredTheme(theme) {
        localStorage.setItem(THEME_KEY, theme);
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const toggleBtn = document.getElementById('themeToggle');
        if (toggleBtn) {
            toggleBtn.textContent = theme === 'dark' ? '🌙' : '☀️';
            toggleBtn.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
        }
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
        setStoredTheme(newTheme);
    }

    // Expose theme functions globally
    window.toggleTheme = toggleTheme;
    window.setTheme = function (theme) {
        applyTheme(theme);
        setStoredTheme(theme);
    };

    // 4. Injection Logic
    function injectNavbar() {
        // Inject Theme Styles first (before navbar styles)
        const themeStyleSheet = document.createElement("style");
        themeStyleSheet.id = "unified-theme-styles";
        themeStyleSheet.textContent = themeStyles;
        document.head.insertBefore(themeStyleSheet, document.head.firstChild);

        // Apply stored theme immediately to prevent flash
        applyTheme(getStoredTheme());

        // Inject Navbar Styles
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
            menuBtn.addEventListener('click', function () {
                const isOpen = navLinks.classList.toggle('open');
                this.setAttribute('aria-expanded', isOpen);
                this.textContent = isOpen ? '✕' : '☰';
                // Prevent body scroll when menu is open
                document.body.style.overflow = isOpen ? 'hidden' : '';
            });

            // Close menu when clicking a link (except dropdown toggles)
            navLinks.querySelectorAll('a:not(.nav-dropdown > a)').forEach(link => {
                link.addEventListener('click', () => {
                    navLinks.classList.remove('open');
                    menuBtn.setAttribute('aria-expanded', 'false');
                    menuBtn.textContent = '☰';
                    document.body.style.overflow = '';
                });
            });

            // Mobile dropdown toggle
            navLinks.querySelectorAll('.nav-dropdown > a').forEach(dropdownToggle => {
                dropdownToggle.addEventListener('click', (e) => {
                    if (window.innerWidth <= 768) {
                        e.preventDefault();
                        dropdownToggle.parentElement.classList.toggle('open');
                    }
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

        // Setup theme toggle
        const themeToggle = nav.querySelector('#themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', toggleTheme);
            // Update button icon based on current theme
            const currentTheme = getStoredTheme();
            themeToggle.textContent = currentTheme === 'dark' ? '🌙' : '☀️';
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
        document.addEventListener('DOMContentLoaded', () => {
            // Check if we should inject navbar or just theme
            const scriptTag = document.querySelector('script[src*="unified-nav.js"]');
            const noNavbar = scriptTag && scriptTag.getAttribute('data-no-navbar') !== null;

            if (noNavbar) {
                // Only inject theme styles and setup theme logic
                const themeStyleSheet = document.createElement("style");
                themeStyleSheet.id = "unified-theme-styles";
                themeStyleSheet.textContent = themeStyles;
                document.head.insertBefore(themeStyleSheet, document.head.firstChild);
                applyTheme(getStoredTheme());
                // Expose setup complete event possibly?
            } else {
                injectNavbar();
            }
        });
    } else {
        const scriptTag = document.querySelector('script[src*="unified-nav.js"]');
        const noNavbar = scriptTag && scriptTag.getAttribute('data-no-navbar') !== null;

        if (noNavbar) {
            const themeStyleSheet = document.createElement("style");
            themeStyleSheet.id = "unified-theme-styles";
            themeStyleSheet.textContent = themeStyles;
            document.head.insertBefore(themeStyleSheet, document.head.firstChild);
            applyTheme(getStoredTheme());
        } else {
            injectNavbar();
        }
    }

})();
