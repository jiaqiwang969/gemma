/**
 * LingKong AI - Internationalization (i18n) Module
 *
 * Design principles:
 * 1. HTML contains default text (English) as fallback
 * 2. Translation only replaces text AFTER JSON is loaded
 * 3. Custom event 'i18n:ready' notifies all components
 * 4. MutationObserver auto-translates dynamically added elements
 */

class I18n {
    constructor() {
        this.lang = localStorage.getItem('lingkong-lang') || this.detectLanguage();
        this.messages = {};
        this.loaded = false;
        this.loading = false;
        this.readyCallbacks = [];
    }

    /**
     * Detect user's preferred language
     */
    detectLanguage() {
        const browserLang = navigator.language || navigator.userLanguage;
        return browserLang.startsWith('zh') ? 'zh' : 'en';
    }

    /**
     * Load language file
     */
    async load(lang) {
        if (this.loading) return this.waitForLoad();
        this.loading = true;

        try {
            const response = await fetch(`/static/i18n/${lang}.json`);
            if (!response.ok) throw new Error(`Failed to load ${lang}.json`);
            this.messages = await response.json();
            this.lang = lang;
            this.loaded = true;
            localStorage.setItem('lingkong-lang', lang);
            document.documentElement.lang = lang;

            // Notify all waiting components
            this.readyCallbacks.forEach(cb => cb());
            this.readyCallbacks = [];

            // Dispatch custom event for external listeners
            window.dispatchEvent(new CustomEvent('i18n:ready', { detail: { lang } }));

            return true;
        } catch (error) {
            console.error('I18n load error:', error);
            this.loaded = true; // Mark as loaded to prevent infinite waiting
            return false;
        } finally {
            this.loading = false;
        }
    }

    /**
     * Wait for current load to complete
     */
    waitForLoad() {
        return new Promise(resolve => {
            if (this.loaded) {
                resolve(true);
            } else {
                this.readyCallbacks.push(() => resolve(true));
            }
        });
    }

    /**
     * Execute callback when i18n is ready
     */
    onReady(callback) {
        if (this.loaded) {
            callback();
        } else {
            this.readyCallbacks.push(callback);
        }
    }

    /**
     * Get translation by key
     * @param {string} key - Dot-notation key (e.g., 'nav.home')
     * @param {object} params - Optional parameters for interpolation
     * @returns {string|null} - Translation or null if not found
     */
    t(key, params = {}) {
        let value = key.split('.').reduce((obj, k) => obj?.[k], this.messages);
        if (value === undefined) {
            return null; // Return null instead of key, let caller decide fallback
        }
        // Simple interpolation: {{name}} -> value
        Object.keys(params).forEach(k => {
            value = value.replace(new RegExp(`{{${k}}}`, 'g'), params[k]);
        });
        return value;
    }

    /**
     * Apply translations to elements
     * @param {Element} root - Root element to search within (default: document)
     */
    apply(root = document) {
        if (!this.loaded) return;

        // Update page title if data-i18n-title-key is set on html element
        const titleKey = document.documentElement.getAttribute('data-i18n-title-key');
        if (titleKey) {
            const title = this.t(titleKey);
            if (title) document.title = title;
        }

        // data-i18n: replace textContent
        root.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const translation = this.t(key);
            if (translation !== null) {
                if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                    if (el.placeholder !== undefined) el.placeholder = translation;
                } else {
                    el.textContent = translation;
                }
            }
            // If translation is null, keep original HTML content as fallback
        });

        // data-i18n-placeholder
        root.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            const translation = this.t(key);
            if (translation !== null) el.placeholder = translation;
        });

        // data-i18n-title
        root.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            const translation = this.t(key);
            if (translation !== null) el.title = translation;
        });

        // data-i18n-html (for HTML content)
        root.querySelectorAll('[data-i18n-html]').forEach(el => {
            const key = el.getAttribute('data-i18n-html');
            const translation = this.t(key);
            if (translation !== null) el.innerHTML = translation;
        });

        this.updateLangButtons(root);
    }

    /**
     * Update language switch button states
     */
    updateLangButtons(root = document) {
        root.querySelectorAll('.lang-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === this.lang);
        });
    }

    /**
     * Switch language
     */
    async switchTo(lang) {
        if (lang === this.lang && this.loaded) {
            this.apply();
            return;
        }
        this.loaded = false;
        await this.load(lang);
        this.apply();
    }

    /**
     * Get current language
     */
    getCurrentLang() {
        return this.lang;
    }

    /**
     * Setup MutationObserver to auto-translate dynamically added elements
     */
    setupObserver() {
        if (this.observer) return;

        this.observer = new MutationObserver(mutations => {
            if (!this.loaded) return;

            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        // Translate the new element and its children
                        if (node.hasAttribute?.('data-i18n') ||
                            node.querySelector?.('[data-i18n]')) {
                            this.apply(node.parentElement || node);
                        }
                    }
                });
            });
        });

        this.observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
}

// Global instance
const i18n = new I18n();

/**
 * Switch language (global function for onclick handlers)
 */
async function switchLang(lang) {
    await i18n.switchTo(lang);
}

/**
 * Initialize i18n on page load
 */
async function initI18n() {
    await i18n.load(i18n.lang);
    i18n.apply();
    i18n.setupObserver();
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initI18n);
} else {
    initI18n();
}
