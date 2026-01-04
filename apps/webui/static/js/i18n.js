/**
 * LingKong AI - Internationalization (i18n) Module
 * Lightweight, no-dependency language switching
 */

class I18n {
    constructor() {
        this.lang = localStorage.getItem('lingkong-lang') || 'en';
        this.messages = {};
        this.loaded = false;
    }

    /**
     * Load language file
     * @param {string} lang - Language code ('en' or 'zh')
     */
    async load(lang) {
        try {
            const response = await fetch(`/static/i18n/${lang}.json`);
            if (!response.ok) throw new Error(`Failed to load ${lang}.json`);
            this.messages = await response.json();
            this.lang = lang;
            this.loaded = true;
            localStorage.setItem('lingkong-lang', lang);
            document.documentElement.lang = lang;
            return true;
        } catch (error) {
            console.error('I18n load error:', error);
            return false;
        }
    }

    /**
     * Get translation by key
     * @param {string} key - Dot-notation key (e.g., 'nav.home')
     * @param {object} params - Optional parameters for interpolation
     */
    t(key, params = {}) {
        let value = key.split('.').reduce((obj, k) => obj?.[k], this.messages);
        if (value === undefined) {
            console.warn(`I18n: Missing translation for "${key}"`);
            return key;
        }
        // Simple interpolation: {{name}} -> value
        Object.keys(params).forEach(k => {
            value = value.replace(new RegExp(`{{${k}}}`, 'g'), params[k]);
        });
        return value;
    }

    /**
     * Apply translations to all elements with data-i18n attribute
     */
    apply() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const translation = this.t(key);

            // Handle different element types
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                if (el.placeholder) el.placeholder = translation;
            } else {
                el.textContent = translation;
            }
        });

        // Handle data-i18n-placeholder
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.placeholder = this.t(key);
        });

        // Handle data-i18n-title
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            el.title = this.t(key);
        });

        // Handle data-i18n-html (for HTML content)
        document.querySelectorAll('[data-i18n-html]').forEach(el => {
            const key = el.getAttribute('data-i18n-html');
            el.innerHTML = this.t(key);
        });

        // Update language switch buttons
        this.updateLangButtons();
    }

    /**
     * Update language switch button states
     */
    updateLangButtons() {
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.lang === this.lang);
        });
    }

    /**
     * Switch language
     * @param {string} lang - Language code
     */
    async switchTo(lang) {
        if (lang === this.lang && this.loaded) return;
        await this.load(lang);
        this.apply();
    }

    /**
     * Get current language
     */
    getCurrentLang() {
        return this.lang;
    }
}

// Global instance
const i18n = new I18n();

/**
 * Switch language (global function for onclick handlers)
 * @param {string} lang - Language code
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
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initI18n);
} else {
    initI18n();
}
