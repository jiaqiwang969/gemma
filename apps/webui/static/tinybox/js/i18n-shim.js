/**
 * I18n Shim for TinyBox Page
 * Translates React-rendered Chinese content to English dynamically
 */

(function () {
    const DEBUG = false;

    // Complete Chinese -> English translation mapping
    const TRANSLATIONS = {
        // Page title and subtitle
        "Tinybox DIY Guide - 6x RX 7900 XTX AI 工作站组装指南": "Tinybox DIY Guide - 6x RX 7900 XTX AI Workstation Assembly Guide",
        "从硬件选型到开源优化，手把手教你组装一台 6x RX 7900 XTX 的 AI 算力怪兽": "From hardware selection to open-source optimization, a step-by-step guide to building a 6x RX 7900 XTX AI computing beast",

        // Section titles
        "硬件选型": "Hardware Selection",
        "组装配置": "Assembly & Config",
        "软件方案": "Software Setup",
        "开始组装": "Start Assembly",
        "手把手组装与配置": "Step-by-Step Assembly & Configuration",
        "物理组装": "Physical Assembly",
        "前言": "Introduction",
        "为何选择 DIY Tinybox？": "Why DIY Tinybox?",

        // Introduction content
        "本指南复刻了 George Hotz (geohot) 的 Tinybox Red 核心理念，融入社区最佳实践和全方位开源优化方案。 无论你是 AI 研究者、深度学习工程师，还是硬核 DIY 玩家，这份指南都将帮助你构建一台真正属于自己的 AI 算力怪兽。": "This guide replicates the core philosophy of George Hotz's (geohot) Tinybox Red, incorporating community best practices and comprehensive open-source optimization. Whether you're an AI researcher, deep learning engineer, or hardcore DIY enthusiast, this guide will help you build your own AI computing beast.",
        "基于开源精神，为 AI 硬核玩家打造": "Built on open-source spirit, for hardcore AI enthusiasts",
        "内容基于社区实践整理": "Content based on community practices",
        "全程拥抱开源生态": "Embracing open-source ecosystem throughout",

        // Hardware section
        "它提供了无与伦比的稳定性和 PCIe 带宽。 以下是完整的 BOM 清单，总预算约为 ": "It provides unparalleled stability and PCIe bandwidth. Here's the complete BOM list, with a total budget of approximately ",
        "获得 144GB 显存和强大的 FP16 计算能力": "Get 144GB VRAM and powerful FP16 computing capability",
        "具备向 UALink 互联标准升级的潜力": "Potential to upgrade to UALink interconnect standard",
        "必须 ECC，插满 8 通道": "Must be ECC, fill all 8 channels",
        "必须双电源，>3000W": "Must have dual PSU, >3000W",
        "也可选 PowerColor Hellhound": "PowerColor Hellhound also available",
        "高质量 PCIe 4.0 x16": "High-quality PCIe 4.0 x16",
        "开放式 6-8 卡矿架": "Open-air 6-8 GPU mining frame",
        "单价 (¥)": "Unit Price ($)",

        // Assembly section
        "根内存安装到 ROMED8-2T 主板上。先用一张显卡直连主板，进行点亮测试。": "Install memory on ROMED8-2T motherboard. First use one GPU directly connected to motherboard for power-on test.",
        "将主板固定在开放式矿架底层。将 6 块显卡均匀地固定在上层。": "Mount the motherboard on the bottom of the open frame. Mount 6 GPUs evenly on the upper level.",
        "延长线分别插入主板的 PCIe x16 插槽，另一端连接到显卡。": "Insert riser cables into motherboard PCIe x16 slots, connect the other end to GPUs.",
        "使用电源同步线连接两个电源。": "Connect two PSUs using a sync cable.",
        "两个电源的交流输入端必须插入同一个接地良好的插排，确保共地，防止电位差烧毁设备。": "Both PSU AC inputs must be plugged into the same well-grounded power strip to ensure common ground and prevent equipment damage from potential difference.",
        "双电源布线 (安全第一！)": "Dual PSU Wiring (Safety First!)",
        "电源 A": "PSU A",
        "电源 B": "PSU B",
        "连接主板、CPU、硬盘，以及靠近 CPU 的 3 块显卡。": "Connect motherboard, CPU, storage, and 3 GPUs near the CPU.",
        "只连接另外 3 块显卡。": "Connect only the other 3 GPUs.",

        // Cooling section
        "最佳散热空间": "Optimal Cooling Space",
        "开放式矿架提供最佳散热空间。确保显卡之间有足够间距， 并配备额外的机箱风扇形成良好的风道。": "Open frame provides optimal cooling space. Ensure adequate spacing between GPUs and add extra case fans for good airflow.",
        "设置激进的风扇曲线，确保高负载下温度不超过 ": "Set aggressive fan curves to ensure temperature stays below ",

        // Power section
        "电源与散热优化 (LACT)": "Power & Cooling Optimization (LACT)",
        "功耗/风扇控制": "Power/Fan Control",
        "使用 LACT 限制每张卡功耗为 280W": "Use LACT to limit each card to 280W",
        "是功能最全面的 GUI 工具，可以统一管理所有 GPU 的功耗和风扇曲线。": "is the most comprehensive GUI tool for unified management of all GPU power and fan curves.",
        "实时监控所有 GPU 的状态，包括温度、功耗、显存使用等。": "Real-time monitoring of all GPU status including temperature, power, and VRAM usage.",
        "检查家庭电路负载能力 (需 16A 插座)": "Check home circuit load capacity (requires 16A outlet)",
        "更换更高质量的延长线 (如 Linkup, ADT-Link)": "Replace with higher quality riser cables (e.g., Linkup, ADT-Link)",
        "瞬时峰值功耗超过承载能力": "Instantaneous peak power exceeds capacity",

        // Storage section
        "系统盘": "System Drive",
        "组建 RAID 0": "Build RAID 0",
        "系统盘使用单独的 NVMe SSD，数据盘组建 RAID 0 以获得最大读写速度。 如果预算充足，可以考虑使用 PCIe 4.0 x4 的企业级 SSD。": "Use a separate NVMe SSD for system drive, build RAID 0 for data drives for maximum read/write speed. If budget allows, consider enterprise-grade PCIe 4.0 x4 SSDs.",

        // Software section - ROCm
        "安装 ROCm": "Install ROCm",
        "方案 A：纯 Tinygrad 路径": "Option A: Pure Tinygrad Path",
        "方案 B：完整 ROCm 路径": "Option B: Full ROCm Path",
        "同方案 A": "Same as Option A",
        "绕过庞大的 ROCm 完整版，直接使用 tinygrad 的用户空间驱动，更轻量、更稳定。": "Bypass the full ROCm stack, use tinygrad's userspace driver directly - lighter and more stable.",
        "兼容性好，适合需要使用 PyTorch、TensorFlow 等主流框架的场景。": "Good compatibility, suitable for scenarios requiring mainstream frameworks like PyTorch, TensorFlow.",

        // Software section - Tinygrad
        "安装 Tinygrad": "Install Tinygrad",
        "轻量级深度学习框架": "Lightweight deep learning framework",
        "高性能 GPU 编程": "High-performance GPU programming",

        // BIOS settings
        "启用大于 4GB 的 BAR 寻址": "Enable BAR addressing larger than 4GB",
        "允许 CPU 一次性访问全部显存": "Allow CPU to access all VRAM at once",
        "提升 P2P 效率": "Improve P2P efficiency",
        "确保 BIOS 中 IOMMU 已启用": "Ensure IOMMU is enabled in BIOS",
        "检查内核参数 amd_iommu=on iommu=pt": "Check kernel parameters amd_iommu=on iommu=pt",

        // Configuration
        "基于应用的配置": "Application-based Configuration",
        "将以下环境变量添加到 ~/.bashrc 文件中：": "Add the following environment variables to ~/.bashrc:",
        "编辑 /etc/default/grub 后运行 sudo update-grub 并重启": "After editing /etc/default/grub, run sudo update-grub and reboot",
        "中加入 pci=noaer 参数": "add pci=noaer parameter",
        "中将 PCIe 降级到 Gen3": "downgrade PCIe to Gen3",

        // Verification
        "如果能看到 6 张卡的信息，则安装成功。": "If you can see info for all 6 cards, installation is successful.",
        "使用 rocm-bandwidth-test 工具诊断": "Use rocm-bandwidth-test tool for diagnosis",

        // PyTorch
        "如果使用 PyTorch，务必从源码编译安装 ROCm 版本的 FlashAttention，可以极大提升 Transformer 模型的性能。": "If using PyTorch, be sure to compile and install the ROCm version of FlashAttention from source - it can greatly improve Transformer model performance.",

        // Recommendations
        "我们推荐采用 ": "We recommend using ",
        "在训练时，使用 ": "During training, use ",

        // Meta description
        "完整的 Tinybox DIY 组装指南，包含硬件选型、组装步骤、软件配置和开源优化方案。": "Complete Tinybox DIY assembly guide, including hardware selection, assembly steps, software configuration, and open-source optimization."
    };

    /**
     * Get current language
     */
    function getCurrentLang() {
        return localStorage.getItem('lingkong-lang') ||
               (navigator.language.startsWith('zh') ? 'zh' : 'en');
    }

    /**
     * Translate text based on current language
     */
    function translate(text) {
        const lang = getCurrentLang();
        if (lang === 'en' && TRANSLATIONS[text]) {
            return TRANSLATIONS[text];
        }
        return text;
    }

    /**
     * Walk the DOM and translate text nodes
     */
    function translatePage(root) {
        const lang = getCurrentLang();
        if (lang === 'zh') {
            if (DEBUG) console.log('[TinyBox i18n] Language is Chinese, skipping translation');
            return;
        }

        if (DEBUG) console.log('[TinyBox i18n] Translating to English...');

        const walker = document.createTreeWalker(
            root,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        const nodesToTranslate = [];
        let node;
        while (node = walker.nextNode()) {
            const text = node.nodeValue.trim();
            if (text && TRANSLATIONS[text]) {
                nodesToTranslate.push({ node, text });
            }
        }

        nodesToTranslate.forEach(({ node, text }) => {
            const translated = TRANSLATIONS[text];
            if (DEBUG) console.log(`[TinyBox i18n] "${text.substring(0, 30)}..." -> "${translated.substring(0, 30)}..."`);
            node.nodeValue = node.nodeValue.replace(text, translated);
        });

        // Also translate title
        if (document.title.includes('工作站组装指南')) {
            document.title = 'Tinybox DIY Guide - 6x RX 7900 XTX AI Workstation Assembly Guide';
        }

        // Translate meta description
        const metaDesc = document.querySelector('meta[name="description"]');
        if (metaDesc && TRANSLATIONS[metaDesc.content]) {
            metaDesc.content = TRANSLATIONS[metaDesc.content];
        }
    }

    /**
     * Observe DOM changes to handle React updates
     */
    let translateTimeout = null;
    const observer = new MutationObserver((mutations) => {
        // Debounce translations
        if (translateTimeout) clearTimeout(translateTimeout);
        translateTimeout = setTimeout(() => {
            translatePage(document.body);
        }, 100);
    });

    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Initial translation
    function init() {
        if (DEBUG) console.log('[TinyBox i18n] Initializing...');
        translatePage(document.body);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Re-translate when language changes
    window.addEventListener('i18n:ready', () => {
        if (DEBUG) console.log('[TinyBox i18n] Language changed, re-translating...');
        translatePage(document.body);
    });

    // Also listen for storage changes (language switch)
    window.addEventListener('storage', (e) => {
        if (e.key === 'lingkong-lang') {
            if (DEBUG) console.log('[TinyBox i18n] Language storage changed');
            // Need to reload for Chinese since we only have zh->en translations
            if (e.newValue === 'zh') {
                location.reload();
            } else {
                translatePage(document.body);
            }
        }
    });

})();
