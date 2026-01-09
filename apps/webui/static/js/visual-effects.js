/**
 * visual-effects.js
 * Implements a defined, "SDF-like" fluid gradient background effect.
 * Uses soft, volumetric orbs with composite blending to create a premium, Apple-style mesh gradient.
 */

(function () {
    // Only run if canvas doesn't exist
    if (document.getElementById('bg-canvas')) return;

    const canvas = document.createElement('canvas');
    canvas.id = 'bg-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '0'; // Put on top of background
    // Use overlay or hard-light for vivid integration
    canvas.style.mixBlendMode = 'overlay';
    canvas.style.willChange = 'transform';
    document.body.appendChild(canvas);

    const ctx = canvas.getContext('2d', { alpha: true });
    let width, height;
    let orbs = [];

    // Configuration
    const ORB_COUNT = 8;

    // Vibrant Palette for SDF effect
    const THEME_COLORS = [
        { r: 99, g: 102, b: 241 },  // Indigo 500
        { r: 139, g: 92, b: 246 }, // Violet 500
        { r: 59, g: 130, b: 246 },  // Blue 500
        { r: 168, g: 85, b: 247 }, // Purple 500
        { r: 6, g: 182, b: 212 },  // Cyan 500
        { r: 236, g: 72, b: 153 }, // Pink 500
    ];

    let mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    let targetMouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };

    // Resize handler
    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width;
        canvas.height = height;
        initOrbs();
    }

    // Orb class
    class Orb {
        constructor() {
            this.init();
        }

        init() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            // Large radius for SDF feel
            this.radius = Math.min(width, height) * (0.3 + Math.random() * 0.3);
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.color = THEME_COLORS[Math.floor(Math.random() * THEME_COLORS.length)];
            this.phase = Math.random() * Math.PI * 2;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;
            this.phase += 0.005;

            // Soft bounce
            if (this.x < -this.radius) this.vx += 0.01;
            if (this.x > width + this.radius) this.vx -= 0.01;
            if (this.y < -this.radius) this.vy += 0.01;
            if (this.y > height + this.radius) this.vy -= 0.01;

            // Mouse interaction
            const dx = mouse.x - this.x;
            const dy = mouse.y - this.y;
            this.vx += dx * 0.00001;
            this.vy += dy * 0.00001;

            this.vx *= 0.98;
            this.vy *= 0.98;
        }

        draw(ctx, isDark) {
            // High visibility
            const opacity = isDark ? 0.35 : 0.6;

            const gradient = ctx.createRadialGradient(
                this.x, this.y, 0,
                this.x, this.y, this.radius * (Math.sin(this.phase) * 0.1 + 1)
            );

            const r = this.color.r;
            const g = this.color.g;
            const b = this.color.b;

            gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${opacity})`);
            gradient.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, ${opacity * 0.5})`);
            gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius * 2, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function initOrbs() {
        orbs = [];
        for (let i = 0; i < ORB_COUNT; i++) {
            orbs.push(new Orb());
        }
    }

    function animate() {
        mouse.x += (targetMouse.x - mouse.x) * 0.05;
        mouse.y += (targetMouse.y - mouse.y) * 0.05;

        // Check theme
        const isDark = document.documentElement.getAttribute('data-theme') !== 'light';

        // Clear previous frame
        ctx.clearRect(0, 0, width, height);

        // Update blending for drawing context
        canvas.style.mixBlendMode = isDark ? 'screen' : 'overlay';
        ctx.globalCompositeOperation = isDark ? 'screen' : 'multiply';

        for (const orb of orbs) {
            orb.update();
            orb.draw(ctx, isDark);
        }

        requestAnimationFrame(animate);
    }

    // Event listeners
    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', (e) => {
        targetMouse.x = e.clientX;
        targetMouse.y = e.clientY;
    });

    resize();
    animate();

})();
