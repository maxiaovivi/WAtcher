/* WAtcher project page — behaviour.
   No dependencies. Each feature runs in its own guarded block, so a failure in one never stops the others. */
(() => {
  'use strict';
  const root = document.documentElement;
  root.classList.add('js');
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const block = (name, fn) => { try { fn(); } catch (e) { if (window.console) console.error('[' + name + ']', e); } };
  const tryPlay = (v) => { const p = v.play(); if (p && p.catch) p.catch(() => { /* autoplay blocked: poster stays */ }); };
  const loadVideo = (v) => {
    if (v.dataset.loaded || !v.dataset.src) return;
    v.dataset.loaded = '1';
    v.src = v.dataset.src;
    v.load();
  };

  /* ── Theme toggle (light / dark, remembered per browser) ── */
  block('theme', () => {
    const KEY = 'watcher-theme';
    const btn = $('#theme-btn');
    if (!btn) return;
    const isDark = () => (root.dataset.theme
      ? root.dataset.theme === 'dark'
      : !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches));
    const label = () => {
      const dark = isDark();
      btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
      btn.setAttribute('title', dark ? 'Light theme' : 'Dark theme');
    };
    btn.addEventListener('click', () => {
      const next = isDark() ? 'light' : 'dark';
      root.dataset.theme = next;
      try { localStorage.setItem(KEY, next); } catch (_) { /* storage unavailable */ }
      label();
    });
    label();
  });

  /* ── Reveal on scroll ── */
  block('reveal', () => {
    const els = $$('.reveal');
    if ('IntersectionObserver' in window && !reduceMotion) {
      const io = new IntersectionObserver((entries) => entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); }
      }), { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });
      els.forEach((el) => io.observe(el));
    } else {
      els.forEach((el) => el.classList.add('visible'));
    }
  });

  /* ── Scrollspy for the header nav ── */
  block('scrollspy', () => {
    const links = $$('.header-nav a[href^="#"]');
    const targets = links.map((a) => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
    if (!targets.length || !('IntersectionObserver' in window)) return;
    const seen = new Map();
    let current = null;
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((e) => seen.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0));
      let best = null; let bestRatio = 0;
      targets.forEach((t) => { const r = seen.get(t.id) || 0; if (r > bestRatio) { bestRatio = r; best = t.id; } });
      if (best === current) return;
      current = best;
      links.forEach((a) => {
        const on = !!best && a.getAttribute('href') === '#' + best;
        a.classList.toggle('active', on);
        if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
        const nav = a.parentElement;
        if (on && nav && nav.scrollWidth > nav.clientWidth) {
          nav.scrollTo({ left: a.offsetLeft - 12, behavior: reduceMotion ? 'auto' : 'smooth' });
        }
      });
    }, { threshold: [0, 0.1, 0.25, 0.5], rootMargin: '-80px 0px -45% 0px' });
    targets.forEach((t) => spy.observe(t));
  });

  /* ── Copy BibTeX ── */
  block('bibtex', () => {
    const btn = $('#copy-btn');
    const code = $('#bibtex');
    if (!btn || !code) return;
    btn.addEventListener('click', () => {
      const text = code.innerText;
      const restore = () => { btn.textContent = 'Copy'; btn.classList.remove('copied'); };
      const done = () => { btn.textContent = 'Copied!'; btn.classList.add('copied'); setTimeout(restore, 2000); };
      const fail = () => { btn.textContent = 'Copy failed'; setTimeout(restore, 2000); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(fail);
      } else {
        const ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (_) { fail(); }
        document.body.removeChild(ta);
      }
    });
  });

  /* ── Lazy, in-view autoplay for muted loop videos ──
     <video data-src="..." muted playsinline preload="none" poster="..." data-autoplay> */
  block('lazy-video', () => {
    const vids = $$('video[data-src]');
    if (!('IntersectionObserver' in window)) {
      vids.forEach((v) => { loadVideo(v); if (v.hasAttribute('data-autoplay') && !reduceMotion) tryPlay(v); });
      return;
    }
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      const v = e.target;
      if (e.isIntersecting) {
        loadVideo(v);
        const userPaused = v.dataset.userPaused === '1' || !!v.closest('[data-paused="true"]');
        if (v.hasAttribute('data-autoplay') && !reduceMotion && !userPaused) tryPlay(v);
      } else if (!v.paused && v.hasAttribute('data-autoplay')) {
        v.pause();
      }
    }), { rootMargin: '300px 0px 300px 0px', threshold: 0.01 });
    vids.forEach((v) => io.observe(v));
  });

  /* ── Pause control for every looping clip outside the demo wall (WCAG 2.2.2) ──
     Click, Enter or Space toggles play/pause. With reduced motion the native controls are shown instead. */
  block('loop-controls', () => {
    $$('video[data-autoplay]').forEach((v) => {
      if (v.closest('.duo-grid')) return;
      if (reduceMotion) { v.controls = true; v.setAttribute('preload', 'none'); return; }
      v.classList.add('loop-ctl');
      v.tabIndex = 0;
      v.setAttribute('role', 'button');
      const base = v.getAttribute('aria-label') || 'video';
      const media = v.parentElement;
      const badge = document.createElement('span');
      badge.className = 'loop-state';
      badge.setAttribute('aria-hidden', 'true');
      if (media) media.appendChild(badge);
      const sync = () => {
        const paused = v.dataset.userPaused === '1';
        v.setAttribute('aria-label', (paused ? 'Play: ' : 'Pause: ') + base);
        v.setAttribute('aria-pressed', String(paused));
        badge.textContent = paused ? '▶' : '❚❚';
        badge.classList.add('show');
        clearTimeout(v._badgeT);
        v._badgeT = setTimeout(() => { if (!paused) badge.classList.remove('show'); }, 700);
      };
      const toggle = () => {
        loadVideo(v);
        if (v.dataset.userPaused === '1' || v.paused) { v.dataset.userPaused = '0'; tryPlay(v); }
        else { v.dataset.userPaused = '1'; v.pause(); }
        sync();
      };
      v.addEventListener('click', toggle);
      v.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
      v.setAttribute('aria-label', 'Pause: ' + base);
      v.setAttribute('aria-pressed', 'false');
    });
  });

  /* ── Paired wall: no guard vs with Safe-Guard ──
     Each .duo-tile holds video.v-danger + video.v-guarded that start from the same state.
     The longer clip is the master; the other follows it and holds its last frame. */
  block('duo-wall', () => {
    $$('.duo-grid').forEach((grid) => {
      const section = grid.closest('[data-duo]') || grid.parentElement;
      const buttons = $$('.seg [data-mode]', section);
      const pauseBtn = $('[data-duo-pause]', section);
      const restartBtn = $('[data-duo-restart]', section);

      function updateFlag(tile) {
        const flag = $('.duo-flag .txt', tile);
        if (!flag) return;
        if (!tile.classList.contains('diverged')) flag.textContent = 'same start';
        else flag.textContent = grid.dataset.mode === 'danger' ? (tile.dataset.hazardLabel || 'hazard') : (tile.dataset.guardLabel || 'retreat');
      }
      const setMode = (mode) => {
        grid.dataset.mode = mode;
        buttons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
        $$('.duo-tile', grid).forEach(updateFlag);
      };
      buttons.forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));

      if (reduceMotion) {
        grid.dataset.paused = 'true';
        if (pauseBtn) pauseBtn.textContent = 'Play all';
      }
      if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
          const wasPaused = grid.dataset.paused === 'true';
          grid.dataset.paused = wasPaused ? 'false' : 'true';
          pauseBtn.textContent = wasPaused ? 'Pause all' : 'Play all';
          $$('video', grid).forEach((v) => { if (wasPaused) { loadVideo(v); tryPlay(v); } else { v.pause(); } });
        });
      }
      if (restartBtn) {
        restartBtn.addEventListener('click', () => {
          $$('video', grid).forEach((v) => {
            loadVideo(v);
            try { v.currentTime = 0; } catch (_) { /* not seekable yet */ }
            if (grid.dataset.paused !== 'true') tryPlay(v);
          });
        });
      }

      $$('.duo-tile', grid).forEach((tile) => {
        const a = $('video.v-danger', tile);
        const b = $('video.v-guarded', tile);
        if (!a || !b) return;
        const div = parseFloat(tile.dataset.div || 'NaN');
        const master = parseFloat(tile.dataset.durDanger || '0') >= parseFloat(tile.dataset.durGuarded || '0') ? a : b;
        const follower = master === a ? b : a;
        master.loop = true; follower.loop = false;
        let last = 0;
        master.addEventListener('timeupdate', () => {
          const t = master.currentTime;
          if (t + 0.25 < last) {
            try { follower.currentTime = 0; } catch (_) { /* ignore */ }
            if (grid.dataset.paused !== 'true') tryPlay(follower);
          } else if (!follower.ended && Math.abs(follower.currentTime - t) > 0.25 && t < (follower.duration || Infinity) - 0.1) {
            try { follower.currentTime = t; } catch (_) { /* ignore */ }
          }
          last = t;
          const now = Number.isFinite(div) && t >= div;
          if (tile.classList.contains('diverged') !== now) { tile.classList.toggle('diverged', now); updateFlag(tile); }
        });
        master.addEventListener('play', () => { if (follower.paused && !follower.ended) tryPlay(follower); });
        master.addEventListener('pause', () => { if (!follower.paused) follower.pause(); });
        updateFlag(tile);
      });
      setMode(grid.dataset.mode || 'danger');
    });
  });

  /* ── Film chapters ── */
  block('chapters', () => {
    const film = $('#film');
    const chaps = $$('.chap[data-t]');
    if (!film || !chaps.length) return;
    chaps.forEach((c) => c.addEventListener('click', () => {
      const t = parseFloat(c.dataset.t);
      const go = () => { try { film.currentTime = t; } catch (_) { /* ignore */ } tryPlay(film); };
      if (film.readyState >= 1) go();
      else { film.addEventListener('loadedmetadata', go, { once: true }); film.load(); }
    }));
    film.addEventListener('timeupdate', () => {
      const t = film.currentTime;
      let cur = null;
      chaps.forEach((c) => { if (t + 0.05 >= parseFloat(c.dataset.t)) cur = c; });
      chaps.forEach((c) => c.classList.toggle('on', c === cur && !film.paused));
    });
    film.addEventListener('pause', () => chaps.forEach((c) => c.classList.remove('on')));
  });

  /* ── Lightbox (figures and videos) ── */
  block('lightbox', () => {
    const lb = $('#lightbox');
    if (!lb) return;
    const media = $('.lightbox-media', lb);
    const cap = $('.lightbox-cap', lb);
    const closeBtn = $('.lightbox-close', lb);
    if (!media || !cap || !closeBtn) return;
    const outside = [$('.site-header'), $('main'), $('.site-footer'), $('.skip')].filter(Boolean);
    let returnTo = null;
    const close = () => {
      if (!lb.classList.contains('on')) return;
      lb.classList.remove('on');
      $$('video', media).forEach((v) => v.pause());
      media.textContent = '';
      document.body.style.overflow = '';
      outside.forEach((el) => { el.inert = false; el.removeAttribute('aria-hidden'); });
      if (returnTo) returnTo.focus();
    };
    const open = (node, caption, from) => {
      returnTo = from || null;
      media.textContent = '';
      media.appendChild(node);
      cap.textContent = (caption || '').replace(/\s*Click to enlarge\.?\s*$/i, '');
      lb.classList.add('on');
      document.body.style.overflow = 'hidden';
      outside.forEach((el) => { el.inert = true; el.setAttribute('aria-hidden', 'true'); });
      closeBtn.focus();
    };
    lb.addEventListener('click', (e) => { if (e.target === lb) close(); });
    closeBtn.addEventListener('click', close);
    document.addEventListener('keydown', (e) => {
      if (!lb.classList.contains('on')) return;
      if (e.key === 'Escape') close();
      if (e.key === 'Tab') {
        const f = $$('button, video[controls], a[href]', lb);
        if (!f.length) return;
        const first = f[0]; const last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });
    $$('img.zoomable').forEach((img) => {
      img.tabIndex = 0;
      img.setAttribute('role', 'button');
      img.setAttribute('aria-label', 'Enlarge figure: ' + (img.alt || '').slice(0, 120));
      const go = () => {
        const big = new Image();
        big.src = img.dataset.full || img.currentSrc || img.src;
        big.alt = img.alt;
        const fig = img.closest('figure');
        const fc = fig && $('figcaption', fig);
        open(big, fc ? fc.textContent.trim() : img.alt, img);
      };
      img.addEventListener('click', go);
      img.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    });
    $$('[data-expand]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const v = document.createElement('video');
        v.src = btn.dataset.expand; v.controls = true; v.loop = true; v.muted = true; v.playsInline = true;
        v.autoplay = !reduceMotion;
        v.setAttribute('aria-label', btn.dataset.caption || 'Enlarged video');
        open(v, btn.dataset.caption || '', btn);
      });
    });
  });

  /* ── Tooltip shared by charts (hover and keyboard focus; Escape hides it) ── */
  const tip = document.createElement('div');
  tip.className = 'tooltip';
  tip.setAttribute('aria-hidden', 'true');
  document.body.appendChild(tip);
  let tipOwner = null;
  const showTip = (owner, x, y, build) => {
    if (owner !== tipOwner) { tip.textContent = ''; build(tip); tipOwner = owner; }
    tip.classList.add('on');
    const r = tip.getBoundingClientRect();
    let left = x + 14; let top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  };
  const hideTip = () => { tip.classList.remove('on'); tipOwner = null; };
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideTip(); });
  const line = (parent, text, cls) => { const d = document.createElement('div'); if (cls) d.className = cls; d.textContent = text; parent.appendChild(d); return d; };

  /* ── Bar chart built from its own table view ──
     <div data-bars data-source="#table" data-min data-max data-ticks data-metric>
     rows: <tr data-name data-group [data-ours] data-v-<metric>="94.2" [data-note] [data-note-<metric>]> */
  block('bars', () => {
    $$('[data-bars]').forEach((chart) => {
      const table = $(chart.dataset.source);
      const host = $('.bars', chart);
      if (!table || !host) return;
      const rows = $$('tbody tr[data-name]', table).map((tr) => {
        const ds = tr.dataset;
        const values = {}; const notes = {};
        Object.keys(ds).forEach((k) => {
          if (/^v[A-Z]/.test(k)) values[k.slice(1).toLowerCase()] = ds[k];
          else if (/^note[A-Z]/.test(k)) notes[k.slice(4).toLowerCase()] = ds[k];
        });
        return { name: ds.name, group: ds.group || '', ours: tr.hasAttribute('data-ours'), note: ds.note || '', notes, values };
      });
      const axis = $('.bar-axis .ticks', chart);
      const min = parseFloat(chart.dataset.min || '0');
      const max = parseFloat(chart.dataset.max || '100');
      const unit = chart.dataset.unit || '%';
      const pct = (v) => Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
      if (axis) {
        (chart.dataset.ticks || '').split(',').filter(Boolean).forEach((t) => {
          const s = document.createElement('span');
          s.style.left = pct(parseFloat(t)) + '%';
          s.textContent = t;
          axis.appendChild(s);
        });
      }
      const metricBtns = $$('[data-metric]', chart);
      const render = (metric) => {
        host.textContent = '';
        hideTip();
        const label = ((chart.querySelector('[data-metric="' + metric + '"]') || {}).textContent || metric).trim();
        const shown = rows.filter((r) => r.values[metric] !== undefined && r.values[metric] !== '')
          .sort((p, q) => parseFloat(q.values[metric]) - parseFloat(p.values[metric]));
        shown.forEach((r) => {
          const v = parseFloat(r.values[metric]);
          const note = r.notes[metric] || r.note;
          const row = document.createElement('div');
          row.className = 'bar-row' + (r.ours ? ' ours' : '');
          row.tabIndex = 0;
          row.setAttribute('role', 'listitem');
          row.setAttribute('aria-label', r.name + ': ' + v.toFixed(1) + unit + (note ? '. ' + note : ''));
          const nm = document.createElement('div'); nm.className = 'bar-name'; nm.textContent = r.name; nm.title = r.name;
          if (/zero-shot/i.test(r.group)) {
            const tag = document.createElement('span'); tag.className = 'bar-tag'; tag.textContent = 'zero-shot';
            nm.appendChild(tag);
          }
          const track = document.createElement('div'); track.className = 'bar-track';
          const fill = document.createElement('div'); fill.className = 'bar-fill'; fill.style.width = reduceMotion ? pct(v) + '%' : '0%';
          track.appendChild(fill);
          const val = document.createElement('div'); val.className = 'bar-val'; val.textContent = v.toFixed(1);
          row.append(nm, track, val);
          host.appendChild(row);
          if (!reduceMotion) requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = pct(v) + '%'; }));
          const build = (t) => {
            line(t, v.toFixed(1) + unit, 'tv');
            line(t, r.name + ' · ' + label);
            if (note) line(t, note);
          };
          row.addEventListener('pointermove', (e) => showTip(row, e.clientX, e.clientY, build));
          row.addEventListener('pointerleave', hideTip);
          row.addEventListener('focus', () => { const b = row.getBoundingClientRect(); showTip(row, b.right - 60, b.top, build); });
          row.addEventListener('blur', hideTip);
        });
        const missing = $('.bars-missing', chart);
        if (missing) {
          const notRun = rows.filter((r) => !shown.includes(r) && !(r.notes[metric] || '').startsWith('Not applicable'));
          const notApplicable = rows.filter((r) => !shown.includes(r) && (r.notes[metric] || '').startsWith('Not applicable'));
          const parts = [];
          if (notRun.length) parts.push('Not run on the full split: ' + notRun.map((r) => r.name).join(', ') + '.');
          if (notApplicable.length) parts.push('Not applicable: ' + notApplicable.map((r) => r.name).join(', ') + ' (an untrained check).');
          missing.textContent = parts.join(' ');
        }
        metricBtns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.metric === metric)));
      };
      metricBtns.forEach((b) => b.addEventListener('click', () => render(b.dataset.metric)));
      render(chart.dataset.metric || (metricBtns[0] && metricBtns[0].dataset.metric));
    });
  });

  /* Redraw an SVG chart at its rendered width (1 viewBox unit = 1 CSS px), on load and on resize. */
  const responsive = (svg, draw) => {
    let lastW = 0;
    const run = () => {
      const w = Math.round(svg.getBoundingClientRect().width || (svg.parentElement ? svg.parentElement.clientWidth : 0) || 760);
      if (Math.abs(w - lastW) < 2) return;
      lastW = w;
      draw(w);
    };
    run();
    if ('ResizeObserver' in window) new ResizeObserver(run).observe(svg.parentElement || svg);
    else window.addEventListener('resize', run);
  };

  /* ── Dumbbell chart (before -> after) built from its table view ──
     rows: <tr data-name data-sub data-before data-after [data-cg] data-detail> */
  block('dumbbell', () => {
    $$('[data-dumbbell]').forEach((chart) => {
      const table = $(chart.dataset.source);
      const svg = $('svg', chart);
      if (!table || !svg) return;
      const NS = 'http://www.w3.org/2000/svg';
      const rows = $$('tbody tr[data-name]', table).map((tr) => ({
        name: tr.dataset.name,
        sub: tr.dataset.sub || '',
        before: parseFloat(tr.dataset.before),
        after: parseFloat(tr.dataset.after),
        cg: tr.dataset.cg !== undefined && tr.dataset.cg !== '' ? parseFloat(tr.dataset.cg) : null,
        detail: tr.dataset.detail || '',
      }));
      const max = parseFloat(chart.dataset.max || '80');
      const ticks = (chart.dataset.ticks || '0,20,40,60,80').split(',').map(Number);
      const draw = (W) => {
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        const narrow = W < 560;
        const rowH = narrow ? 58 : 46;
        const top = 14;
        const left = narrow ? 14 : Math.min(230, Math.round(W * 0.3));
        const right = narrow ? 22 : 40;
        const H = top + rows.length * rowH + 30;
        svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
        svg.setAttribute('width', W);
        svg.setAttribute('height', H);
        const x = (v) => left + (v / max) * (W - left - right);
        const el = (name, attrs, parent) => {
          const n = document.createElementNS(NS, name);
          Object.keys(attrs).forEach((k) => n.setAttribute(k, attrs[k]));
          (parent || svg).appendChild(n);
          return n;
        };
        ticks.forEach((t) => {
          el('line', { x1: x(t), x2: x(t), y1: top - 4, y2: H - 26, class: 'grid' });
          const tt = el('text', { x: x(t), y: H - 8, 'text-anchor': t === 0 ? 'start' : 'middle' });
          tt.textContent = t + '%';
        });
        rows.forEach((r, i) => {
          const y0 = top + i * rowH;
          const cy = narrow ? y0 + 40 : y0 + rowH / 2;
          const g = el('g', {
            class: 'row', tabindex: '0', role: 'listitem',
            'aria-label': r.name + (r.sub ? ' (' + r.sub + ')' : '') + ': ' + r.before.toFixed(1) + '% to ' + r.after.toFixed(1) + '% hazardous'
              + (r.cg !== null ? '; ContactGuard-style ' + r.cg.toFixed(1) + '%' : '') + (r.detail ? '. ' + r.detail : ''),
          });
          el('rect', { x: 0, y: y0 + 2, width: W, height: rowH - 4, rx: 8, class: 'bg' }, g);
          if (narrow) {
            const nm = el('text', { x: 8, y: y0 + 16, class: 'row-name' }, g);
            nm.textContent = r.name + (r.sub ? ' · ' + r.sub : '');
          } else {
            const nm = el('text', { x: 10, y: r.sub ? cy - 3 : cy + 4, class: 'row-name' }, g); nm.textContent = r.name;
            if (r.sub) { const sb = el('text', { x: 10, y: cy + 13 }, g); sb.textContent = r.sub; sb.setAttribute('font-size', '11.5'); }
          }
          el('line', { x1: x(r.after), x2: x(r.before), y1: cy, y2: cy, class: 'link' }, g);
          if (r.cg !== null) el('rect', { x: x(r.cg) - 5, y: cy - 5, width: 10, height: 10, transform: 'rotate(45 ' + x(r.cg) + ' ' + cy + ')', class: 'm-cg' }, g);
          el('circle', { cx: x(r.before), cy, r: 6, class: 'm-before' }, g);
          el('circle', { cx: x(r.after), cy, r: 7, class: 'm-ours' }, g);
          const vb = el('text', { x: x(r.before) + 11, y: cy + 4, class: 'val' }, g); vb.textContent = r.before.toFixed(1);
          const va = el('text', { x: x(r.after) - 11, y: cy + 4, 'text-anchor': 'end', class: 'val ours' }, g); va.textContent = r.after.toFixed(1);
          const build = (t) => {
            line(t, r.before.toFixed(1) + '% → ' + r.after.toFixed(1) + '%', 'tv');
            line(t, r.name + (r.sub ? ' · ' + r.sub : ''));
            if (r.cg !== null) line(t, 'ContactGuard-style: ' + r.cg.toFixed(1) + '%');
            if (r.detail) line(t, r.detail);
          };
          g.addEventListener('pointermove', (e) => showTip(g, e.clientX, e.clientY, build));
          g.addEventListener('pointerleave', hideTip);
          g.addEventListener('focus', () => { const b = g.getBoundingClientRect(); showTip(g, b.left + b.width * 0.6, b.top, build); });
          g.addEventListener('blur', hideTip);
        });
      };
      responsive(svg, draw);
    });
  });

  /* ── Risk gate (illustrative scores, not model outputs) ── */
  block('gate', () => {
    const gate = $('#gate');
    if (!gate) return;
    const svg = $('svg', gate);
    const slider = $('#tau', gate);
    const tauOut = $('#tau-out', gate);
    const out = $('.gate-out', gate);
    const rhoOut = $('#rho-out', gate);
    const why = $('#gate-why', gate);
    if (!svg || !slider || !tauOut || !out || !rhoOut || !why) return;
    const NS = 'http://www.w3.org/2000/svg';
    const presets = [
      { name: 'smooth approach', steps: [0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.08, 0.09, 0.10, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13], cls: 0.10 },
      { name: 'contact at step 11', steps: [0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.19, 0.26, 0.41, 0.83, 0.78, 0.62, 0.40, 0.30, 0.26], cls: 0.55 },
      { name: 'slip that builds over the chunk', steps: [0.08, 0.09, 0.11, 0.12, 0.14, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35], cls: 0.71 },
    ];
    let cur = 1;
    let W = 640;
    const H = 230;
    const draw = () => {
      const pad = { l: 34, r: 14, t: 16, b: 36 };
      const tau = parseFloat(slider.value);
      const p = presets[cur];
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      const n = p.steps.length + 1;
      const bw = (W - pad.l - pad.r) / n;
      const y = (v) => pad.t + (1 - v) * (H - pad.t - pad.b);
      const mk = (name, attrs) => { const e = document.createElementNS(NS, name); Object.keys(attrs).forEach((k) => e.setAttribute(k, attrs[k])); svg.appendChild(e); return e; };
      [0, 0.5, 1].forEach((t) => {
        mk('line', { x1: pad.l, x2: W - pad.r, y1: y(t), y2: y(t), stroke: 'var(--c-axis)', 'stroke-width': 1 });
        const tx = mk('text', { x: pad.l - 6, y: y(t) + 4, 'text-anchor': 'end' }); tx.textContent = t.toFixed(1);
      });
      const vals = p.steps.concat([p.cls]);
      const rho = Math.max.apply(null, vals);
      const argmax = vals.indexOf(rho);
      const every = W < 420 ? 5 : 3;
      vals.forEach((v, i) => {
        const isCls = i === vals.length - 1;
        const x0 = pad.l + i * bw + 2 + (isCls ? Math.min(6, bw * 0.2) : 0);
        const w = Math.max(2, bw - 4 - (isCls ? Math.min(6, bw * 0.2) : 0));
        const fill = i === argmax && rho >= tau ? 'var(--hazard)' : (isCls ? 'var(--c-before)' : 'var(--green-l)');
        mk('rect', { x: x0, y: y(v), width: w, height: Math.max(1, y(0) - y(v)), rx: 3, fill });
        if (i % every === 0 || isCls) {
          const lt = mk('text', { x: x0 + w / 2, y: H - 18, 'text-anchor': 'middle' });
          lt.textContent = isCls ? 'CLS' : String(i + 1);
        }
      });
      mk('line', { x1: pad.l, x2: W - pad.r, y1: y(tau), y2: y(tau), stroke: 'var(--hazard)', 'stroke-width': 1.5, 'stroke-dasharray': '6 5' });
      const tl = mk('text', { x: pad.l + 4, y: y(tau) - 6, 'text-anchor': 'start', class: 'tau' });
      tl.textContent = 'τᵣ = ' + tau.toFixed(2);
      const ax = mk('text', { x: pad.l, y: H - 2 }); ax.textContent = 'command k = 1 … 16, then the [CLS] chunk score';
      tauOut.textContent = tau.toFixed(2);
      rhoOut.textContent = rho.toFixed(2);
      const retreat = rho >= tau;
      out.className = 'gate-out ' + (retreat ? 'retreat' : 'exec');
      out.textContent = retreat ? 'ρ ≥ τᵣ → bounded retreat' : 'ρ < τᵣ → execute the first K commands';
      why.textContent = 'Candidate: ' + p.name + '. ρ is the largest of the 16 step scores and the [CLS] score; here it is '
        + (argmax === vals.length - 1 ? 'the [CLS] score, which sees the chunk as a whole.' : 'the score of step ' + (argmax + 1) + '.');
    };
    slider.addEventListener('input', draw);
    $$('[data-preset]', gate).forEach((b) => b.addEventListener('click', () => {
      cur = parseInt(b.dataset.preset, 10);
      $$('[data-preset]', gate).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      draw();
    }));
    responsive(svg, (w) => { W = Math.max(280, w); draw(); });
  });
})();
