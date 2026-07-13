// Sequence carousel switcher for DexYCB / HOT3D results.
// Each [data-tabs] group is independent: tabs jump, arrows step with wrap-around,
// the track slides with an eased transition, and only the active panel's videos play.
document.querySelectorAll('[data-tabs]').forEach(group => {
  const tabs = Array.from(group.querySelectorAll('.seq-tab'));
  const panels = Array.from(group.querySelectorAll('.seq-panel'));
  const track = group.querySelector('.seq-track');
  const prevBtn = group.querySelector('.seq-arrow-prev');
  const nextBtn = group.querySelector('.seq-arrow-next');
  const count = panels.length;
  if (!count || !track) return;
  let current = 0;

  const playAll = (panel) => panel.querySelectorAll('video').forEach(v => v.play().catch(() => {}));
  const pauseAll = (panel) => panel.querySelectorAll('video').forEach(v => v.pause());

  function go(index) {
    current = ((index % count) + count) % count; // wrap-around (handles negatives)
    track.style.transform = `translateX(-${current * 100}%)`;
    tabs.forEach((t, i) => t.classList.toggle('is-active', i === current));
    panels.forEach((p, i) => { i === current ? playAll(p) : pauseAll(p); });
  }

  tabs.forEach((t, i) => t.addEventListener('click', () => go(i)));
  if (prevBtn) prevBtn.addEventListener('click', () => go(current - 1));
  if (nextBtn) nextBtn.addEventListener('click', () => go(current + 1));

  // init: slide to first, play it, pause the rest
  go(0);
});
