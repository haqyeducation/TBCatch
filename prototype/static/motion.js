'use strict';
// Choreography adapted from the supplied 8B template; measurement controls stay unpinned.
gsap.registerPlugin(ScrollTrigger, SplitText);
const queueMotion = fn => fn();
const adjacent=(a,b)=>document.getElementById(a).nextElementSibling?.id===b;
const lockWords=target=>{const split=SplitText.create(target,{type:'words',aria:'auto'});return {words:split.words,revert:()=>split.revert()};};
const effects={embolden:(targets,{from=200,stagger=.06,duration=1.1}={})=>{const rest=targets.map(el=>getComputedStyle(el).fontWeight);return gsap.timeline().fromTo(targets,{fontWeight:from},{fontWeight:i=>rest[i],duration,stagger,ease:'power2.out'});}};

gsap.registerPlugin(ScrollTrigger, SplitText);
ScrollTrigger.config({ autoRefreshEvents: 'visibilitychange,DOMContentLoaded,resize' });
function refreshScroll() {
  const bar = document.querySelector('#nav .nav__bar');
  const offset = bar ? bar.offsetHeight : 0;
  document.documentElement.style.scrollPaddingTop = offset + 'px';
  if (window.lenis && typeof window.lenis.options.anchors === 'object') window.lenis.options.anchors.offset = -offset;
  ScrollTrigger.sort();
  ScrollTrigger.refresh();
}
gsap.matchMedia().add('(prefers-reduced-motion: no-preference)', () => {
  const bar = document.querySelector('#nav .nav__bar');
  const lenis = window.lenis = new Lenis({ lerp: .075, wheelMultiplier: .9, anchors: { offset: -(bar ? bar.offsetHeight : 0) } });
  const tick = time => lenis.raf(time * 1000);
  lenis.on('scroll', ScrollTrigger.update);
  gsap.ticker.add(tick);
  gsap.ticker.lagSmoothing(0);
  return () => { gsap.ticker.remove(tick); gsap.ticker.lagSmoothing(500, 33); lenis.destroy(); window.lenis = null; };
});


queueMotion(() => {
  gsap.matchMedia().add('(prefers-reduced-motion: no-preference)', () => {
    if (!adjacent('hero', 'manifesto')) return;
    const hero = document.getElementById('hero');
    const q = selector => hero.querySelector(selector);
    const dome = q('.hero__dome'), slot = q('.hero__slot');
    const g = {};
    const measure = () => { g.W = hero.clientWidth; g.H = dome.offsetHeight; g.w = slot.offsetWidth; g.h = slot.offsetHeight; };
    measure();
    const from = () => `inset(calc(100% - ${g.h}px) calc(50% - ${g.w / 2}px) 0px calc(50% - ${g.w / 2}px) round ${g.w / 2}px ${g.w / 2}px 0px 0px / ${g.w / 2}px ${g.w / 2}px 0px 0px)`;
    const to = () => `inset(calc(0% - 0px) calc(0% - 0px) 0px calc(0% - 0px) round ${g.W / 2}px ${g.W / 2}px 0px 0px / ${g.H}px ${g.H}px 0px 0px)`;
    const phone = () => g.W < 720;
    gsap.timeline({ defaults: { duration: 1 }, scrollTrigger: { id: 'hero:exit', trigger: hero, start: 'bottom bottom', end: 'bottom 30%', scrub: true, invalidateOnRefresh: true, onRefreshInit: measure } })
      .fromTo(dome, { clipPath: from }, { clipPath: to, ease: 'none' }, 0)
      .fromTo(q('.hero__word--soft'), { xPercent: 0, yPercent: 0, scale: 1 }, { xPercent: () => phone() ? -8 : -12, yPercent: -16, scale: .84, ease: 'power1.in' }, 0)
      .fromTo(q('.hero__word--life'), { xPercent: 0, yPercent: 0, scale: 1 }, { xPercent: () => phone() ? 8 : 12, yPercent: -16, scale: .84, ease: 'power1.in' }, 0)
      .fromTo(q('.hero__tag'), { y: 0, opacity: 1 }, { y: -24, opacity: 0, ease: 'power1.in' }, 0)
      .fromTo(q('.hero__group--l'), { x: 0, opacity: 1 }, { x: () => -g.W * .3, opacity: 0, ease: 'power1.in' }, 0)
      .fromTo(q('.hero__group--r'), { x: 0, opacity: 1 }, { x: () => g.W * .3, opacity: 0, ease: 'power1.in' }, 0)
      .fromTo(q('.hero__photo-frame'), { x: 0, y: 0, rotation: 0 }, { x: () => phone() ? g.W * .18 : g.W * .07, y: () => phone() ? -40 : -70, rotation: 6, ease: 'power1.in' }, 0)
      .fromTo(q('.hero__lede-inner'), { x: 0, y: 0 }, { x: () => phone() ? 0 : -g.W * .05, y: () => phone() ? -40 : -60, ease: 'power1.in' }, 0)
      .fromTo(q('.hero__lede-inner'), { opacity: 1 }, { opacity: () => phone() ? 1 : 0, ease: 'power1.in', duration: .7 }, 0);
  });
}, { first: true, id: 'scene:hero' });


queueMotion(() => {
  gsap.matchMedia().add('(prefers-reduced-motion: no-preference)', () => {
    const section = document.getElementById('manifesto');
    const ground = section.querySelector('.manifesto__ground');
    if (adjacent('hero', 'manifesto')) {
      const slot = document.querySelector('#hero .hero__slot');
      let half = slot.offsetWidth / 2;
      gsap.timeline({ scrollTrigger: { id: 'manifesto:entry', trigger: section, start: 'top bottom', end: 'top 30%', scrub: true, invalidateOnRefresh: true, onRefreshInit: () => { half = slot.offsetWidth / 2; } } })
        .fromTo(ground, { clipPath: () => `inset(0px calc(50% - ${half}px) 0px calc(50% - ${half}px))` }, { clipPath: 'inset(0px calc(0% - 0px) 0px calc(0% - 0px))', ease: 'none', duration: 1 }, 0)
        .fromTo(section.querySelector('.manifesto__inner'), { y: () => innerHeight * .16 }, { y: 0, ease: 'none', duration: 1 }, 0)
        .fromTo(section.querySelector('.manifesto__inner'), { opacity: 0 }, { opacity: 1, ease: 'power1.out', duration: .2 }, .8);
    }
    const text = section.querySelector('.manifesto__text');
    const lock = lockWords(text);
    const bold = lock.words.filter(word => word.closest('b'));
    const read = gsap.timeline({ scrollTrigger: { id: 'manifesto:read', trigger: section, start: () => `top+=${text.offsetTop}px 80%`, end: () => `top+=${text.offsetTop + text.offsetHeight}px 45%`, scrub: .6, invalidateOnRefresh: true } });
    const plain = getComputedStyle(section).color, accent = bold.length ? getComputedStyle(bold[0]).color : plain;
    read.add(effects.embolden(lock.words, { from: 200, stagger: .14, duration: 1 }), 0);
    read.fromTo(bold, { color: plain }, { color: accent, duration: .8, stagger: (i, word) => lock.words.indexOf(word) * .14 - lock.words.indexOf(bold[0]) * .14, ease: 'none' }, lock.words.indexOf(bold[0]) * .14 + .3);
    return () => lock.revert();
  });
}, { id: 'scene:manifesto' });


queueMotion(() => {
  const section = document.getElementById('week');
  const days = [...section.querySelectorAll('.week__day')];
  gsap.matchMedia().add({ desk: '(min-width: 900px)', phone: '(max-width: 899px)', motion: '(prefers-reduced-motion: no-preference)' }, context => {
    if (!context.conditions.motion) return;
    const desk = context.conditions.desk;
    const parts = days.map(day => ({ frame: day.querySelector('.week__frame'), swatch: day.querySelector('.week__swatch'), img: day.querySelector('.week__art'), text: day.querySelectorAll('.week__title, .week__note') }));
    let sinks = [];
    const measure = () => { sinks = parts.map(part => part.frame.offsetHeight - part.frame.offsetWidth * (desk ? .85 : .95)); };
    measure();
    const entrances=[{scale:1.2,rotation:0,xPercent:0},{scale:1.08,rotation:-5,xPercent:-8},{scale:1,rotation:0,xPercent:22}];
    const step = (timeline, part, at) => timeline
      .fromTo(part.frame, { y: () => sinks[parts.indexOf(part)] }, { y: 0, duration: 1, ease: 'power2.inOut' }, at)
      .fromTo(part.swatch, { y: 0, yPercent: 0 }, { y: 0, yPercent: 101, duration: .8, ease: 'power2.inOut' }, at + .6)
      .fromTo(part.img, entrances[parts.indexOf(part)], { scale: 1, rotation:0, xPercent:0, duration: 1.1, ease: 'power2.out' }, at + .55)
      .fromTo(part.text, { y: 18, opacity: 0 }, { y: 0, opacity: 1, duration: .5, stagger: .1, ease: 'power2.out' }, at + .9);
    if (desk) {
      const timeline = gsap.timeline({ scrollTrigger: { id: 'week', trigger: section, start: 'top top', end: () => '+=' + innerHeight * 1.6, pin: true, scrub: .5, invalidateOnRefresh: true, onRefreshInit: measure } });
      parts.forEach((part, i) => step(timeline, part, i * .3));
    } else {
      parts.forEach((part, i) => step(gsap.timeline({ scrollTrigger: { id: `week:day-${i + 1}`, trigger: days[i], start: 'top 95%', end: 'top 50%', scrub: .5, invalidateOnRefresh: true, onRefreshInit: i ? undefined : measure } }), part, 0));
    }
  });
}, { id: 'scene:week' });

const menu=document.getElementById('mobile-menu'),toggle=document.getElementById('menu-toggle');
function closeMenu(){menu.hidden=true;toggle.setAttribute('aria-expanded','false');document.body.style.overflow='';document.getElementById('main').inert=false;window.lenis?.start();toggle.focus({preventScroll:true});}
toggle.addEventListener('click',()=>{menu.hidden=false;toggle.setAttribute('aria-expanded','true');document.body.style.overflow='hidden';document.getElementById('main').inert=true;window.lenis?.stop();document.getElementById('menu-close').focus();});
document.getElementById('menu-close').addEventListener('click',closeMenu);
menu.querySelectorAll('a').forEach(a=>a.addEventListener('click',closeMenu));
document.addEventListener('keydown',e=>{if(menu.hidden)return;if(e.key==='Escape')closeMenu();if(e.key==='Tab'){const items=[...menu.querySelectorAll('a,button')];const first=items[0],last=items.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}});
document.querySelectorAll('a[href="#check"]').forEach(a=>a.addEventListener('click',()=>{document.getElementById('check-title').focus({preventScroll:true});}));
const visibility=new IntersectionObserver(entries=>entries.forEach(e=>e.target.toggleAttribute('data-paused',!e.isIntersecting)));
visibility.observe(document.getElementById('hero'));
document.querySelectorAll('.illustration').forEach(el=>visibility.observe(el));
document.fonts.ready.then(refreshScroll);
