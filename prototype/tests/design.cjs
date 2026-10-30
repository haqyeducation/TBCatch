/* Real browser checks for layout, motion, assets and screening copy. */
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert/strict');
let pw;try{pw=require('playwright');}catch{pw=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));}
const BASE=process.env.BASE_URL||'https://localhost:8443',OUT=path.resolve(__dirname,'../test-results');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
 const browser=await pw.chromium.launch({channel:'chrome',headless:true});const checks=[];
 try{
  for(const width of [1440,768,375]){
   const ctx=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width,height:width===375?812:1000}});
   const page=await ctx.newPage(),errors=[],failed=[];page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)failed.push(r.url());});
   await page.goto(BASE);await page.evaluate(()=>document.fonts.ready);await wait(1400);
   assert.equal(await page.title(),'TBCatch — Initial TB screening');
   assert.equal(await page.locator('html').getAttribute('lang'),'en');
   assert.equal(await page.getByText(/Ready when you are|At your pace|A calmer first step/i).count(),0);
   assert.equal(await page.locator('.nav__logo svg.brand-spark').count(),1);
   assert.equal(await page.locator('.footer-brand svg.brand-spark').count(),1);
   assert(!(await page.locator('body').innerText()).includes('\u2733'));
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   await page.screenshot({path:path.join(OUT,`tbcatch-${width}-hero.png`)});
   const before=await page.locator('.hero__dome').evaluate(el=>getComputedStyle(el).clipPath);
   await page.evaluate(()=>window.lenis.scrollTo(document.getElementById('hero').offsetHeight-80,{immediate:true}));await wait(900);
   const after=await page.locator('.hero__dome').evaluate(el=>getComputedStyle(el).clipPath);assert.notEqual(before,after);
   await page.evaluate(()=>window.lenis.scrollTo(document.getElementById('manifesto'),{immediate:true}));await wait(900);
   assert(Number(await page.locator('.manifesto__inner').evaluate(el=>getComputedStyle(el).opacity))>.9);
   await page.screenshot({path:path.join(OUT,`tbcatch-${width}-manifesto.png`)});
   await page.evaluate(()=>{const t=ScrollTrigger.getById('week');window.lenis.scrollTo(t?t.end-10:document.getElementById('week').offsetTop+220,{immediate:true});});await wait(1200);
   await page.screenshot({path:path.join(OUT,`tbcatch-${width}-steps.png`)});
   if(width===375){
    await page.locator('#menu-toggle').click();assert(await page.locator('#mobile-menu').isVisible());await page.keyboard.press('Escape');assert(!(await page.locator('#mobile-menu').isVisible()));
    await page.locator('#menu-toggle').click();await page.locator('#mobile-menu a[href="#check"]').click();assert(!(await page.locator('#mobile-menu').isVisible()));
   }else await page.locator('.nav__cta').click();
   await wait(1400);assert(await page.locator('#check-title').evaluate(el=>{const r=el.getBoundingClientRect();return r.top>=0&&r.top<innerHeight;}));
   await page.screenshot({path:path.join(OUT,`tbcatch-${width}-check.png`)});
   assert(await page.locator('#start-pulse').isVisible());assert(await page.locator('#analyze-button').isDisabled());
   const images=await page.locator('img').evaluateAll(imgs=>imgs.map(i=>({src:i.getAttribute('src'),ok:i.complete&&i.naturalWidth>0})));
   assert(images.every(i=>i.ok&&i.src.endsWith('-cartoon.webp')),JSON.stringify(images));
   assert.equal(images.length,5);assert.equal(new Set(images.map(i=>i.src)).size,5);
   const animations=await page.locator('.illustration img').evaluateAll(imgs=>imgs.map(i=>getComputedStyle(i).animationName));assert.equal(new Set(animations).size,5,JSON.stringify(animations));
   assert.match(await page.locator('.grip-instruction figcaption').innerText(),/index finger behind the phone/);
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
   checks.push({width,passed:true,distinct_cartoon_assets:5,distinct_animations:5,svg_brand_mark:true,informative_copy:true,hero_dome_motion:true,manifesto_reveal:true,clear_primary_path:true,no_console_errors:true});await ctx.close();
  }
  const ctx=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width:375,height:812},reducedMotion:'reduce'});const page=await ctx.newPage();await page.goto(BASE);await wait(500);
  assert.equal(await page.evaluate(()=>ScrollTrigger.getAll().length),0);
  assert.equal(await page.locator('.manifesto__inner').evaluate(el=>getComputedStyle(el).opacity),'1');
  assert.equal(await page.locator('.week__title').first().evaluate(el=>getComputedStyle(el).opacity),'1');
  assert((await page.locator('.illustration img').evaluateAll(imgs=>imgs.map(i=>getComputedStyle(i).animationName))).every(n=>n==='none'));
  await page.locator('.nav__cta').click();await wait(100);assert(await page.locator('#check-title').evaluate(el=>{const r=el.getBoundingClientRect();return r.top>=0&&r.top<innerHeight;}));
  checks.push({reduced_motion:true,passed:true,content_visible:true});await ctx.close();
  fs.writeFileSync(path.join(OUT,'design-results.json'),JSON.stringify({checks},null,2));console.log(JSON.stringify(checks,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
