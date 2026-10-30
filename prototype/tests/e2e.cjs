/* Real browser + real inference. Simulated media devices are labeled in results. */
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert/strict');
let pw;try{pw=require('playwright');}catch{pw=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));}
const BASE=process.env.BASE_URL||'https://localhost:8443';
const OUT=path.resolve(__dirname,'../test-results');fs.mkdirSync(OUT,{recursive:true});
const checks=[];const check=(name,extra={})=>{checks.push({name,passed:true,...extra});console.log('PASS',name);};
const wait=ms=>new Promise(r=>setTimeout(r,ms));
async function main(){
 const browser=await pw.chromium.launch({channel:'chrome',headless:true,args:['--use-fake-device-for-media-stream','--use-file-for-fake-video-capture='+path.join(__dirname,'synthetic-72bpm.y4m')]});
 const context=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width:1365,height:1000},acceptDownloads:true});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(BASE);await page.getByRole('heading',{name:'TBCatch',exact:true}).waitFor();
  assert(await page.locator('#analyze-button').isDisabled());check('Initial state blocks prediction without inputs');
  await page.screenshot({path:path.join(OUT,'01-desktop.png'),fullPage:true});
  await page.getByRole('button',{name:'Measure my pulse',exact:true}).click();
  await page.locator('#pulse-error').waitFor();assert.match(await page.locator('#pulse-error').innerText(),/permission/);check('Denied camera permission has actionable error');
  await page.getByRole('button',{name:'Try the demo'}).click();await page.locator('#audio-panel').waitFor({state:'visible'});
  assert(await page.locator('#demo-banner').isVisible());assert.equal(await page.locator('#bpm').textContent(),'56');
  check('Real Welltory PPG replay yields quality-accepted demo BPM',{bpm:56,source:'recording, not physical camera'});
  const started=Date.now();while(true){const h=await (await context.request.get(BASE+'/api/health')).json();if(h.error)throw Error(h.error);if(h.ready)break;if(Date.now()-started>300000)throw Error('Model readiness timeout');await wait(1500);}
  await page.waitForFunction(()=>!document.getElementById('analyze-button').disabled,{},{timeout:10000});
  let responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/analyze')&&r.request().method()==='POST',{timeout:180000});
  await page.locator('#analyze-button').click();let response=await responsePromise;let first=await response.json();assert.equal(response.status(),200,JSON.stringify(first));assert.equal(first.model,'beats_hr');assert.equal(first.demo,true);assert(Number.isFinite(first.score));
  await page.locator('#result-panel').waitFor({state:'visible'});assert.equal(first.model_family,'BEATs + CatBoost');check('Real BEATs + CatBoost initial inference',{score:first.score,seconds:first.elapsed_seconds});
  await page.screenshot({path:path.join(OUT,'02-initial-result.png'),fullPage:true});
  if(!await page.locator('#questionnaire-details').evaluate(el=>el.open))await page.locator('#questionnaire-details summary').click();
  await page.locator('#age').fill('');let postCount=0;const counter=r=>{if(r.url().endsWith('/api/analyze'))postCount++;};page.on('request',counter);
  await page.locator('#questionnaire-button').click();await wait(200);assert.equal(postCount,0);page.off('request',counter);check('Incomplete questionnaire cannot be submitted');
  await page.locator('#age').fill('35');responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/analyze')&&r.request().method()==='POST',{timeout:180000});
  await page.locator('#questionnaire-button').click();response=await responsePromise;const second=await response.json();assert.equal(response.status(),200,JSON.stringify(second));assert.equal(second.model,'beats_mobile');assert.equal(second.stage,'questionnaire');
  assert.equal(second.threshold,2/11);assert.equal(second.category,second.score>=second.threshold?'research_screening_positive':'research_screening_negative');assert.match(await page.locator('.screening-guidance').innerText(),/Initial screening, not a diagnosis/);assert.match(await page.locator('.screening-guidance').innerText(),/negative result cannot rule out TB/);assert(second.guidance.testing.includes('sputum rapid molecular test'));check('Real questionnaire model inference',{score:second.score,seconds:second.elapsed_seconds});
  const requestBody=response.request().postDataJSON();const token=await page.locator('meta[name="api-key"]').getAttribute('content');
  const again=await context.request.post(BASE+'/api/analyze',{headers:{Origin:BASE,'X-MVP-Key':token},data:requestBody,timeout:180000});assert.equal(again.status(),200);const repeat=await again.json();assert.equal(repeat.raw_score,second.raw_score);assert.equal(repeat.score,second.score);check('Repeated singleton inference returns identical raw and calibrated score');
  const dlPromise=page.waitForEvent('download');await page.locator('#download-result').click();const dl=await dlPromise;await dl.saveAs(path.join(OUT,'demo-summary.json'));const summary=JSON.parse(fs.readFileSync(path.join(OUT,'demo-summary.json')));assert.equal(summary.demo,true);assert(!summary.audio);check('Explicit summary download, without audio payload');
  await page.screenshot({path:path.join(OUT,'03-questionnaire-result.png'),fullPage:true});
  const noToken=await context.request.post(BASE+'/api/analyze',{headers:{Origin:BASE},data:{}});assert.equal(noToken.status(),403);
  const badOrigin=await context.request.post(BASE+'/api/analyze',{headers:{Origin:'https://other.invalid','X-MVP-Key':token},data:{}});assert.equal(badOrigin.status(),403);
  const badPulse=await context.request.post(BASE+'/api/analyze',{headers:{Origin:BASE,'X-MVP-Key':token},data:{...requestBody,pulse:{bpm:0,source:'camera'}}});assert.equal(badPulse.status(),400);
  const modelFile=await context.request.get(BASE+'/research/model_bank_v5/outer_00/models/beats_hr/bundle.joblib');assert.equal(modelFile.status(),404);
  const forbiddenHost=await context.request.get(BASE+'/api/health',{headers:{Host:'untrusted.invalid'}});assert.equal(forbiddenHost.status(),421);check('API rejects unauthorized origin, missing token, invalid pulse, model-file access and untrusted host');
  await page.locator('#new-check').click();assert.equal(await page.locator('#bpm').textContent(),'—');assert.equal(await page.locator('#manual-bpm').inputValue(),'');assert.equal(await page.locator('#age').inputValue(),'');check('New examination clears prior results and answers');
  await page.locator('#pulse-panel summary').click();await page.locator('#manual-bpm').fill('0');await page.locator('#use-manual').click();assert(await page.locator('#pulse-error').isVisible());
  await page.locator('#manual-bpm').fill('78');await page.locator('#use-manual').click();await page.locator('#next-audio').click();assert(!(await page.locator('#demo-banner').isVisible()));
  await page.locator('#record-button').click();await page.locator('#audio-error').waitFor();assert.match(await page.locator('#audio-error').innerText(),/Microphone permission/);check('Denied microphone permission preserves upload path');
  await page.locator('#audio-upload').setInputFiles(path.join(__dirname,'silence.wav'));await page.waitForFunction(()=>document.getElementById('audio-preview').hidden===false);
  await page.locator('#cough-confirm').check();responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/analyze'),{timeout:10000});await page.locator('#analyze-button').click();response=await responsePromise;assert.equal(response.status(),400);await page.locator('#audio-error').waitFor();assert.match(await page.locator('#audio-error').innerText(),/quiet|empty/);check('Silence is rejected rather than assigned a disease score');
  await page.locator('#audio-upload').setInputFiles(path.join(__dirname,'cough_example.wav'));await page.waitForFunction(()=>document.getElementById('audio-filename').textContent==='cough_example.wav');await page.locator('#cough-confirm').check();
  responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/analyze'),{timeout:180000});await page.locator('#analyze-button').click();response=await responsePromise;const uploaded=await response.json();assert.equal(response.status(),200,JSON.stringify(uploaded));assert.equal(uploaded.demo,false);assert.equal(uploaded.pulse_source,'manual');check('Browser decodes uploaded WAV and completes real inference with manual-reference HR');
  assert.equal(await page.evaluate(()=>localStorage.length),0);check('No health session is persisted in localStorage');
  await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(OUT,'04-mobile-result.png'),fullPage:true});
  await page.locator('#new-check').click();await page.screenshot({path:path.join(OUT,'05-mobile-start.png'),fullPage:true});check('390px mobile layout has no horizontal overflow');
  assert.equal(errors.length,0,JSON.stringify(errors));check('No browser JavaScript exceptions');
  // Exercise capture lifecycle using Chromium's simulated hardware; not an HR accuracy test.
  const media=await browser.newContext({ignoreHTTPSErrors:true,permissions:['camera','microphone'],viewport:{width:390,height:844}});
  await media.addInitScript(()=>{window.testTracks=[];const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async c=>{const s=await original(c);window.testTracks.push(...s.getTracks());return s;};});
  const capture=await media.newPage();await capture.goto(BASE);await capture.locator('#start-pulse').click();await capture.waitForFunction(()=>window.testTracks.length>0,{},{timeout:15000});await wait(2000);assert.equal(await capture.locator('#next-audio').isVisible(),false);await capture.locator('#stop-pulse').click();await capture.waitForFunction(()=>window.testTracks.every(t=>t.readyState==='ended'));check('Simulated camera starts/stops and insufficient capture is not accepted as HR');
  await capture.locator('#pulse-panel summary').click();await capture.locator('#manual-bpm').fill('80');await capture.locator('#use-manual').click();await capture.locator('#next-audio').click();await capture.locator('#record-button').click();await capture.waitForFunction(()=>document.querySelector('.record-zone').classList.contains('recording'),{},{timeout:10000});await wait(2000);await capture.locator('#stop-record').click();await capture.waitForFunction(()=>document.getElementById('audio-preview').hidden===false,{},{timeout:15000});assert(!(await capture.locator('#cough-confirm').isChecked()));assert(await capture.evaluate(()=>window.testTracks.every(t=>t.readyState==='ended')));check('Simulated microphone recording decodes successfully and releases its stream');
  await capture.locator('[data-step="pulse"]').click();await capture.locator('#start-pulse').click();
  await capture.locator('#next-audio').waitFor({state:'visible',timeout:65000});
  const cameraBpm=Number(await capture.locator('#bpm').textContent());assert(cameraBpm>=69&&cameraBpm<=75,`Synthetic camera returned ${cameraBpm} BPM`);
  assert.match(await capture.locator('#summary-pulse').textContent(),/camera/);assert(await capture.evaluate(()=>window.testTracks.every(t=>t.readyState==='ended')));
  check('Full camera measurement accepts a stable synthetic 72 BPM video and releases camera',{estimated_bpm:cameraBpm,source:'synthetic video, not physical phone'});
  await media.close();
  fs.writeFileSync(path.join(OUT,'playwright-results.json'),JSON.stringify({base_url:BASE,browser:'Chrome via Playwright',browser_version:browser.version(),model_family:'BEATs + CatBoost',checks,hardware_validation:false,real_model_inference:true,limitations:['Camera and microphone use simulated Chromium devices; camera happy path uses synthetic 72 BPM video','No clinical validation or physical-phone accuracy test']},null,2));
  console.log(`${checks.length} checks passed.`);
 }catch(e){await page.screenshot({path:path.join(OUT,'failure.png'),fullPage:true}).catch(()=>{});fs.writeFileSync(path.join(OUT,'playwright-failure.json'),JSON.stringify({checks,error:e.stack,browserErrors:errors},null,2));throw e;}
 finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
