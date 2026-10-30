'use strict';
const $=id=>document.getElementById(id);
let pulse=null,audio=null,result=null,modelReady=false,busy=false;
let ppg=null,pulseTimer=null,pulseAbort=null,pulseStarted=0,goodWindows=[],waveValues=[],pulseAttempt=0;
let recorder=null,micStream=null,recordTimer=null,recordStarted=0,recordAttempt=0,previewUrl=null,audioAttempt=0;
const apiKey=document.querySelector('meta[name="api-key"]').content;
const qualityText={no_finger:'Cover the rear camera with your index fingertip, keeping the screen facing you.',settling:'Keep your index finger still while the signal settles.',collecting:'Collecting your signal. Keep still.',saturated:'Too much light. Adjust your index finger gently.',weak_pulse:'No clear pulse yet. Check your index finger position and lighting.',motion:'Keep your hand as still as you can.',irregular:'The signal is not stable yet. Rest and try again if needed.',morphology:'The signal needs adjusting. Reposition your index finger gently.',fft_disagree:'Readings are not consistent yet. Keep still.',double_count:'Readings are not consistent yet. Keep still.',missed_beats:'Readings are not consistent yet. Keep still.'};

function error(id,message=''){ $(id).textContent=message;$(id).hidden=!message; }
function go(step){
  if(step==='audio'&&!pulse){error('pulse-error','Measure your pulse first, or try the demo.');step='pulse';}
  if(step==='result'&&!result){error('audio-error','Analyze your cough and pulse first.');step=pulse?'audio':'pulse';}
  if(step!=='audio'&&(micStream||$('stop-record').hidden===false))cancelRecording();
  for(const s of ['pulse','audio','result'])$(s+'-panel').hidden=s!==step;
  document.querySelectorAll('[data-step]').forEach(b=>{b.classList.toggle('active',b.dataset.step===step);b.setAttribute('aria-current',b.dataset.step===step?'step':'false');});
  const target=$('check');if(window.lenis)window.lenis.scrollTo(target,{offset:-75,immediate:true});else target.scrollIntoView({block:'start'});
}
function update(){
  $('analyze-button').disabled=busy||!modelReady||!pulse||!audio||!$('cough-confirm').checked;
  $('analyze-hint').textContent=!modelReady?'Waiting for the local model to be ready.':(!pulse||!audio||!$('cough-confirm').checked)?'Add your pulse, cough audio, and recording confirmation.':'Ready for analysis on the local server.';
  $('next-audio').hidden=!pulse;
  $('summary-pulse').textContent=pulse?`${pulse.bpm} BPM · ${pulse.source==='demo'?'demo':pulse.source==='manual'?'manual device reading':'camera'}`:'Not available yet';
  $('summary-audio').textContent=audio?`${audio.duration.toFixed(1)} detik · ${audio.source==='demo'?'demo':'your recording'}`:'Not available yet';
  $('demo-banner').hidden=!(pulse?.source==='demo'||audio?.source==='demo');
}
function setBusy(value){
  busy=value;$('busy').hidden=!value;
  for(const id of ['demo-button','start-pulse','use-manual','record-button','audio-upload','questionnaire-button','reset-button','new-check','download-result','manual-bpm','cough-confirm'])$(id).disabled=value;
  $('questionnaire').querySelectorAll('input,select').forEach(el=>{el.disabled=value;});update();
}
function invalidate(){result=null;$('download-result').hidden=true;$('summary-clinical').textContent='Not added yet';$('result-content').replaceChildren();$('questionnaire-details').open=false;error('result-error');}
function clearAudio(){audioAttempt++;audio=null;invalidate();if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;$('audio-preview').pause();$('audio-preview').removeAttribute('src');$('audio-preview').hidden=true;$('audio-tag').textContent='No recording yet';$('audio-tag').classList.remove('success');$('audio-filename').textContent='Your recording is processed only when you choose to analyze it.';$('cough-confirm').checked=false;update();}
function drawWave(value){
  if(Number.isFinite(value)){waveValues.push(value);if(waveValues.length>280)waveValues.shift();}
  const c=$('pulse-wave'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
  ctx.strokeStyle='#72080822';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,50);ctx.lineTo(c.width,50);ctx.stroke();
  if(waveValues.length<3)return;
  const low=Math.min(...waveValues),span=Math.max(...waveValues)-low;if(span<1e-8)return;
  ctx.strokeStyle='#a10728';ctx.lineWidth=2;ctx.beginPath();waveValues.forEach((v,i)=>{const x=i*c.width/279,y=85-(v-low)*70/span;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
}
function stopPulse(){
  pulseAttempt++;clearInterval(pulseTimer);pulseTimer=null;
  pulseAbort?.abort();pulseAbort=null;if(ppg){ppg.stop();ppg.destroy();ppg=null;}
  $('stop-pulse').hidden=true;$('start-pulse').hidden=false;
}
function acceptPulse(value){
  if(!Number.isFinite(value.bpm)||value.bpm<45||value.bpm>220)throw new Error('Pulse is outside this prototype’s measurement range (45–220 BPM).');
  invalidate();pulse={...value,bpm:Math.round(value.bpm)};$('bpm').textContent=pulse.bpm;
  $('pulse-tag').textContent=value.source==='demo'?'Demo recording':value.source==='manual'?'Device reading':'Measurement complete';
  $('pulse-tag').classList.add('success');update();
}
async function startPulse(){
  if(busy)return;stopPulse();invalidate();pulse=null;update();error('pulse-error');
  $('bpm').textContent='—';$('pulse-tag').textContent='Measuring';$('pulse-tag').classList.remove('success');$('pulse-progress').value=0;waveValues=[];drawWave();goodWindows=[];
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){error('pulse-error','Camera access needs HTTPS on your phone or localhost on the laptop. Open the HTTPS address shown by the server.');return;}
  const attempt=++pulseAttempt;const capture=new PPG.PPG({signal:{respiration:false},debug:{persistLastSession:false},motion:false});ppg=capture;pulseAbort=new AbortController();
  $('start-pulse').hidden=true;$('stop-pulse').hidden=false;$('pulse-guidance').textContent='Allow camera access, then place your index fingertip over the rear camera with the screen facing you.';
  capture.addEventListener('ready',e=>{if(attempt===pulseAttempt&&!e.detail.torchSupported)$('pulse-guidance').textContent='Flash is unavailable in this browser. Use good lighting; we will only accept a stable signal.';});
  capture.addEventListener('state',e=>{if(attempt===pulseAttempt)$('pulse-guidance').textContent=e.detail.state==='MEASURING'?'Keep still while we look for consistent readings.':qualityText[e.detail.state==='NO_FINGER'?'no_finger':'settling'];});
  capture.addEventListener('waveform',e=>{if(attempt===pulseAttempt)drawWave(e.detail.value);});
  capture.addEventListener('metrics',e=>{
    if(attempt!==pulseAttempt)return;const m=e.detail;
    if(m.quality?.good&&Number.isFinite(m.heartRate)&&m.heartRate>=45&&m.heartRate<=220){
      goodWindows.push({bpm:m.heartRate,time:performance.now()});if(goodWindows.length>3)goodWindows.shift();$('bpm').textContent=Math.round(m.heartRate);$('pulse-guidance').textContent='We have a signal. Keep still until the measurement finishes.';
    }else{goodWindows=[];$('bpm').textContent='—';$('pulse-guidance').textContent=qualityText[m.quality?.code]||'No stable signal yet. Keep still and adjust your index finger.';}
  });
  capture.addEventListener('error',e=>{if(attempt===pulseAttempt){error('pulse-error',cameraError(e.detail.error));stopPulse();$('pulse-guidance').textContent='Measurement did not finish. You can try again.';}});
  try{
    await capture.start(pulseAbort.signal);if(attempt!==pulseAttempt)return;
    pulseStarted=performance.now();pulseTimer=setInterval(()=>{
      const seconds=(performance.now()-pulseStarted)/1000;$('pulse-progress').value=Math.min(30,seconds);
      const values=goodWindows.map(v=>v.bpm);
      if(seconds>=30&&values.length===3&&performance.now()-goodWindows[2].time<7000&&Math.max(...values)-Math.min(...values)<=8){
        const bpm=[...values].sort((a,b)=>a-b)[1];stopPulse();acceptPulse({bpm,source:'camera',quality_passed:true,capture_seconds:seconds,good_windows:3});$('pulse-guidance').textContent='Measurement complete. You can move on to your cough.';
      }else if(seconds>=75){stopPulse();$('bpm').textContent='—';$('pulse-tag').textContent='Try again';error('pulse-error','We could not get enough stable readings. Adjust your index finger and try again.');}
    },250);
  }catch(e){if(attempt===pulseAttempt){stopPulse();error('pulse-error',cameraError(e));}}
}
function cameraError(e){return e?.code==='permission_denied'||e?.name==='NotAllowedError'?'Camera permission was denied. Enable camera access in your browser, then try again.':e?.name==='NotFoundError'||e?.code==='no_camera'?'No camera found. Use a phone with a rear camera.':'Camera unavailable. Close other camera apps, check permissions, and try again.';}

function bytesToBase64(buffer){const b=new Uint8Array(buffer);let s='';for(let i=0;i<b.length;i+=8192)s+=String.fromCharCode(...b.subarray(i,i+8192));return btoa(s);}
function wavFromBuffer(buffer){
  const n=buffer.length,channels=buffer.numberOfChannels,ab=new ArrayBuffer(44+n*2),v=new DataView(ab);
  const str=(i,s)=>{for(let j=0;j<s.length;j++)v.setUint8(i+j,s.charCodeAt(j));};
  str(0,'RIFF');v.setUint32(4,36+n*2,true);str(8,'WAVE');str(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,buffer.sampleRate,true);v.setUint32(28,buffer.sampleRate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);str(36,'data');v.setUint32(40,n*2,true);
  const arrays=Array.from({length:channels},(_,i)=>buffer.getChannelData(i));
  for(let i=0;i<n;i++){let x=0;for(const a of arrays)x+=a[i]/channels;x=Math.max(-1,Math.min(1,x));v.setInt16(44+i*2,x<0?x*32768:x*32767,true);}return ab;
}
function setAudio(value,blob){
  audioAttempt++;invalidate();audio=value;if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=URL.createObjectURL(blob);$('audio-preview').src=previewUrl;$('audio-preview').hidden=false;
  $('audio-filename').textContent=value.name;$('audio-tag').textContent=`${value.duration.toFixed(1)} seconds`;$('audio-tag').classList.add('success');$('record-time').textContent='Recording ready';$('cough-confirm').checked=false;update();
}
async function loadAudio(blob,name){
  const attempt=++audioAttempt;error('audio-error');if(blob.size>12_000_000)throw new Error('This file is too large. Choose a recording up to 30 seconds.');
  const C=window.AudioContext||window.webkitAudioContext;if(!C)throw new Error('Audio processing is unavailable. Use an up-to-date browser.');
  const ctx=new C();let decoded;
  try{decoded=await ctx.decodeAudioData(await blob.arrayBuffer());}catch{throw new Error('We could not read this audio format. Try a WAV or MP3, or record here.');}finally{await ctx.close();}
  if(decoded.duration<.25||decoded.duration>30)throw new Error('Recordings must last between 0.25 and 30 seconds.');
  if(attempt!==audioAttempt)return;
  const wav=wavFromBuffer(decoded);setAudio({data:bytesToBase64(wav),duration:decoded.duration,name,source:'user'},new Blob([wav],{type:'audio/wav'}));
}
function releaseMic(){clearInterval(recordTimer);recordTimer=null;micStream?.getTracks().forEach(t=>t.stop());micStream=null;$('stop-record').hidden=true;$('record-button').hidden=false;document.querySelector('.record-zone').classList.remove('recording');}
function stopRecording(){recordAttempt++;if(recorder?.state==='recording')recorder.stop();releaseMic();}
function cancelRecording(){stopRecording();recordAttempt++;}
async function startRecording(){
  if(busy)return;error('audio-error');if(!navigator.mediaDevices?.getUserMedia){error('audio-error','Microphone access needs HTTPS on your phone. You can also upload a recording.');return;}
  clearAudio();const attempt=++recordAttempt;let chunks=[];
  $('record-button').hidden=true;$('stop-record').hidden=false;$('record-time').textContent='Waiting for microphone permission…';
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    if(attempt!==recordAttempt){stream.getTracks().forEach(t=>t.stop());return;}micStream=stream;
    recorder=new MediaRecorder(stream);const thisRecorder=recorder;
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    recorder.onstop=async()=>{const blob=new Blob(chunks,{type:thisRecorder.mimeType});if(attempt+1!==recordAttempt)return;try{await loadAudio(blob,'Cough recording');}catch(e){error('audio-error',e.message);}};
    recorder.onerror=()=>{stopRecording();error('audio-error','Recording stopped unexpectedly. Check your microphone or upload audio.');};
    recorder.start();recordStarted=performance.now();document.querySelector('.record-zone').classList.add('recording');
    recordTimer=setInterval(()=>{const t=(performance.now()-recordStarted)/1000;$('record-time').textContent=`Recording · ${t.toFixed(0)} seconds`;
      if(t>=28)stopRecording();},200);
  }catch(e){releaseMic();error('audio-error',e.name==='NotAllowedError'?'Microphone permission was denied. Enable access or upload a cough recording.':'Microphone unavailable. Try again or upload a recording.');}
}

function showResult(data){
  result=data;const box=document.createElement('div');box.className='result-box'+(data.category==='research_screening_negative'?' negative':'');const h=document.createElement('h3');h.textContent=data.title;const p=document.createElement('p');p.textContent=data.message;box.append(h,p);
  const score=document.createElement('p');score.className='result-score';score.textContent=`Model score: ${(100*data.score).toFixed(1)} out of 100. This is not the probability that you have TB.`;
  const meta=document.createElement('p');meta.className='result-meta';meta.textContent=`${data.demo?'Demo data · ':''}${data.audio_seconds} seconds of audio · ${data.elapsed_seconds} seconds to process`;
  $('result-content').replaceChildren(box,score);
  if(data.stage==='questionnaire'){
    const track=document.createElement('div');track.className='score-track';track.setAttribute('aria-hidden','true');
    const fill=document.createElement('div');fill.className='score-fill';fill.style.width=(100*data.score)+'%';
    const marker=document.createElement('span');marker.className='score-marker';marker.style.left=(100*data.threshold)+'%';track.append(fill,marker);
    const note=document.createElement('p');note.className='threshold-note';note.textContent=`App screening threshold: ${(100*data.threshold).toFixed(2)} / 100. A higher threshold can miss more TB cases.`;
    $('result-content').append(track,note);
  }
  if(data.guidance){
    const advice=document.createElement('section');advice.className='screening-guidance';advice.setAttribute('aria-labelledby','guidance-title');
    const heading=document.createElement('h3');heading.id='guidance-title';heading.textContent=data.guidance.heading;advice.append(heading);
    for(const key of ['limitation','next_step','testing','symptoms']){const p=document.createElement('p');p.textContent=data.guidance[key];if(key==='next_step')p.className='next-step';advice.append(p);}
    const source=document.createElement('a');source.href=data.guidance.source_url;source.textContent=data.guidance.source_label;source.target='_blank';source.rel='noopener';advice.append(source);$('result-content').append(advice);
  }
  $('result-content').append(meta);$('result-tag').textContent=data.stage==='initial'?'Cough + pulse':'Cough + pulse + answers';$('download-result').hidden=false;
  if(data.stage==='initial')$('questionnaire-details').open=!!data.questionnaire_recommended;
  else $('summary-clinical').textContent='9 answers complete';go('result');
}
async function analyze(stage){
  if(busy||!pulse||!audio||!modelReady)return;
  if(!$('cough-confirm').checked){error('audio-error','Confirm that your recording contains coughs.');return;}
  const clinical=stage==='questionnaire'?Object.fromEntries([...new FormData($('questionnaire'))].map(([k,v])=>[k,Number(v)])):null;
  const payload={stage,pulse,audio:audio.data,audio_source:audio.source,...(clinical?{clinical}:{})};setBusy(true);error('audio-error');error('result-error');
  try{
    const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json','X-MVP-Key':apiKey},body:JSON.stringify(payload)});
    const d=await r.json();if(!r.ok)throw new Error(d.error||'Analysis did not finish. Please try again.');showResult(d);
  }catch(e){error(stage==='initial'?'audio-error':'result-error',e.message==='Failed to fetch'?'Server disconnected. Check the project laptop and try again.':e.message);}
  finally{setBusy(false);}
}
async function demo(){
  if(busy)return;stopPulse();cancelRecording();error('pulse-error');error('audio-error');setBusy(true);
  try{
    const r=await fetch('/api/demo');if(!r.ok)throw new Error('Demo data is unavailable.');const d=await r.json();const engine=new PPG.PpgEngine({respiration:false});let windows=[];
    for(const [t,red,green,blue] of d.ppg_samples){const v=engine.push({t,r:red,g:green,b:blue});drawWave(v.waveform);if(v.window){if(v.window.quality.good)windows.push(v.window.heartRate);else windows=[];}}
    if(windows.length<3)throw new Error('The demo signal did not pass the quality check.');const latest=windows.slice(-3).sort((a,b)=>a-b);if(latest[2]-latest[0]>8)throw new Error('Demo readings are not consistent.');
    acceptPulse({bpm:latest[1],source:'demo',quality_passed:true,capture_seconds:d.ppg_samples.at(-1)[0],good_windows:windows.length});$('pulse-progress').value=30;$('pulse-guidance').textContent='Pulse was calculated from a demo PPG recording, not your camera.';
    const bin=Uint8Array.from(atob(d.audio),c=>c.charCodeAt(0));setAudio({data:d.audio,duration:d.audio_seconds,name:'Research cough sample',source:'demo'},new Blob([bin],{type:'audio/wav'}));
    for(const [name,value] of Object.entries(d.clinical_demo))$('questionnaire').elements.namedItem(name).value=String(value);
    $('cough-confirm').checked=true;go('audio');
  }catch(e){error('pulse-error',e.message);go('pulse');}finally{setBusy(false);}
}
function reset(){if(busy)return;stopPulse();cancelRecording();pulse=null;clearAudio();$('pulse-panel').querySelector('details').open=false;$('bpm').textContent='—';$('pulse-tag').textContent='Not measured';$('pulse-tag').classList.remove('success');$('audio-upload').value='';$('record-time').textContent='Microphone not recording';$('pulse-guidance').textContent='Measurement takes at least 30 seconds and requires a stable signal.';$('pulse-progress').value=0;$('manual-bpm').value='';$('questionnaire').reset();waveValues=[];drawWave();for(const id of ['pulse-error','audio-error','result-error'])error(id);update();go('pulse');}
async function health(){
  try{const r=await fetch('/api/health');if(!r.ok)throw Error();const d=await r.json();modelReady=d.ready;const el=$('server-status');el.replaceChildren();const dot=document.createElement('span');dot.className='status-dot';el.append(dot,document.createTextNode(d.error||d.phase));el.classList.toggle('ready',d.ready);el.classList.toggle('failed',!!d.error);}
  catch{modelReady=false;$('server-status').textContent='Server disconnected. Check the project laptop.';}update();
}
document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>go(b.dataset.step)));
$('start-pulse').addEventListener('click',startPulse);$('stop-pulse').addEventListener('click',()=>{stopPulse();$('bpm').textContent='—';$('pulse-tag').textContent='Stopped';$('pulse-guidance').textContent='Measurement stopped. Start a new measurement to continue.';});
$('use-manual').addEventListener('click',()=>{try{const value=$('manual-bpm').value;if(!value)throw new Error('Enter a reading from a reference device first.');const bpm=Number(value);if(!Number.isFinite(bpm)||bpm<45||bpm>220)throw new Error('Enter a measured pulse between 45 and 220 BPM.');stopPulse();error('pulse-error');acceptPulse({bpm,source:'manual'});$('pulse-guidance').textContent='Device reading saved as manual input.';}catch(e){error('pulse-error',e.message);}});
$('next-audio').addEventListener('click',()=>go('audio'));$('record-button').addEventListener('click',startRecording);$('stop-record').addEventListener('click',stopRecording);
$('audio-upload').addEventListener('change',async e=>{if(!e.target.files[0])return;cancelRecording();clearAudio();try{await loadAudio(e.target.files[0],e.target.files[0].name);}catch(err){error('audio-error',err.message);}});
$('cough-confirm').addEventListener('change',()=>{invalidate();update();});$('analyze-button').addEventListener('click',()=>analyze('initial'));
$('questionnaire').addEventListener('submit',e=>{e.preventDefault();if($('questionnaire').reportValidity())analyze('questionnaire');});
$('questionnaire').addEventListener('input',()=>{if(result?.stage==='questionnaire'){$('download-result').hidden=true;$('summary-clinical').textContent='Answers changed; update your result';const meta=document.querySelector('.result-meta');if(meta)meta.textContent='Your answers changed. Run analysis again to update this result.';}});
$('demo-button').addEventListener('click',demo);$('reset-button').addEventListener('click',reset);$('new-check').addEventListener('click',reset);
$('download-result').addEventListener('click',()=>{if(!result)return;const data={...result,pulse,created_at:new Date().toISOString(),notice:'Research prototype summary. Not a diagnosis.'};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='tbcatch-research-summary.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&ppg){stopPulse();$('bpm').textContent='—';$('pulse-tag').textContent='Stopped';$('pulse-guidance').textContent='Measurement stopped because this page became inactive. Start a new measurement to continue.';}if(document.hidden&&micStream){cancelRecording();error('audio-error','Recording stopped because this page became inactive. Record again to continue.');}});
window.addEventListener('pagehide',()=>{stopPulse();cancelRecording();});drawWave();update();health();setInterval(health,4000);
