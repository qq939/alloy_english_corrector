let audioStream = null;
let audioCtx = null;
let analyser = null;
let pcmBuffer = [];

const wave = document.getElementById('wave');
const logsBox = document.getElementById('logs');
const secureWarn = document.getElementById('secureWarn');

let lastLogsSig = '';

function logClient(msg) {
  const div = document.createElement('div');
  div.textContent = `[client] ${msg}`;
  logsBox.appendChild(div);
  logsBox.scrollTop = logsBox.scrollHeight;
}

const isSecure = location.protocol === 'https:' || location.hostname === 'localhost';
if (!isSecure) secureWarn.classList.remove('hidden');

 

function animateLogo() {
  const el = document.getElementById('logo');
  const target = 'TRAE SOLO';
  const chars = '01[]{}<>/\\|';
  let steps = 0;
  const timer = setInterval(() => {
    steps++;
    let out = '';
    for (let i = 0; i < target.length; i++) {
      if (Math.random() < steps / 20) out += target[i];
      else out += chars[Math.floor(Math.random() * chars.length)];
    }
    el.textContent = out;
    if (steps > 20) clearInterval(timer);
  }, 60);
}
function animateLogoSafe() {
  const targetEl = document.getElementById('logoText') || document.getElementById('logo');
  if (!targetEl) return;
  const target = 'MR. JIANG';
  const chars = '01[]{}<>/\\|';
  let steps = 0;
  const timer = setInterval(() => {
    steps++;
    let out = '';
    for (let i = 0; i < target.length; i++) {
      if (Math.random() < steps / 20) out += target[i];
      else out += chars[Math.floor(Math.random() * chars.length)];
    }
    targetEl.textContent = out;
    if (steps > 20) clearInterval(timer);
  }, 60);
}
animateLogoSafe();

function drawWave() {
  const ctx = wave.getContext('2d');
  const dataArray = new Uint8Array(analyser.frequencyBinCount);
  function resizeWave() {
    const dpr = window.devicePixelRatio || 1;
    const w = wave.parentElement.clientWidth || window.innerWidth;
    const parentH = wave.parentElement.clientHeight || Math.floor((window.innerHeight || 600) * 0.26);
    wave.width = Math.floor(w * dpr);
    let hpx = parentH;
    wave.height = Math.max(Math.floor(60 * dpr), Math.floor(hpx * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resizeWave();
  window.addEventListener('resize', resizeWave);
  function loop() {
    if (!analyser) return;
    analyser.getByteTimeDomainData(dataArray);
    ctx.clearRect(0, 0, wave.width, wave.height);
    ctx.beginPath();
    const slice = wave.width / dataArray.length;
    for (let i = 0; i < dataArray.length; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * (wave.height / (window.devicePixelRatio || 1))) / 2;
      const x = i * slice;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#0aa';
    ctx.stroke();
    requestAnimationFrame(loop);
  }
  loop();
}


async function startAudio() {
  // Call reset to clear server buffers and UI
  try {
    await fetch('/reset', { method: 'POST' });
    const liveBox = document.getElementById('live');
    if (liveBox) liveBox.textContent = '';
    logClient('缓冲区已重置');
  } catch (e) {
    console.error('Reset failed', e);
  }

  const baseConstraints = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };
  let constraints = { audio: baseConstraints };
  
  try {
    // 1. 初次尝试获取流（触发权限请求）
    let stream = await navigator.mediaDevices.getUserMedia(constraints);
    
    // 2. 检查是否为iPhone麦克风
    const track = stream.getAudioTracks()[0];
    if (track && track.label && track.label.toLowerCase().includes('iphone')) {
      logClient(`检测到iPhone麦克风 [${track.label}]，尝试切换...`);
      
      // 3. 枚举设备寻找替代品
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter(d => d.kind === 'audioinput' && !d.label.toLowerCase().includes('iphone'));
      
      if (inputs.length > 0) {
        // 优先选择内置麦克风 (Built-in) 或 MacBook 麦克风
        const best = inputs.find(d => d.label.toLowerCase().includes('built-in') || d.label.toLowerCase().includes('macbook')) || inputs[0];
        
        logClient(`切换至: ${best.label}`);
        
        // 停止旧流
        stream.getTracks().forEach(t => t.stop());
        
        // 使用新设备ID重新获取
        constraints.audio = { ...baseConstraints, deviceId: { exact: best.deviceId } };
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } else {
        logClient('未找到非iPhone麦克风，继续使用当前设备。');
      }
    } else if (track) {
      logClient(`使用麦克风: ${track.label}`);
    }

    // 4. 初始化音频上下文
    audioStream = stream;
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    } catch (_) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    const proc = audioCtx.createScriptProcessor(4096, 1, 1);
    source.connect(analyser);
    source.connect(proc);
    proc.connect(audioCtx.destination);
    proc.onaudioprocess = e => {
      const chan = e.inputBuffer.getChannelData(0);
      pcmBuffer.push(new Float32Array(chan));
    };
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    } else if (audioCtx && audioCtx.resume) {
      audioCtx.resume().catch(() => {});
    }
    setInterval(flushAudio, 1000);
    drawWave();
    document.getElementById('startAudio').classList.add('pulse');
    
  } catch (err) {
    logClient(`麦克风错误: ${err && (err.message || err.name) || 'Unknown'}`);
  }
}

function stopAudio() {
  if (audioStream) {
    audioStream.getTracks().forEach(t => t.stop());
    audioStream = null;
  }
  if (audioCtx) audioCtx.close();
  audioCtx = null; analyser = null;
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  function writeString(offset, str) { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); }
  const sRate = sampleRate || (audioCtx ? audioCtx.sampleRate : 16000);
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sRate, true);
  view.setUint32(28, sRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

function encodePCM16(samples) {
  const buf = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buf);
  let offset = 0;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([view], { type: 'application/octet-stream' });
}

function flattenPCM(chunks) {
  let length = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Float32Array(length);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out;
}

function flushAudio() {
  if (!pcmBuffer.length) return;
  const samples = flattenPCM(pcmBuffer);
  pcmBuffer = [];
  const pcm = encodePCM16(samples);
  const form = new FormData();
  form.append('audio_raw', pcm, 'chunk.pcm');
  form.append('sr', String(audioCtx ? audioCtx.sampleRate : 16000));
  fetch('/api/audio', { method: 'POST', body: form })
    .then(r => r.json())
    .then(_ => fetchLogs());
}

function fetchLogs() {
  fetch('/api/logs').then(r => r.json()).then(d => {
    const liveBox = document.getElementById('live');
    if (liveBox) liveBox.textContent = d.live || '';
    const logsSig = JSON.stringify(d.logs || []);
    if (logsSig !== lastLogsSig) {
      logsBox.innerHTML = (d.logs || []).map(x => `<div>${x}</div>`).join('') + '<div class="blockline"></div>';
      logsBox.scrollTop = logsBox.scrollHeight;
      lastLogsSig = logsSig;
    }
  });
}

document.getElementById('startAudio').onclick = startAudio;
document.getElementById('stopAudio').onclick = stopAudio;
const btnRecognize = document.getElementById('recognizeAudio');
if (btnRecognize) {
  btnRecognize.addEventListener('click', function(){
    logClient('识别按钮点击');
    btnRecognize.disabled = true;
    try { flushAudio(); } catch (e) {}
    setTimeout(function(){
      fetch('/api/recognize', { method: 'POST' })
        .then(r => r.json())
        .then(_ => fetchLogs())
        .finally(() => { btnRecognize.disabled = false; });
    }, 150);
  });
}
const btnDownload = document.getElementById('downloadSubmits');
if (btnDownload) {
  btnDownload.addEventListener('click', function(){
    window.location.href = '/api/download_submissions';
  });
}
setInterval(fetchLogs, 1000);
