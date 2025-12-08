let audioStream = null;
let audioCtx = null;
let analyser = null;
let pcmBuffer = [];

const wave = document.getElementById('wave');
const logsBox = document.getElementById('logs');
const secureWarn = document.getElementById('secureWarn');

// 修复日志过长不换行导致组件宽度变化的问题
if (logsBox) {
    const s = document.createElement('style');
    s.textContent = `
      #logs { min-width: 0; }
      #logs div { 
          word-break: break-word; 
          overflow-wrap: break-word; 
          white-space: pre-wrap; 
      }
    `;
    document.head.appendChild(s);
}

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
    
    // Fix: 显式设置样式宽高，防止 Canvas 分辨率设置导致组件撑大
    wave.style.width = w + 'px';
    const finalH = parentH || 60;
    wave.style.height = finalH + 'px';

    wave.width = Math.floor(w * dpr);
    wave.height = Math.max(Math.floor(60 * dpr), Math.floor(finalH * dpr));
    
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
    // 1. 尝试先枚举设备（如果已有权限，可以直接拿到列表，避免触发默认设备）
    let devices = await navigator.mediaDevices.enumerateDevices();
    let audioInputs = devices.filter(d => d.kind === 'audioinput');
    
    // 检查是否已有权限看到设备标签
    const hasLabels = audioInputs.some(d => d.label && d.label.length > 0);
    let bestDeviceId = null;
    
    // 辅助函数：选择最佳设备
    const pickBestDevice = (inputs) => {
        const candidates = inputs.filter(d => !d.label.toLowerCase().includes('iphone'));
        if (candidates.length === 0) return null;
        
        candidates.sort((a, b) => {
            const getScore = (d) => {
               const l = d.label.toLowerCase();
               if (l.includes('built-in') || l.includes('internal') || l.includes('macbook')) return 2;
               if (d.deviceId === 'default') return 0; 
               return 1;
            };
            return getScore(b) - getScore(a);
        });
        return candidates[0];
    };

    if (hasLabels) {
       logClient(`已获权限，发现设备: ${audioInputs.map(d => d.label).join(', ')}`);
       const best = pickBestDevice(audioInputs);
       if (best) {
           bestDeviceId = best.deviceId;
           logClient(`优先选择设备: ${best.label}`);
           constraints.audio = { ...baseConstraints, deviceId: { exact: bestDeviceId } };
       }
    }
    
    // 2. 获取音频流 (如果指定了 bestDeviceId，直接请求该设备；否则请求默认)
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch(e) {
        // 如果指定设备失败，回退到默认
        if (bestDeviceId) {
            logClient("指定设备获取失败，尝试默认设备...");
            constraints.audio = baseConstraints;
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        } else {
            throw e;
        }
    }

    // 3. (仅首次访问) 如果之前没拿到标签，现在拿到了，需要再次检查是否用的是 iPhone
    if (!hasLabels) {
        // 此时 stream 已打开，权限已获取
        devices = await navigator.mediaDevices.enumerateDevices();
        audioInputs = devices.filter(d => d.kind === 'audioinput');
        
        const currentTrack = stream.getAudioTracks()[0];
        const currentLabel = currentTrack ? currentTrack.label : '';
        logClient(`当前默认设备: ${currentLabel}`);
        
        const best = pickBestDevice(audioInputs);
        
        // 判定是否需要切换：如果当前是 iPhone，或者有更好的非 iPhone 设备
        const isCurrentIphone = currentLabel.toLowerCase().includes('iphone');
        const isTargetDifferent = best && best.label !== currentLabel;
        
        if (best && (isCurrentIphone || isTargetDifferent)) {
            logClient(`首次授权，切换至更佳设备: ${best.label}`);
            stream.getTracks().forEach(t => t.stop());
            constraints.audio = { ...baseConstraints, deviceId: { exact: best.deviceId } };
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        }
    } else {
        // 已有标签的情况下，我们已经尽力选了 bestDeviceId
        // 打印一下最终结果
        const currentTrack = stream.getAudioTracks()[0];
        logClient(`最终使用麦克风: ${currentTrack ? currentTrack.label : 'Unknown'}`);
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
    
    // Web Audio API Oscillator Hack for Background Persistence
    // 在同一个 AudioContext 中创建一个不可听的振荡器，保持音频引擎活跃
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.value = 10; // 10Hz, below human hearing mostly
        gain.gain.value = 0.001; // Extremely low volume
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        // 保存引用以便停止
        audioCtx.keepAliveOsc = osc; 
        console.log('Oscillator hack started');
    } catch(e) {
        console.warn('Oscillator hack failed', e);
    }

    setInterval(flushAudio, 1000);
    drawWave();
    document.getElementById('startAudio').classList.add('pulse');
    
  } catch (err) {
    logClient(`麦克风错误: ${err && (err.message || err.name) || 'Unknown'}`);
  }
}

function stopAudio() {
  if (audioCtx && audioCtx.keepAliveOsc) {
      try {
          audioCtx.keepAliveOsc.stop();
          audioCtx.keepAliveOsc.disconnect();
          audioCtx.keepAliveOsc = null;
      } catch(e) {}
  }
  
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
