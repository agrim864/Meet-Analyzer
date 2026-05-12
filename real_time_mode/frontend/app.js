const { useEffect, useMemo, useRef, useState } = React;

function formatTs(seconds) {
  const value = Number(seconds || 0);
  return `${value.toFixed(2)}s`;
}

function formatMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} ms` : "--";
}

function statusClass(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("error") || text.includes("failed")) return "bad";
  if (text.includes("active") || text.includes("connected") || text.includes("ready") || text.includes("ok") || text.includes("success")) return "good";
  if (text.includes("waiting") || text.includes("idle") || text.includes("skipped")) return "warn";
  return "";
}

function macroTypes(payload) {
  const layout = payload?.macro_layout || [];
  return layout
    .map((item) => {
      const label = String(item?.type || item?.label || "").trim();
      if (!label) return "";
      const conf = Number(item?.confidence);
      return Number.isFinite(conf) ? `${label} (${conf.toFixed(2)})` : label;
    })
    .filter(Boolean);
}

function microElementCount(payload) {
  return Array.isArray(payload?.micro_elements) ? payload.micro_elements.length : 0;
}

function cursorDisplay(response) {
  const cursor = response?.cursor || {};
  const target = response?.visual_payload?.cursor_target || {};
  return {
    visible: Boolean(cursor.detected ?? cursor.visible),
    type: cursor.type || cursor.cursor_type || "unknown",
    micro: target.element_text || target.element_type || "none",
    macro: target.parent_macro_region_type || "none",
  };
}

function summaryStatus(response) {
  const summary = String(response?.summary || "").trim();
  const debugError = String(response?.debug?.summary_error || "").trim();
  if (summary === "Gemini not available.") {
    if (/429|quota|Too Many Requests/i.test(debugError)) {
      return {
        provider: "gemini",
        model: response?.debug?.summary_model || "gemini",
        text: "Gemini quota exceeded.",
        detail: debugError,
      };
    }
    return {
      provider: "gemini",
      model: response?.debug?.summary_model || "gemini",
      text: "Gemini not available.",
      detail: debugError || "No response from Gemini.",
    };
  }
  return {
    provider: response?.debug?.summary_provider || "heuristic",
    model: response?.debug?.summary_model || "heuristic",
    text: summary || "No summary yet.",
    detail: debugError,
  };
}

function transcriptIdentity(item) {
  if (!item) return "transcript";
  if (item.chunk_id) return String(item.chunk_id);
  return [
    item.speaker || "speaker",
    Number(item.start_time || 0).toFixed(2),
    Number(item.end_time || 0).toFixed(2),
    String(item.text || "").trim(),
  ].join("|");
}

function mergeTranscriptChunks(previous, incoming) {
  const seen = new Set();
  const merged = [];
  [...previous, ...incoming].forEach((item) => {
    const key = transcriptIdentity(item);
    if (seen.has(key)) return;
    seen.add(key);
    merged.push(item);
  });
  return merged.slice(-30);
}

function App() {
  const [config, setConfig] = useState(null);
  const [sessionId, setSessionId] = useState("");
  const [sessionTitle, setSessionTitle] = useState("Live screen-share session");
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [isSharingScreen, setIsSharingScreen] = useState(false);
  const [currentMode, setCurrentMode] = useState("unknown");
  const [latestTranscript, setLatestTranscript] = useState(null);
  const [transcriptHistory, setTranscriptHistory] = useState([]);
  const [latestFrameEvent, setLatestFrameEvent] = useState(null);
  const [latestModalResponse, setLatestModalResponse] = useState(null);
  const [latestChangeDebug, setLatestChangeDebug] = useState(null);
  const [framesSubmitted, setFramesSubmitted] = useState(0);
  const [framesSent, setFramesSent] = useState(0);
  const [framesSkipped, setFramesSkipped] = useState(0);
  const [framesQueued, setFramesQueued] = useState(0);
  const [framesDropped, setFramesDropped] = useState(0);
  const [pendingFrameCount, setPendingFrameCount] = useState(0);
  const [latestFrameGapMs, setLatestFrameGapMs] = useState(null);
  const [latestAnalysisLagMs, setLatestAnalysisLagMs] = useState(null);
  const [isProcessingFrame, setIsProcessingFrame] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [sessionStatus, setSessionStatus] = useState("idle");
  const [captureStatus, setCaptureStatus] = useState("idle");
  const [modalStatus, setModalStatus] = useState("idle");
  const [supabaseStatus, setSupabaseStatus] = useState("not configured");
  const [warmupDetails, setWarmupDetails] = useState("Not started.");
  const [transcriptDraft, setTranscriptDraft] = useState("");
  const [speakerDraft, setSpeakerDraft] = useState("speaker_1");
  const [micStatus, setMicStatus] = useState("idle");
  const [showRawJson, setShowRawJson] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const lowResCanvasRef = useRef(null);
  const frameTimerRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const pendingStreamRef = useRef(null);
  const micStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const audioSourceRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const pcmChunksRef = useRef([]);
  const pcmSampleRateRef = useRef(16000);
  const workerRef = useRef(null);
  const pendingDiffRef = useRef(null);
  const captureBusyRef = useRef(false);
  const processingRef = useRef(false);
  const inFlightFramesRef = useRef(new Map());
  const clientFrameIdRef = useRef(0);
  const latestAppliedFrameIdRef = useRef(0);
  const pendingFrameRef = useRef(null);
  const transcriptBusyRef = useRef(false);
  const transcriptDisabledRef = useRef(false);
  const transcriptErrorCountRef = useRef(0);
  const pendingTranscriptBlobRef = useRef(null);
  const transcriptChunkIntervalRef = useRef(null);
  const transcriptRestartTimeoutRef = useRef(null);
  const recorderChunksRef = useRef([]);
  const sessionIdRef = useRef("");
  const configRef = useRef(null);
  const currentModeRef = useRef("unknown");
  const sharingRef = useRef(false);
  const sessionActiveRef = useRef(false);
  const lastModeRef = useRef("unknown");
  const lastSampleTsRef = useRef(0);
  const lastChangedFrameTsRef = useRef(0);
  const recentKeyframesRef = useRef([]);
  const recentTranscriptRef = useRef([]);
  const warmupPollRef = useRef(null);

  useEffect(() => {
    fetch("/api/config")
      .then((res) => res.json())
      .then((payload) => {
        setConfig(payload);
        if (payload.supabaseUrl && payload.supabaseAnonKey) {
          setSupabaseStatus("ready");
        }
        if (payload.runtime?.warmed) {
          setModalStatus("ready");
          setWarmupDetails("Models and Deepgram are ready.");
        } else if (payload.runtime?.running) {
          setModalStatus("warming");
          setWarmupDetails("Models and Deepgram are warming up.");
        }
      })
        .catch((err) => setErrorMessage(`Config load failed: ${err.message}`));
    }, []);

  useEffect(() => {
    const worker = new Worker("/static/diff-worker.js");
    worker.onmessage = (event) => {
      if (pendingDiffRef.current) {
        pendingDiffRef.current(event.data);
        pendingDiffRef.current = null;
      }
    };
    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  useEffect(() => {
    recentTranscriptRef.current = transcriptHistory.slice(-6);
  }, [transcriptHistory]);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    configRef.current = config;
  }, [config]);

  useEffect(() => {
    currentModeRef.current = currentMode;
  }, [currentMode]);

  useEffect(() => {
    processingRef.current = isProcessingFrame;
  }, [isProcessingFrame]);

  useEffect(() => {
    sharingRef.current = isSharingScreen;
  }, [isSharingScreen]);

  useEffect(() => {
    sessionActiveRef.current = isSessionActive;
  }, [isSessionActive]);

  const frameIntervals = useMemo(
    () => config?.frameIntervalsMs || { demo: 250, code: 500, slide: 750, other: 1000, unknown: 500 },
    [config]
  );

  const maxInFlightFrames = Math.max(1, Number(config?.maxInFlightFrames || 2));
  const fullFrameChangeRatio = Math.max(0.10, Number(config?.fullFrameChangeRatio || 0.55));
  const supersedeInFlightChangeRatio = Math.max(0.02, Number(config?.supersedeInFlightChangeRatio || 0.10));
  const frameRequestTimeoutMs = Math.max(30000, Number(config?.frameRequestTimeoutMs || 120000));

  async function insertSupabase(table, row, method = "POST", filters = "") {
    const runtimeConfig = configRef.current;
    if (!runtimeConfig?.supabaseUrl || !runtimeConfig?.supabaseAnonKey) {
      return;
    }
    const url = `${runtimeConfig.supabaseUrl}/rest/v1/${table}${filters}`;
    const response = await fetch(url, {
      method,
      headers: {
        apikey: runtimeConfig.supabaseAnonKey,
        Authorization: `Bearer ${runtimeConfig.supabaseAnonKey}`,
        "Content-Type": "application/json",
        Prefer: "return=minimal",
      },
      body: method === "PATCH" ? JSON.stringify(row) : JSON.stringify(row),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${table} ${method} failed: ${response.status} ${text}`);
    }
  }

  function resetDiff(mode = "unknown") {
    lastModeRef.current = mode;
    if (workerRef.current) {
      workerRef.current.postMessage({ type: "reset", payload: { mode } });
    }
  }

  function resetFrameCounters() {
    setFramesSubmitted(0);
    setFramesSent(0);
    setFramesSkipped(0);
    setFramesQueued(0);
    setFramesDropped(0);
    setPendingFrameCount(0);
    setLatestFrameGapMs(null);
    setLatestAnalysisLagMs(null);
    setLatestChangeDebug(null);
    setLatestModalResponse(null);
    setLatestFrameEvent(null);
  }

  function runDiff(imageData, mode) {
    return new Promise((resolve) => {
      pendingDiffRef.current = resolve;
      workerRef.current.postMessage(
        {
          type: "diff",
          payload: {
            width: imageData.width,
            height: imageData.height,
            rgbaBuffer: imageData.data.buffer,
            mode,
          },
        },
        [imageData.data.buffer]
      );
    });
  }

  function canvasToJpegDataUrl(canvas, quality = 0.72) {
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Canvas JPEG encoding failed"));
            return;
          }
          const reader = new FileReader();
          reader.onloadend = () => resolve(String(reader.result || ""));
          reader.onerror = () => reject(reader.error || new Error("Canvas JPEG read failed"));
          reader.readAsDataURL(blob);
        },
        "image/jpeg",
        quality
      );
    });
  }

  async function attachScreenStream(stream) {
    mediaStreamRef.current = stream;
    const video = videoRef.current;
    video.srcObject = stream;
    await video.play();
    stream.getVideoTracks()[0].addEventListener("ended", () => stopSession());
    setIsSharingScreen(true);
  }

  function chooseRecorderMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const candidate of candidates) {
      if (window.MediaRecorder?.isTypeSupported?.(candidate)) {
        return candidate;
      }
    }
    return "";
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(reader.error || new Error("FileReader failed"));
      reader.readAsDataURL(blob);
    });
  }

  function encodeWavBlobFromFloat32(samples, sampleRate) {
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeString(offset, value) {
      for (let i = 0; i < value.length; i += 1) {
        view.setUint8(offset + i, value.charCodeAt(i));
      }
    }

    writeString(0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }

    return new Blob([buffer], { type: "audio/wav" });
  }

  function collectPcmChunkBlob() {
    const chunks = pcmChunksRef.current.splice(0, pcmChunksRef.current.length);
    if (!chunks.length) {
      return null;
    }
    const totalLength = chunks.reduce((sum, item) => sum + item.length, 0);
    const merged = new Float32Array(totalLength);
    let cursor = 0;
    chunks.forEach((item) => {
      merged.set(item, cursor);
      cursor += item.length;
    });
    return encodeWavBlobFromFloat32(merged, pcmSampleRateRef.current || 16000);
  }

  async function flushTranscriptBlob(blob) {
    if (!blob || !blob.size || !sessionIdRef.current) {
      return;
    }
    const minBytes = Math.max(512, Number(configRef.current?.minTranscriptBlobBytes || 2048));
    if (blob.size < minBytes) {
      return;
    }
    if (transcriptDisabledRef.current) {
      return;
    }
    if (transcriptBusyRef.current) {
      pendingTranscriptBlobRef.current = blob;
      return;
    }
    transcriptBusyRef.current = true;
    try {
      const audio_base64 = await blobToBase64(blob);
      const response = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          audio_base64,
          mime_type: blob.type || "audio/webm",
          speaker: speakerDraft || "speaker_1",
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      if (payload?.transcript_error) {
        const message = String(payload.transcript_error || "Transcript processing failed.");
        transcriptErrorCountRef.current += 1;
        const maxErrors = Math.max(1, Number(configRef.current?.transcriptMaxConsecutiveErrors || 4));
        const permanent = /api[_ -]?key|required|unauthorized|forbidden|401|403|quota|billing/i.test(message);
        if (permanent || transcriptErrorCountRef.current >= maxErrors) {
          transcriptDisabledRef.current = true;
          setMicStatus("transcript unavailable");
          setErrorMessage(message);
        } else {
          setMicStatus("transcript retrying");
        }
        pendingTranscriptBlobRef.current = null;
        return;
      }
      if (payload?.transcript_warning) {
        setMicStatus("listening");
        return;
      }
      const chunks = Array.isArray(payload?.chunks) ? payload.chunks.filter((item) => item?.text) : [];
      transcriptErrorCountRef.current = 0;
      if (chunks.length) {
        setTranscriptHistory((prev) => {
          return mergeTranscriptChunks(prev, chunks);
        });
        setLatestTranscript(chunks[chunks.length - 1]);
        if (configRef.current?.supabaseUrl && configRef.current?.supabaseAnonKey) {
          try {
            for (const chunk of chunks) {
              await insertSupabase("transcript_segments", {
                meeting_id: sessionIdRef.current,
                start_time: chunk.start_time,
                end_time: chunk.end_time,
                speaker: chunk.speaker,
                text: chunk.text,
                is_final: true,
              });
            }
            setSupabaseStatus("transcript inserted");
          } catch (err) {
            setSupabaseStatus(`write failed: ${err.message}`);
          }
        }
      }
      setMicStatus("listening");
    } catch (err) {
      transcriptErrorCountRef.current += 1;
      const maxErrors = Math.max(1, Number(configRef.current?.transcriptMaxConsecutiveErrors || 4));
      if (transcriptErrorCountRef.current >= maxErrors) {
        setMicStatus("error");
        setErrorMessage(`Transcript processing failed: ${err.message}`);
      } else {
        setMicStatus("transcript retrying");
      }
    } finally {
      transcriptBusyRef.current = false;
      const pending = pendingTranscriptBlobRef.current;
      pendingTranscriptBlobRef.current = null;
      if (pending) {
        flushTranscriptBlob(pending);
      }
    }
  }

  async function startTranscriptCapture(audioStream) {
    if (!audioStream) {
      setMicStatus("unavailable");
      return;
    }
    try {
      const mimeType = chooseRecorderMimeType();
      if (window.MediaRecorder && mimeType) {
        const recorder = new MediaRecorder(audioStream, { mimeType });
        mediaRecorderRef.current = recorder;
        recorder.ondataavailable = (event) => {
          const minBytes = Math.max(512, Number(configRef.current?.minTranscriptBlobBytes || 2048));
          if (event.data && event.data.size >= minBytes) {
            flushTranscriptBlob(event.data);
          }
        };
        recorder.onerror = (event) => {
          const message = event?.error?.message || "MediaRecorder failed.";
          setMicStatus("error");
          setErrorMessage(`Transcript capture failed: ${message}`);
        };
        recorder.start(10000);
        setMicStatus("listening");
        return;
      }

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) {
        throw new Error("No supported audio capture API was available.");
      }
      const audioContext = new AudioCtx({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(audioStream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      pcmChunksRef.current = [];
      pcmSampleRateRef.current = audioContext.sampleRate || 16000;
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(input));
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      audioContextRef.current = audioContext;
      audioSourceRef.current = source;
      scriptProcessorRef.current = processor;
      setMicStatus("listening");
      transcriptChunkIntervalRef.current = window.setInterval(() => {
        const blob = collectPcmChunkBlob();
        if (blob && blob.size > 44) {
          flushTranscriptBlob(blob);
        }
      }, 12000);
    } catch (err) {
      setMicStatus("error");
      setErrorMessage(`Microphone capture failed: ${err.message}`);
    }
  }

  function stopWarmupPolling() {
    if (warmupPollRef.current) {
      clearInterval(warmupPollRef.current);
      warmupPollRef.current = null;
    }
  }

  function renderWarmupDetails(runtime) {
    if (!runtime) return "Not started.";
    if (runtime.error) return `Warmup failed: ${runtime.error}`;
    if (runtime.remote_warning) return `Remote warmup is retrying: ${runtime.remote_warning}`;
    if (runtime.warmed && runtime.deepgram_ready) return "All models and Deepgram are ready.";
    if (runtime.running) return "Warming models and Deepgram...";
    return "Waiting to start warmup.";
  }

  async function pollWarmupUntilReady() {
    stopWarmupPolling();
    return new Promise((resolve, reject) => {
      warmupPollRef.current = setInterval(async () => {
        try {
          const response = await fetch("/api/warmup/status");
          if (!response.ok) {
            throw new Error(await response.text());
          }
          const runtime = await response.json();
          setWarmupDetails(renderWarmupDetails(runtime));
          if (runtime.error) {
            stopWarmupPolling();
            reject(new Error(runtime.error));
            return;
          }
          if (runtime.remote_warning) {
            setSessionStatus("warming");
            setModalStatus("warming");
            return;
          }
          if (runtime.running) {
            setSessionStatus("warming");
            setModalStatus("warming");
            return;
          }
          if (runtime.warmed && runtime.deepgram_ready) {
            stopWarmupPolling();
            resolve(runtime);
          }
        } catch (err) {
          stopWarmupPolling();
          reject(err);
        }
      }, 800);
    });
  }

  function startCaptureLoop() {
    setCaptureStatus("capturing");
    lastSampleTsRef.current = 0;
    lastChangedFrameTsRef.current = 0;
    pendingFrameRef.current = null;
    clientFrameIdRef.current = 0;
    latestAppliedFrameIdRef.current = 0;
    abortInFlightFrames();
    updateFrameProcessingState();
    resetDiff(currentModeRef.current || "unknown");
    if (frameTimerRef.current) {
      clearInterval(frameTimerRef.current);
    }
    frameTimerRef.current = setInterval(() => {
      if (!captureBusyRef.current) {
        captureBusyRef.current = true;
        captureLoop().finally(() => {
          captureBusyRef.current = false;
        });
      }
    }, 75);
  }

  function noteChangedFrame(now) {
    if (lastChangedFrameTsRef.current > 0) {
      setLatestFrameGapMs(now - lastChangedFrameTsRef.current);
    }
    lastChangedFrameTsRef.current = now;
  }

  function queueLatestFrame(payload, changedDebug, mode) {
    const replacingQueuedFrame = Boolean(pendingFrameRef.current);
    pendingFrameRef.current = { payload, changedDebug, mode };
    if (!replacingQueuedFrame) {
      setFramesQueued((value) => value + 1);
    }
    setPendingFrameCount(1);
    if (replacingQueuedFrame) {
      setFramesSkipped((value) => value + 1);
    }
    setModalStatus("queued latest frame");
  }

  function updateFrameProcessingState() {
    const busy = inFlightFramesRef.current.size > 0;
    processingRef.current = busy;
    setIsProcessingFrame(busy);
    setPendingFrameCount(pendingFrameRef.current ? 1 : 0);
  }

  function abortInFlightFrames() {
    inFlightFramesRef.current.forEach((request) => {
      try {
        if (request.timeoutId) {
          clearTimeout(request.timeoutId);
        }
        request.controller.abort();
      } catch (_) {}
    });
    inFlightFramesRef.current.clear();
  }

  function abortOlderInFlightFrames(frameId) {
    let aborted = 0;
    inFlightFramesRef.current.forEach((request, requestFrameId) => {
      if (Number(requestFrameId) < Number(frameId)) {
        try {
          if (request.timeoutId) {
            clearTimeout(request.timeoutId);
          }
          request.controller.abort();
        } catch (_) {}
        inFlightFramesRef.current.delete(requestFrameId);
        aborted += 1;
      }
    });
    if (aborted) {
      setFramesDropped((value) => value + aborted);
      updateFrameProcessingState();
    }
  }

  function abortOldestInFlightFrame() {
    let oldestFrameId = null;
    let oldestRequest = null;
    inFlightFramesRef.current.forEach((request, requestFrameId) => {
      if (!oldestRequest || Number(request.startedAt || 0) < Number(oldestRequest.startedAt || 0)) {
        oldestFrameId = requestFrameId;
        oldestRequest = request;
      }
    });
    if (!oldestRequest) {
      return false;
    }
    try {
      if (oldestRequest.timeoutId) {
        clearTimeout(oldestRequest.timeoutId);
      }
      oldestRequest.controller.abort();
    } catch (_) {}
    inFlightFramesRef.current.delete(oldestFrameId);
    setFramesDropped((value) => value + 1);
    updateFrameProcessingState();
    return true;
  }

  function flushQueuedFrame() {
    if (!sharingRef.current || !sessionActiveRef.current || inFlightFramesRef.current.size >= maxInFlightFrames) {
      return;
    }
    const pending = pendingFrameRef.current;
    if (!pending) {
      return;
    }
    pendingFrameRef.current = null;
    setPendingFrameCount(0);
    dispatchFramePayload(pending.payload, pending.changedDebug, pending.mode);
  }

  async function dispatchFramePayload(payload, changedDebug, mode) {
    const frameId = Number(payload.client_frame_id || 0);
    const isFullFrameChange = Number(payload.client_change?.changed_pixel_ratio || 0) >= fullFrameChangeRatio;
    const isSupersedingChange = Number(payload.client_change?.changed_pixel_ratio || 0) >= supersedeInFlightChangeRatio;
    const hasCompletedAnalysis = latestAppliedFrameIdRef.current > 0;
    if (!hasCompletedAnalysis && inFlightFramesRef.current.size > 0) {
      queueLatestFrame(payload, changedDebug, mode);
      return;
    }
    if (hasCompletedAnalysis && (isFullFrameChange || isSupersedingChange) && frameId > 0) {
      abortOlderInFlightFrames(frameId);
    }
    if (inFlightFramesRef.current.size >= maxInFlightFrames) {
      const freedSlot = hasCompletedAnalysis && isSupersedingChange ? abortOldestInFlightFrame() : false;
      if (!freedSlot && inFlightFramesRef.current.size >= maxInFlightFrames) {
        queueLatestFrame(payload, changedDebug, mode);
        return;
      }
    }
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => {
      try {
        controller.abort();
      } catch (_) {}
    }, frameRequestTimeoutMs);
    if (frameId > 0) {
      inFlightFramesRef.current.set(frameId, { controller, startedAt: performance.now(), timeoutId });
    }
    updateFrameProcessingState();
    setModalStatus("processing");
    setFramesSubmitted((value) => value + 1);
    try {
      const response = await fetch(configRef.current?.modalProcessFrameUrl || "/api/process-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const modalJson = await response.json();
      const responseFrameId = Number(modalJson.client_frame_id || frameId || 0);
      if (modalJson.stale || modalJson.dropped || (responseFrameId && responseFrameId < latestAppliedFrameIdRef.current)) {
        setFramesDropped((value) => value + 1);
        setModalStatus(pendingFrameRef.current ? "queued latest frame" : "waiting for latest frame");
        return;
      }
      const analysisLagMs = Date.now() - (Number(payload.timestamp || 0) * 1000);
      const enrichedModalJson = {
        ...modalJson,
        analyzed_frame_data_url: payload.frame_jpeg_base64,
        analyzed_frame_timestamp: payload.timestamp,
        client_analysis_lag_ms: analysisLagMs,
      };
      if (responseFrameId) {
        latestAppliedFrameIdRef.current = Math.max(latestAppliedFrameIdRef.current, responseFrameId);
      }
      setFramesSent((value) => value + 1);
      setModalStatus("connected");
      setErrorMessage("");
      setCurrentMode(modalJson.mode || mode || "unknown");
      setLatestModalResponse(enrichedModalJson);
      setLatestAnalysisLagMs(analysisLagMs);
      setLatestFrameEvent(enrichedModalJson.raw_record || enrichedModalJson);
      recentKeyframesRef.current = [...recentKeyframesRef.current.slice(-29), {
        timestamp: modalJson.timestamp,
        mode: modalJson.mode,
        summary: modalJson.summary,
      }];

      if (configRef.current?.supabaseUrl && configRef.current?.supabaseAnonKey) {
        void (async () => {
          try {
            await insertSupabase("frame_events", {
              meeting_id: sessionIdRef.current,
              timestamp: modalJson.timestamp,
              mode: modalJson.mode,
              mode_confidence: modalJson.mode_confidence,
              visual_change: modalJson.visual_change,
              processor_used: modalJson.processor_used,
              summary: modalJson.summary,
              payload_json: modalJson.visual_payload,
              debug_json: modalJson.debug,
            });
            setSupabaseStatus("frame event inserted");
          } catch (err) {
            setSupabaseStatus(`write failed: ${err.message}`);
          }
        })();
      }
    } catch (err) {
      if (err?.name === "AbortError") {
        setModalStatus(pendingFrameRef.current ? "queued latest frame" : "superseded old frame");
        return;
      }
      setModalStatus("error");
      setErrorMessage(`Frame processing failed: ${err.message}`);
    } finally {
      if (frameId > 0) {
        const request = inFlightFramesRef.current.get(frameId);
        if (request?.timeoutId) {
          clearTimeout(request.timeoutId);
        }
        inFlightFramesRef.current.delete(frameId);
      }
      updateFrameProcessingState();
      flushQueuedFrame();
    }
  }

  async function startSession() {
    const id = sessionId || crypto.randomUUID();
    let stream = null;
    let audioStream = null;
    resetFrameCounters();
    setSessionId(id);
    setSessionStatus("starting");
    setCaptureStatus("requesting permission");
    setModalStatus("warming");
    setMicStatus("requesting");
    transcriptDisabledRef.current = false;
    setWarmupDetails("Waiting for screen-share permission...");
    setErrorMessage("");
    try {
      // Request the screen share inside the direct user gesture so the browser does not suppress the popup.
      stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      pendingStreamRef.current = stream;
      stream.getVideoTracks()[0].addEventListener("ended", () => stopSession());
      setCaptureStatus("permission granted");
      setWarmupDetails("Screen share permission granted. Requesting microphone and warming models before capture starts...");
      const sharedAudioTracks = stream.getAudioTracks ? stream.getAudioTracks() : [];
      try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        micStreamRef.current = audioStream;
        setMicStatus("microphone");
      } catch (micErr) {
        if (sharedAudioTracks && sharedAudioTracks.length) {
          audioStream = new MediaStream(sharedAudioTracks);
          micStreamRef.current = audioStream;
          setMicStatus("shared audio");
        } else {
          setMicStatus("denied");
          setWarmupDetails("No microphone or share-audio track was available. Screen capture will continue, but transcript will stay unavailable.");
        }
      }

      const response = await fetch("/api/sessions/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: id, title: sessionTitle }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }

      setIsSessionActive(true);
      setSessionStatus("warming");
      setModalStatus("warming");
      setWarmupDetails("Warming SigLIP, cursor, OCR, YOLO layout, slide analysis, LLM summarizer, and Deepgram...");

      const warmResponse = await fetch("/api/warmup/start", { method: "POST" });
      if (!warmResponse.ok) {
        throw new Error(await warmResponse.text());
      }
      const warmPayload = await warmResponse.json();
      setWarmupDetails(renderWarmupDetails(warmPayload));
      if (warmPayload?.error && !warmPayload?.remote_warning) throw new Error(warmPayload.error);
      await pollWarmupUntilReady();

      await attachScreenStream(stream);
      pendingStreamRef.current = null;
      if (micStreamRef.current) {
        await startTranscriptCapture(micStreamRef.current);
      }
      setSessionStatus("active");
      setModalStatus("ready");
      setCaptureStatus("capturing");
      setWarmupDetails("All models and Deepgram are ready. Screen capture has started.");
      startCaptureLoop();

      if (configRef.current?.supabaseUrl && configRef.current?.supabaseAnonKey) {
        try {
          await insertSupabase("meetings", {
            id,
            title: sessionTitle,
            started_at: new Date().toISOString(),
            status: "active",
          });
          setSupabaseStatus("meeting row inserted");
        } catch (err) {
          setSupabaseStatus(`write failed: ${err.message}`);
        }
      }
    } catch (err) {
      stopWarmupPolling();
      if (stream) {
        try {
          stream.getTracks().forEach((track) => track.stop());
        } catch (_) {}
      }
      if (audioStream) {
        try {
          audioStream.getTracks().forEach((track) => track.stop());
        } catch (_) {}
      }
      pendingStreamRef.current = null;
      mediaStreamRef.current = null;
      micStreamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsSharingScreen(false);
      setSessionStatus("error");
      setCaptureStatus("error");
      setModalStatus("error");
      setWarmupDetails("Warmup failed.");
      setErrorMessage(`Session start failed: ${err.message}`);
    }
  }

  async function stopSession() {
    stopWarmupPolling();
    if (frameTimerRef.current) {
      clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }
    abortInFlightFrames();
    if (pendingStreamRef.current) {
      pendingStreamRef.current.getTracks().forEach((track) => track.stop());
      pendingStreamRef.current = null;
    }
    if (transcriptChunkIntervalRef.current) {
      clearInterval(transcriptChunkIntervalRef.current);
      transcriptChunkIntervalRef.current = null;
    }
    if (transcriptRestartTimeoutRef.current) {
      clearTimeout(transcriptRestartTimeoutRef.current);
      transcriptRestartTimeoutRef.current = null;
    }
    if (mediaRecorderRef.current) {
      try {
        if (mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
        }
      } catch (_) {}
      mediaRecorderRef.current.ondataavailable = null;
      mediaRecorderRef.current.onerror = null;
      mediaRecorderRef.current = null;
    }
    if (scriptProcessorRef.current) {
      try {
        scriptProcessorRef.current.disconnect();
      } catch (_) {}
      scriptProcessorRef.current.onaudioprocess = null;
      scriptProcessorRef.current = null;
    }
    if (audioSourceRef.current) {
      try {
        audioSourceRef.current.disconnect();
      } catch (_) {}
      audioSourceRef.current = null;
    }
    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (_) {}
      audioContextRef.current = null;
    }
    const finalTranscriptBlob = collectPcmChunkBlob();
    if (finalTranscriptBlob && finalTranscriptBlob.size > 44) {
      void flushTranscriptBlob(finalTranscriptBlob);
    }
    recorderChunksRef.current = [];
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    resetDiff("unknown");
    setIsSharingScreen(false);
    setCaptureStatus("stopped");
    setModalStatus("idle");
    setMicStatus("idle");
    setWarmupDetails("Not started.");
    setIsProcessingFrame(false);
    captureBusyRef.current = false;
    processingRef.current = false;
    pendingFrameRef.current = null;
    inFlightFramesRef.current.clear();
    setPendingFrameCount(0);
    lastChangedFrameTsRef.current = 0;
    transcriptDisabledRef.current = false;
    if (sessionId) {
      try {
        await fetch(`/api/sessions/${sessionId}/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ended_at: Date.now() / 1000 }),
        });
      } catch (_) {}
      if (configRef.current?.supabaseUrl && configRef.current?.supabaseAnonKey) {
        try {
          await insertSupabase("meetings", { ended_at: new Date().toISOString(), status: "ended" }, "PATCH", `?id=eq.${sessionIdRef.current}`);
          setSupabaseStatus("meeting ended");
        } catch (err) {
          setSupabaseStatus(`write failed: ${err.message}`);
        }
      }
    }
    setIsSessionActive(false);
    setSessionStatus("stopped");
    setSessionId("");
  }

  async function addTranscriptChunk() {
    const text = transcriptDraft.trim();
    if (!text || !sessionId) {
      return;
    }
    const nowSec = performance.now() / 1000;
    const chunk = {
      chunk_id: crypto.randomUUID(),
      start_time: nowSec,
      end_time: nowSec + 1.5,
      speaker: speakerDraft || "speaker_1",
      text,
      is_final: true,
    };
    setTranscriptDraft("");
    setLatestTranscript(chunk);
    setTranscriptHistory((prev) => [...prev.slice(-19), chunk]);
    if (config?.supabaseUrl && config?.supabaseAnonKey) {
      try {
        await insertSupabase("transcript_segments", {
          meeting_id: sessionId,
          start_time: chunk.start_time,
          end_time: chunk.end_time,
          speaker: chunk.speaker,
          text: chunk.text,
          is_final: true,
        });
        setSupabaseStatus("transcript inserted");
      } catch (err) {
        setSupabaseStatus(`write failed: ${err.message}`);
      }
    }
  }

  async function startScreenShare() {
    if (!isSessionActive) {
      setErrorMessage("Click Start Session + Share. It now handles warmup and the screen-share prompt together.");
      return;
    }
    if (modalStatus !== "ready") {
      setErrorMessage("Models are still warming up. Wait for session status to become active.");
      return;
    }
    setCaptureStatus("requesting");
    setErrorMessage("");
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      await attachScreenStream(stream);
      setCaptureStatus("capturing");
      startCaptureLoop();
    } catch (err) {
      setCaptureStatus("error");
      setErrorMessage(`Screen share failed: ${err.message}`);
    }
  }

  function scaleBbox(box, srcWidth, srcHeight, dstWidth, dstHeight) {
    if (!box) return null;
    const [x1, y1, x2, y2] = box;
    return [
      Math.round((x1 / srcWidth) * dstWidth),
      Math.round((y1 / srcHeight) * dstHeight),
      Math.round((x2 / srcWidth) * dstWidth),
      Math.round((y2 / srcHeight) * dstHeight),
    ];
  }

  async function captureLoop() {
    if (!videoRef.current || !canvasRef.current || !lowResCanvasRef.current || !sharingRef.current) {
      return;
    }
    const video = videoRef.current;
    if (video.readyState < 2) {
      return;
    }
    const mode = currentModeRef.current || "unknown";
    const intervalMs = frameIntervals[mode] || frameIntervals.unknown || 3000;
    const now = performance.now();
    if (now - lastSampleTsRef.current < intervalMs) {
      return;
    }
    lastSampleTsRef.current = now;

    if (lastModeRef.current !== mode) {
      resetDiff(mode);
      lastModeRef.current = mode;
    }

    const videoWidth = video.videoWidth || 1;
    const videoHeight = video.videoHeight || 1;
    const maxProcessingSide = Math.max(320, Number(configRef.current?.captureMaxSide || 512));
    const longestSide = Math.max(videoWidth, videoHeight);
    const scale = longestSide > maxProcessingSide ? (maxProcessingSide / longestSide) : 1.0;
    const targetWidth = Math.max(1, Math.round(videoWidth * scale));
    const targetHeight = Math.max(1, Math.round(videoHeight * scale));

    const lowCanvas = lowResCanvasRef.current;
    const lowCtx = lowCanvas.getContext("2d", { willReadFrequently: true });
    lowCanvas.width = 320;
    lowCanvas.height = 180;
    lowCtx.drawImage(video, 0, 0, lowCanvas.width, lowCanvas.height);
    const imageData = lowCtx.getImageData(0, 0, lowCanvas.width, lowCanvas.height);
    let diff = await runDiff(imageData, mode);
    const needsBootstrapFrame = (
      latestAppliedFrameIdRef.current === 0 &&
      inFlightFramesRef.current.size === 0 &&
      !pendingFrameRef.current
    );
    if (!diff.changed && needsBootstrapFrame) {
      diff = {
        ...diff,
        changed: true,
        changedPixelRatio: 1,
        changedBboxes: [[0, 0, lowCanvas.width, lowCanvas.height]],
        mergedChangedBbox: [0, 0, lowCanvas.width, lowCanvas.height],
        bootstrapRetry: true,
      };
    }

    const changedDebug = {
      frameTimestamp: Date.now() / 1000,
      changed: diff.changed,
      changedPixelRatio: diff.changedPixelRatio,
      changedBboxes: diff.changedBboxes,
      bootstrapRetry: Boolean(diff.bootstrapRetry),
      mergedChangedBbox: diff.mergedChangedBbox
        ? scaleBbox(diff.mergedChangedBbox, lowCanvas.width, lowCanvas.height, targetWidth, targetHeight)
        : null,
    };
    setLatestChangeDebug(changedDebug);

    if (!diff.changed) {
      setFramesSkipped((value) => value + 1);
      return;
    }

    const fullCanvas = canvasRef.current;
    const fullCtx = fullCanvas.getContext("2d", { willReadFrequently: true });
    fullCanvas.width = targetWidth;
    fullCanvas.height = targetHeight;
    fullCtx.drawImage(video, 0, 0, fullCanvas.width, fullCanvas.height);

    const jpegQuality = Math.max(0.25, Math.min(0.85, Number(configRef.current?.frameJpegQuality || 0.50)));
    const frameDataUrl = await canvasToJpegDataUrl(fullCanvas, jpegQuality);
    noteChangedFrame(now);
    const clientFrameId = clientFrameIdRef.current + 1;
    clientFrameIdRef.current = clientFrameId;
    const payload = {
      session_id: sessionIdRef.current,
      timestamp: Date.now() / 1000,
      client_frame_id: clientFrameId,
      client_sent_at: Date.now() / 1000,
      client_mode_hint: mode,
      frame_jpeg_base64: frameDataUrl,
      client_change: {
        changed: true,
        changed_pixel_ratio: diff.changedPixelRatio,
        changed_bboxes: (diff.changedBboxes || []).map((box) =>
          scaleBbox(box, lowCanvas.width, lowCanvas.height, fullCanvas.width, fullCanvas.height)
        ),
        merged_changed_bbox: changedDebug.mergedChangedBbox,
      },
      recent_transcript: recentTranscriptRef.current,
    };
    changedDebug.clientFrameId = clientFrameId;
    changedDebug.fullFrameChange = diff.changedPixelRatio >= fullFrameChangeRatio;
    changedDebug.supersedingChange = diff.changedPixelRatio >= supersedeInFlightChangeRatio;

    if (latestAppliedFrameIdRef.current > 0 && (changedDebug.fullFrameChange || changedDebug.supersedingChange)) {
      abortOlderInFlightFrames(clientFrameId);
    }

    if (inFlightFramesRef.current.size >= maxInFlightFrames) {
      queueLatestFrame(payload, changedDebug, mode);
      return;
    }

    dispatchFramePayload(payload, changedDebug, mode);
  }

  const latestJson = useMemo(
    () => (
      showRawJson && latestModalResponse
        ? JSON.stringify(
            latestModalResponse,
            (key, value) => (key.endsWith("_data_url") ? "[image data url omitted]" : value),
            2
          )
        : ""
    ),
    [latestModalResponse, showRawJson]
  );
  const latestMacroTypes = macroTypes(latestModalResponse?.visual_payload);
  const latestMicroCount = microElementCount(latestModalResponse?.visual_payload);
  const latestCursor = cursorDisplay(latestModalResponse);
  const latestSummary = summaryStatus(latestModalResponse);
  const latestTimings = latestModalResponse?.debug?.stage_timings_ms || {};

  return (
    <div className="page">
      <div className="header">
        <div>
          <h1>Live Meeting Screen Understanding</h1>
          <p className="subtle">Client-side screen capture and change detection, Modal-backed heavy inference, and timestamped Supabase output.</p>
        </div>
        <div className="pill mono">{sessionId || "no-session"}</div>
      </div>

      <div className="layout">
        <section className="panel">
          <h2>Capture Console</h2>

          <div className="row" style={{ marginBottom: 12 }}>
              <input className="input" value={sessionTitle} onChange={(event) => setSessionTitle(event.target.value)} placeholder="Session title" />
              <input className="input mono" value={sessionId} onChange={(event) => setSessionId(event.target.value)} placeholder="Session UUID (optional)" />
              <div className="controls" style={{ marginBottom: 0, justifyContent: "flex-end" }}>
                <button className="button primary" onClick={startSession} disabled={isSessionActive && sessionStatus !== "error"}>Start Session + Share</button>
                <button className="button secondary" onClick={startScreenShare} disabled={!isSessionActive || isSharingScreen || modalStatus !== "ready" || sessionStatus !== "active"}>Share Screen Again</button>
                <button className="button stop" onClick={stopSession}>Stop</button>
              </div>
            </div>

          <div className="status-grid">
            {[
              ["Session", sessionStatus],
              ["Capture", captureStatus],
              ["Modal", modalStatus],
              ["Microphone", micStatus],
              ["Supabase", supabaseStatus],
            ].map(([label, value]) => (
              <div className="status-card" key={label}>
                <span className="status-label">{label}</span>
                <div className={`status-value ${statusClass(value)}`}>{value}</div>
              </div>
            ))}
          </div>

          <div className="debug-grid">
            {[
              ["Current mode", currentMode],
              ["Frames submitted", framesSubmitted],
              ["Frames completed", framesSent],
              ["Queued updates", framesQueued],
              ["Pending queue", pendingFrameCount],
              ["Dropped stale", framesDropped],
              ["In flight", inFlightFramesRef.current.size],
              ["Frames skipped", framesSkipped],
              ["Latest frame gap", latestFrameGapMs == null ? "--" : `${latestFrameGapMs.toFixed(1)} ms`],
              ["Analysis lag", latestAnalysisLagMs == null ? "--" : `${latestAnalysisLagMs.toFixed(1)} ms`],
              ["Processing", isProcessingFrame ? "yes" : "no"],
              ["Warmup", warmupDetails],
              ["Changed", latestChangeDebug?.changed ? "true" : "false"],
              ["Changed ratio", latestChangeDebug ? latestChangeDebug.changedPixelRatio?.toFixed(4) : "0.0000"],
              ["Changed bbox", latestChangeDebug?.mergedChangedBbox ? latestChangeDebug.mergedChangedBbox.join(", ") : "none"],
              ["Timestamp", latestChangeDebug ? formatTs(latestChangeDebug.frameTimestamp) : "--"],
            ].map(([label, value]) => (
              <div className="debug-card" key={label}>
                <span className="debug-label">{label}</span>
                <div className="mono">{String(value)}</div>
              </div>
            ))}
          </div>

          <div className="preview-wrap">
            <video ref={videoRef} className="preview" muted playsInline />
            {!isSharingScreen && <div className="preview-overlay">Screen preview will appear here after screen sharing starts.</div>}
          </div>

          <div className="stack" style={{ marginTop: 16 }}>
            <div className="mini-panel">
              <div className="section-header">
                <h3>Transcript Input</h3>
                <span className="timestamp">Auto via Deepgram, manual fallback below</span>
              </div>
              <div className="transcript-form">
                <input className="input" value={speakerDraft} onChange={(event) => setSpeakerDraft(event.target.value)} placeholder="Speaker" />
                <textarea className="textarea" value={transcriptDraft} onChange={(event) => setTranscriptDraft(event.target.value)} placeholder="Paste or type a transcript chunk here..." />
                <div className="controls" style={{ marginBottom: 0 }}>
                  <button className="button secondary" onClick={addTranscriptChunk} disabled={!isSessionActive || !transcriptDraft.trim()}>
                    Add Transcript Chunk
                  </button>
                </div>
              </div>
            </div>

            {errorMessage && (
              <div className="mini-panel">
                <span className="status-label">Last error</span>
                <div className="bad">{errorMessage}</div>
              </div>
            )}
          </div>

          <canvas ref={canvasRef} style={{ display: "none" }} />
          <canvas ref={lowResCanvasRef} style={{ display: "none" }} />
        </section>

        <section className="panel stack">
          <div className="mini-panel">
            <div className="section-header">
              <h3>Screen Type</h3>
              <span className="pill">{latestModalResponse?.mode || currentMode || "unknown"}</span>
            </div>
            <div className="list">
              <div className="list-item"><strong>Confidence:</strong> {latestModalResponse?.mode_confidence?.toFixed?.(3) ?? "--"}</div>
              <div className="list-item"><strong>Decision path:</strong> <span className="mono">{latestModalResponse?.decision_path || latestModalResponse?.raw_record?.confidence?.route?.decision_path || "--"}</span></div>
            </div>
          </div>

          <div className="mini-panel">
            <h3>Transcript</h3>
            {latestTranscript ? (
              <div className="transcript-list">
                <div className="transcript-item">
                  <div className="timestamp">{formatTs(latestTranscript.start_time)} - {formatTs(latestTranscript.end_time)} {latestTranscript.speaker ? `| ${latestTranscript.speaker}` : ""}</div>
                  <div>{latestTranscript.text}</div>
                </div>
              </div>
            ) : (
              <div className="empty-state">No transcript chunks yet.</div>
            )}
            <div className="transcript-list" style={{ marginTop: 10 }}>
              {transcriptHistory.slice().reverse().slice(0, 6).map((item) => (
                <div className="transcript-item" key={transcriptIdentity(item)}>
                  <div className="timestamp">{formatTs(item.start_time)} - {formatTs(item.end_time)} {item.speaker ? `| ${item.speaker}` : ""}</div>
                  <div>{item.text}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="mini-panel">
            <h3>Screen Layout / OCR</h3>
            {!latestModalResponse && <div className="empty-state">Waiting for the first processed frame.</div>}

            {latestModalResponse?.mode === "code" && (
              <div className="list">
                <div className="list-item"><strong>OCR text</strong><div className="mono">{latestModalResponse.visual_payload?.ocr_text || "No OCR text returned."}</div></div>
                <div className="list-item"><strong>Changed bbox</strong><div className="mono">{JSON.stringify(latestModalResponse.visual_payload?.changed_bbox || latestModalResponse.debug?.client_change?.merged_changed_bbox || null)}</div></div>
              </div>
            )}

            {latestModalResponse?.mode === "slide" && (
              <div className="list">
                <div className="list-item"><strong>Full text</strong><div>{latestModalResponse.visual_payload?.full_text || "No slide text returned."}</div></div>
                <div className="list-item"><strong>Layout blocks</strong><div className="mono">{JSON.stringify(latestModalResponse.visual_payload?.layout || [], null, 2)}</div></div>
              </div>
            )}

            {latestModalResponse?.mode === "demo" && (
              <div className="list">
                <div className="list-item"><strong>Changed region</strong><div className="mono">{JSON.stringify(latestModalResponse.visual_payload?.changed_bbox || latestModalResponse.debug?.client_change?.merged_changed_bbox || null)}</div></div>
                <div className="list-item"><strong>Macro layout types</strong><div className="mono">{latestMacroTypes.length ? latestMacroTypes.join(", ") : "none"}</div></div>
                <div className="list-item"><strong>Micro elements seen</strong><div className="mono">{String(latestMicroCount)}</div></div>
                <div className="list-item"><strong>Cursor</strong><div className="mono">{`visible=${latestCursor.visible} | type=${latestCursor.type} | micro=${latestCursor.micro} | macro=${latestCursor.macro}`}</div></div>
              </div>
            )}

            {latestModalResponse?.mode === "other" && (
              <div className="empty-state">No meaningful visual content detected.</div>
            )}
          </div>

          <div className="mini-panel">
            <h3>Latest Combined Summary</h3>
            <div className="timestamp">LLM: {latestSummary.provider} / {latestSummary.model}</div>
            <div className="summary-text">{latestSummary.text}</div>
            {latestSummary.detail ? (
              <div className="timestamp" style={{ marginTop: 8 }}>
                {latestSummary.detail}
              </div>
            ) : null}
          </div>

          <div className="mini-panel">
            <h3>Annotated Detection View</h3>
            {latestModalResponse?.annotated_image_data_url ? (
              <div className="annotation-compare">
                <div>
                  <div className="timestamp">Analyzed input frame</div>
                  <img
                    src={latestModalResponse.analyzed_frame_data_url}
                    alt="The captured frame sent for analysis"
                  />
                </div>
                <div>
                  <div className="timestamp">
                    Annotated output
                    {latestModalResponse.client_analysis_lag_ms != null ? ` | lag ${formatMs(latestModalResponse.client_analysis_lag_ms)}` : ""}
                  </div>
                  <img
                    src={latestModalResponse.annotated_image_data_url}
                    alt="Annotated macro, micro, and cursor detections"
                  />
                </div>
              </div>
            ) : (
              <div className="empty-state">Annotated image will appear after the first processed frame.</div>
            )}
          </div>

          <div className="mini-panel">
            <div className="section-header">
              <h3>Complete JSON Output</h3>
              <button className="button secondary" onClick={() => setShowRawJson((value) => !value)}>
                {showRawJson ? "Hide" : "Show"}
              </button>
            </div>
            {showRawJson ? (
              <div className="json-box mono">{latestJson || "No response yet."}</div>
            ) : (
              <div className="empty-state">Hidden while live updates are running.</div>
            )}
          </div>

          <div className="mini-panel">
            <h3>Debug</h3>
            <div className="list">
              <div className="list-item"><strong>Processor:</strong> <span className="mono">{latestModalResponse?.processor_used || "--"}</span></div>
              <div className="list-item"><strong>Macro detector ran:</strong> {String((latestModalResponse?.debug?.macro_detector_ran ?? latestModalResponse?.debug?.rfdetr_ran) || false)}</div>
              <div className="list-item"><strong>YOLO layout ran:</strong> {String(latestModalResponse?.debug?.macro_detector_ran || false)}</div>
              <div className="list-item"><strong>DocLayout ran:</strong> {String(latestModalResponse?.debug?.doclayout_ran || false)}</div>
              <div className="list-item"><strong>Surya ran:</strong> {String(latestModalResponse?.debug?.surya_ran || false)}</div>
              <div className="list-item"><strong>OCR ran:</strong> {String(latestModalResponse?.debug?.ocr_ran || false)}</div>
              <div className="list-item"><strong>Processing time:</strong> {latestModalResponse?.debug?.processing_time_ms?.toFixed?.(1) ?? "--"} ms</div>
              <div className="list-item"><strong>Endpoint total:</strong> <span className="mono">{formatMs(latestTimings.endpoint_total_ms)}</span></div>
              <div className="list-item"><strong>Proxy:</strong> <span className="mono">{formatMs(latestTimings.modal_proxy_ms)}</span></div>
              <div className="list-item"><strong>Macro / OCR:</strong> <span className="mono">{formatMs(latestTimings.macro_layout_ms)} / {formatMs(latestTimings.ocr_ms)}</span></div>
              <div className="list-item"><strong>DocLayout:</strong> <span className="mono">{formatMs(latestTimings.doclayout_ms)}</span></div>
              <div className="list-item"><strong>Summary / Annotation:</strong> <span className="mono">{formatMs(latestTimings.summary_ms)} / {formatMs(latestTimings.annotation_ms)}</span></div>
            </div>
          </div>

          {config?.enableDebugJson && null}
        </section>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
