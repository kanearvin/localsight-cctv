import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Crosshair,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Shield,
  Sparkles,
  Trash2,
  Undo2,
  Upload,
  X,
} from 'lucide-react';
import { ApiError, getHealth, getVideo, indexVideo, reindexVideo, resolveUrl, searchEvents } from './api';
import type { HealthResponse, IndexResponse, Point, Provider, SecurityEvent } from './types';

const GEMINI_KEY_STORAGE = 'localsight.geminiKey'; // sessionStorage only — never persisted to disk/backend
const LAST_VIDEO_STORAGE = 'localsight.lastVideoId';

interface ContentBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

function safeStorage(kind: 'session' | 'local'): Storage | null {
  try {
    return kind === 'session' ? window.sessionStorage : window.localStorage;
  } catch {
    return null;
  }
}

const errorMessage = (err: unknown) =>
  err instanceof ApiError || err instanceof Error ? err.message : 'Unexpected error';

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [roiMode, setRoiMode] = useState<boolean>(false);
  const [roiPoints, setRoiPoints] = useState<Point[]>([]);
  const [appliedRoi, setAppliedRoi] = useState<Point[]>([]);
  const [query, setQuery] = useState<string>('');
  const [provider, setProvider] = useState<Provider>('ollama');
  const [apiKey, setApiKey] = useState<string>(() => safeStorage('session')?.getItem(GEMINI_KEY_STORAGE) ?? '');
  const [busy, setBusy] = useState<null | 'indexing' | 'rescanning' | 'searching'>(null);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [searched, setSearched] = useState<boolean>(false);
  const [activeEventId, setActiveEventId] = useState<number | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [box, setBox] = useState<ContentBox | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<SVGSVGElement | null>(null);

  // ---- health + restore last session ----
  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    const lastId = safeStorage('local')?.getItem(LAST_VIDEO_STORAGE);
    if (lastId) {
      getVideo(lastId)
        .then((v) => {
          applyIndexResponse(v);
          setFileName(v.original_filename);
          if (v.roi) {
            setRoiPoints(v.roi);
            setAppliedRoi(v.roi);
          }
        })
        .catch(() => safeStorage('local')?.removeItem(LAST_VIDEO_STORAGE));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const store = safeStorage('session');
    if (!store) return;
    if (apiKey) store.setItem(GEMINI_KEY_STORAGE, apiKey);
    else store.removeItem(GEMINI_KEY_STORAGE);
  }, [apiKey]);

  // ---- keep the SVG overlay aligned with the rendered video frame (object-contain letterboxing) ----
  const recomputeBox = useCallback(() => {
    const stage = stageRef.current;
    const video = videoRef.current;
    if (!stage || !video || !video.videoWidth || !video.videoHeight) return;
    const sw = stage.clientWidth;
    const sh = stage.clientHeight;
    const aspect = video.videoWidth / video.videoHeight;
    let width = sw;
    let height = sw / aspect;
    if (height > sh) {
      height = sh;
      width = sh * aspect;
    }
    setBox({ left: (sw - width) / 2, top: (sh - height) / 2, width, height });
  }, []);

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const ro = new ResizeObserver(recomputeBox);
    ro.observe(stage);
    return () => ro.disconnect();
  }, [recomputeBox, videoUrl]);

  function applyIndexResponse(data: IndexResponse) {
    setVideoId(data.video_id);
    setVideoUrl(resolveUrl(data.playable_url));
    setEvents(data.events);
    setSearched(false);
    setActiveEventId(null);
    safeStorage('local')?.setItem(LAST_VIDEO_STORAGE, data.video_id);
  }

  // ---- actions ----
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setFileName(file.name);
    setError(null);
    setBusy('indexing');
    setRoiPoints([]);
    setAppliedRoi([]);
    setRoiMode(false);
    setVideoUrl(null);
    setBox(null);
    try {
      applyIndexResponse(await indexVideo(file));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const handleRescan = async () => {
    if (!videoId) return;
    setError(null);
    setBusy('rescanning');
    setRoiMode(false);
    try {
      const roi = roiPoints.length >= 3 ? roiPoints : null;
      applyIndexResponse(await reindexVideo(videoId, roi));
      setAppliedRoi(roi ?? []);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!roiMode || !overlayRef.current) return;
    const rect = overlayRef.current.getBoundingClientRect();
    const clamp = (v: number) => Math.min(1000, Math.max(0, Math.round(v)));
    const x = clamp(((e.clientX - rect.left) / rect.width) * 1000);
    const y = clamp(((e.clientY - rect.top) / rect.height) * 1000);
    setRoiPoints((prev) => [...prev, { x, y }]);
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoId || !query.trim()) return;
    if (provider === 'gemini' && !apiKey.trim()) {
      setError('Enter your Gemini API key to use Cloud BYOK mode.');
      return;
    }
    setError(null);
    setBusy('searching');
    try {
      const data = await searchEvents({
        video_id: videoId,
        query: query.trim(),
        provider,
        api_key: provider === 'gemini' ? apiKey.trim() : null,
      });
      setEvents(data.results);
      setSearched(true);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const jumpToTimestamp = (ev: SecurityEvent) => {
    setActiveEventId(ev.id);
    const v = videoRef.current;
    if (v) {
      v.currentTime = ev.startSec;
      void v.play().catch(() => undefined);
    }
  };

  // ---- derived ----
  const polygonString = roiPoints.map((p) => `${p.x},${p.y}`).join(' ');
  const matched = events.filter((ev) => ev.matched);
  const errored = events.filter((ev) => ev.tags.includes('error')).length;
  const effectiveRoi = roiPoints.length >= 3 ? roiPoints : [];
  const roiDirty = JSON.stringify(effectiveRoi) !== JSON.stringify(appliedRoi);
  const isBusy = busy !== null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-6 font-sans">
      <header className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-400">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">LocalSight CCTV Scrubber</h1>
            <p className="text-xs text-slate-400">100% Local-First Home Footage Indexing & Semantic Event Search</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <HealthBadge health={health} />
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value as Provider)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200"
          >
            <option value="ollama">Offline Ollama (qwen2.5-vl:7b)</option>
            <option value="gemini">Cloud BYOK (Gemini 2.5 Flash)</option>
          </select>

          {provider === 'gemini' && (
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Gemini API Key"
              autoComplete="off"
              title="Kept in this browser tab's sessionStorage only. Only 1440x540 contact sheets are sent to Gemini."
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 w-44"
            />
          )}

          <label
            className={`cursor-pointer inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-sm font-medium transition ${
              isBusy ? 'opacity-50 pointer-events-none' : ''
            }`}
          >
            {busy === 'indexing' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            <span>{busy === 'indexing' ? 'Indexing…' : fileName ? 'Switch CCTV File' : 'Load Local CCTV Video'}</span>
            <input
              type="file"
              accept="video/*,.dav,.mkv,.h264,.h265"
              onChange={handleFileUpload}
              className="hidden"
              disabled={isBusy}
            />
          </label>
        </div>
      </header>

      {error && (
        <div className="max-w-7xl mx-auto mt-4 flex items-start gap-2 rounded-xl border border-red-900 bg-red-950/40 p-3 text-sm text-red-300">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span className="flex-1 break-words">{error}</span>
          <button type="button" onClick={() => setError(null)} className="text-red-400 hover:text-red-200" title="Dismiss">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6 pt-6">
        <div className="lg:col-span-7 space-y-4 min-w-0">
          <div
            ref={stageRef}
            className="relative aspect-video bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center"
          >
            {videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  preload="metadata"
                  onLoadedMetadata={(e) => {
                    setDuration(e.currentTarget.duration || 0);
                    recomputeBox();
                  }}
                  className="w-full h-full object-contain"
                />
                {box && (
                  <svg
                    ref={overlayRef}
                    viewBox="0 0 1000 1000"
                    preserveAspectRatio="none"
                    onClick={handleCanvasClick}
                    style={{ left: box.left, top: box.top, width: box.width, height: box.height }}
                    className={`absolute ${roiMode ? 'cursor-crosshair pointer-events-auto' : 'pointer-events-none'}`}
                  >
                    {roiMode && <rect x="0" y="0" width="1000" height="1000" fill="rgba(2,6,23,0.25)" />}
                    {roiPoints.length > 1 && (
                      <polygon
                        points={polygonString}
                        fill="rgba(16, 185, 129, 0.2)"
                        stroke="#10b981"
                        strokeWidth="4"
                        vectorEffect="non-scaling-stroke"
                      />
                    )}
                    {roiPoints.map((pt, i) => (
                      <ellipse
                        key={i}
                        cx={pt.x}
                        cy={pt.y}
                        rx={box.width ? (6 * 1000) / box.width : 8}
                        ry={box.height ? (6 * 1000) / box.height : 8}
                        fill="#10b981"
                      />
                    ))}
                  </svg>
                )}
              </>
            ) : (
              <div className="text-center p-8 space-y-3">
                {busy === 'indexing' ? (
                  <>
                    <Loader2 className="w-10 h-10 text-emerald-500 mx-auto animate-spin" />
                    <p className="text-sm text-slate-300 font-medium">Remuxing & scanning {fileName}…</p>
                    <p className="text-xs text-slate-500">Large NVR exports can take a minute. Everything runs locally.</p>
                  </>
                ) : (
                  <>
                    <Upload className="w-10 h-10 text-slate-600 mx-auto" />
                    <p className="text-sm text-slate-300 font-medium">Load an exported NVR/SD-card clip (.mp4, .mkv, .dav)</p>
                    <p className="text-xs text-slate-500 max-w-sm mx-auto">
                      Video stays strictly on your machine. Motion filtering strips 90%+ of static frames before AI indexing.
                    </p>
                  </>
                )}
              </div>
            )}
          </div>

          {videoUrl && duration > 0 && (
            <Timeline duration={duration} events={events} activeId={activeEventId} onJump={jumpToTimestamp} />
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/70 border border-slate-800 p-3.5 rounded-xl">
            <div className="flex items-center gap-2 text-xs text-slate-300">
              <Crosshair className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                <strong>Focus Zone (ROI):</strong>{' '}
                {roiMode
                  ? `Click the video to add points (${roiPoints.length} placed, need ≥ 3)`
                  : appliedRoi.length >= 3
                    ? `Active (${appliedRoi.length}-point mask ignoring street background)`
                    : 'Full frame (Draw a zone around your porch or parked car)'}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={!videoUrl || isBusy}
                onClick={() => setRoiMode(!roiMode)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition disabled:opacity-50 ${
                  roiMode ? 'bg-emerald-500 text-slate-950' : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                }`}
              >
                {roiMode ? 'Done Drawing Zone' : 'Draw Detection Zone'}
              </button>
              {roiPoints.length > 0 && (
                <>
                  <button
                    type="button"
                    onClick={() => setRoiPoints((p) => p.slice(0, -1))}
                    className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 transition"
                    title="Undo last point"
                  >
                    <Undo2 className="w-4 h-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setRoiPoints([])}
                    className="p-1.5 rounded-lg bg-slate-800 hover:bg-red-950/60 text-slate-400 hover:text-red-400 transition"
                    title="Clear ROI"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </>
              )}
              {videoId && roiDirty && (roiPoints.length >= 3 || roiPoints.length === 0) && (
                <button
                  type="button"
                  onClick={handleRescan}
                  disabled={isBusy}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 transition"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${busy === 'rescanning' ? 'animate-spin' : ''}`} />
                  {roiPoints.length >= 3 ? 'Re-scan with zone' : 'Re-scan full frame'}
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 flex flex-col space-y-4 min-w-0">
          <form onSubmit={handleSearch} className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Ask Your Footage</label>
            <div className="relative flex items-center">
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder='e.g., "delivery parcel" or "person near car"'
                className="w-full bg-slate-900 border border-slate-700 focus:border-emerald-500 rounded-xl pl-10 pr-28 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none"
              />
              <button
                type="submit"
                disabled={isBusy || !videoId || !query.trim()}
                className="absolute right-2 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg text-xs font-semibold transition flex items-center gap-1.5"
              >
                {busy === 'searching' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                {busy === 'searching' ? 'Scanning…' : 'Find'}
              </button>
            </div>
          </form>

          <div className="flex-1 space-y-3 overflow-y-auto max-h-[540px] pr-1">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>
                {searched
                  ? `${matched.length} of ${events.length} events match`
                  : `Detected Motion Events (${events.length})`}
                {errored > 0 && <span className="text-amber-400"> · {errored} failed to evaluate</span>}
              </span>
              <span>Click card to jump</span>
            </div>

            {videoId && matched.length === 0 && !isBusy && (
              <p className="text-sm text-slate-500 py-6 text-center">
                {searched ? 'No events matched that query.' : 'No motion detected in this clip.'}
              </p>
            )}

            {matched.map((ev) => (
              <div
                key={ev.id}
                role="button"
                tabIndex={0}
                onClick={() => jumpToTimestamp(ev)}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && jumpToTimestamp(ev)}
                className={`p-4 rounded-xl border cursor-pointer transition space-y-2 ${
                  activeEventId === ev.id
                    ? 'bg-emerald-950/30 border-emerald-500'
                    : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="inline-flex items-center gap-1.5 text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-md">
                    <Clock className="w-3.5 h-3.5" />
                    {ev.timestampStr} – {ev.endStr}
                  </span>
                  <span className="inline-flex items-center gap-1 text-xs text-slate-400">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                    {Math.round(ev.confidence * 100)}% match
                  </span>
                </div>

                <p className="text-sm text-slate-200 leading-relaxed">{ev.summary}</p>

                <div className="flex items-center justify-between gap-2 pt-1">
                  <div className="flex flex-wrap gap-1.5">
                    {ev.tags.map((tag) => (
                      <span key={tag} className="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                        #{tag}
                      </span>
                    ))}
                  </div>
                  <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-400 shrink-0">
                    <Play className="w-3 h-3 fill-current" /> Jump
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

function HealthBadge({ health }: { health: HealthResponse | null }) {
  if (!health) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-red-400">
        <span className="w-2 h-2 rounded-full bg-red-500" /> Backend offline
      </span>
    );
  }
  const Dot = ({ ok, label }: { ok: boolean; label: string }) => (
    <span className="inline-flex items-center gap-1.5" title={ok ? `${label} ready` : `${label} unavailable`}>
      <span className={`w-2 h-2 rounded-full ${ok ? 'bg-emerald-500' : 'bg-amber-500'}`} />
      {label}
    </span>
  );
  return (
    <span className="inline-flex items-center gap-3 text-xs text-slate-400">
      <Dot ok={health.ffmpeg} label="FFmpeg" />
      <Dot ok={health.ollama_reachable} label="Ollama" />
    </span>
  );
}

function Timeline({
  duration,
  events,
  activeId,
  onJump,
}: {
  duration: number;
  events: SecurityEvent[];
  activeId: number | null;
  onJump: (ev: SecurityEvent) => void;
}) {
  return (
    <div className="relative h-6 rounded-lg bg-slate-900 border border-slate-800 overflow-hidden" aria-label="Event timeline">
      {events.map((ev) => {
        const left = (Math.max(0, ev.startSec) / duration) * 100;
        const width = Math.max(0.6, ((ev.endSec - ev.startSec + 1) / duration) * 100);
        const color = !ev.matched
          ? 'bg-slate-700/60'
          : activeId === ev.id
            ? 'bg-emerald-300'
            : 'bg-emerald-500/80 hover:bg-emerald-400';
        return (
          <button
            key={ev.id}
            type="button"
            title={`${ev.timestampStr} – ${ev.endStr}`}
            onClick={() => onJump(ev)}
            style={{ left: `${left}%`, width: `${width}%` }}
            className={`absolute top-1 bottom-1 rounded-sm transition ${color}`}
          />
        );
      })}
    </div>
  );
}
