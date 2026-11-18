import type {
  HealthResponse,
  IndexResponse,
  OpenRouterModelsResponse,
  Point,
  SearchRequest,
  SearchResponse,
  VideoDetailResponse,
} from './types';

/** Empty by default so requests go through the Vite dev proxy (/api -> 127.0.0.1:8000). */
const API_BASE: string = import.meta.env.VITE_API_BASE ?? '';

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError('Cannot reach the LocalSight backend on port 8000. Is uvicorn running?', 0);
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === 'string') detail = body.detail;
      else if (body?.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export const resolveUrl = (path: string): string => (/^https?:\/\//.test(path) ? path : `${API_BASE}${path}`);

export const getHealth = () => request<HealthResponse>('/api/health');

export function indexVideo(file: File, roi?: Point[]): Promise<IndexResponse> {
  const form = new FormData();
  form.append('file', file);
  if (roi && roi.length >= 3) form.append('roi_json', JSON.stringify(roi));
  return request<IndexResponse>('/api/index', { method: 'POST', body: form });
}

export const reindexVideo = (videoId: string, roi: Point[] | null) =>
  request<IndexResponse>(`/api/videos/${encodeURIComponent(videoId)}/reindex`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ roi: roi && roi.length >= 3 ? roi : null }),
  });

export const getVideo = (videoId: string) =>
  request<VideoDetailResponse>(`/api/videos/${encodeURIComponent(videoId)}`);

export const searchEvents = (req: SearchRequest) =>
  request<SearchResponse>('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

export const getOpenRouterModels = () => request<OpenRouterModelsResponse>('/api/openrouter/models');
