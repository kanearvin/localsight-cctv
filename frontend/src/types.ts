// Mirrors the FastAPI JSON responses in backend/main.py.

export type Provider = 'ollama' | 'gemini';

/** ROI vertex in the UI's normalized 0–1000 coordinate space. */
export interface Point {
  x: number;
  y: number;
}

export interface SecurityEvent {
  id: number;
  startSec: number;
  endSec: number;
  timestampStr: string;
  endStr: string;
  confidence: number;
  summary: string;
  tags: string[];
  matched: boolean;
}

export interface IndexResponse {
  video_id: string;
  playable_url: string;
  event_count: number;
  events: SecurityEvent[];
}

export interface VideoDetailResponse extends IndexResponse {
  original_filename: string;
  duration_sec: number | null;
  roi: Point[] | null;
}

export interface SearchRequest {
  video_id: string;
  query: string;
  provider: Provider;
  api_key?: string | null;
  ollama_model?: string;
}

export interface SearchResponse {
  results: SecurityEvent[];
}

export interface HealthResponse {
  status: 'ok';
  ffmpeg: boolean;
  ollama_reachable: boolean;
}
