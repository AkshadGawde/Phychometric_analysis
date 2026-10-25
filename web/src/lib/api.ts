import type { Sample } from "../types";

// Where the API lives. Empty in development (Vite proxies /api to localhost:8000) and when the
// API serves the built app itself (make demo). On Vercel: VITE_API_BASE=https://<your-api>.onrender.com
const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const api = (path: string) => `${BASE}${path}`;

export interface Health { ready: boolean; status: string; error: string | null; device: string | null; queue: number; whisper: string }

export async function getHealth(): Promise<Health> {
  const res = await fetch(api("/api/health"));
  if (!res.ok) throw new Error(`API not reachable (${res.status})`);
  return res.json();
}

export async function getSamples(): Promise<Sample[]> {
  const res = await fetch(api("/api/samples"));
  return res.ok ? res.json() : [];
}

export async function getModelCard(): Promise<any> {
  const res = await fetch(api("/api/model-card"));
  if (!res.ok) throw new Error("model card not available");
  return res.json();
}

/** Upload a file; returns the URL of its event stream. */
export async function startJob(file: Blob, filename: string): Promise<string> {
  const body = new FormData();
  body.append("file", file, filename);
  const res = await fetch(api("/api/analyze"), { method: "POST", body });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? "upload failed");
  }
  const { job_id } = await res.json();
  return api(`/api/jobs/${job_id}/events`);
}

export const sampleEventsUrl = (name: string, speed = 1) => api(`/api/samples/${name}/events?speed=${speed}`);
export const sampleMediaUrl = (name: string, file: string) => api(`/api/samples/${name}/media/${file}`);
