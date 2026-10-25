import { useEffect, useReducer, useRef } from "react";
import type { ServerEvent, StageDataMap, StageName } from "../types";

// One run = one event stream. Every stage card renders only what its own event delivered.

export type StageStatus = "waiting" | "running" | "done" | "skipped" | "failed";

export interface StageState {
  status: StageStatus;
  startedAt?: number;      // browser clock when its stage_start arrived (for the live timer)
  seconds?: number;        // the real duration measured on the server
  data?: unknown;
}

export interface RunState {
  phase: "connecting" | "queued" | "running" | "done" | "error";
  queue?: { position: number; status: string; ready: boolean };
  stages: Record<StageName, StageState>;
  error?: { stage: StageName; message: string };
  total?: number;
}

const STAGES: StageName[] = ["ingest", "audio_clean", "audio_features", "transcribe", "visual", "unimodal", "fusion", "explain"];

const initial = (): RunState => ({
  phase: "connecting",
  stages: Object.fromEntries(STAGES.map((s) => [s, { status: "waiting" }])) as Record<StageName, StageState>,
});

type Action = { kind: "event"; event: ServerEvent; at: number } | { kind: "lost" } | { kind: "reset" };

function reduce(state: RunState, action: Action): RunState {
  if (action.kind === "reset") return initial();
  if (action.kind === "lost") {
    if (state.phase === "done" || state.phase === "error") return state;
    return { ...state, phase: "error", error: { stage: "ingest", message: "the connection to the server was lost" } };
  }
  const e = action.event;
  switch (e.type) {
    case "queued":
      return { ...state, phase: "queued", queue: { position: e.position, status: e.status, ready: e.ready } };
    case "stage_start":
      return { ...state, phase: "running", stages: { ...state.stages, [e.stage]: { status: "running", startedAt: action.at } } };
    case "stage": {
      const skipped = e.stage === "visual" && (e.data as { available?: boolean }).available === false;
      return {
        ...state, phase: "running",
        stages: { ...state.stages, [e.stage]: { ...state.stages[e.stage], status: skipped ? "skipped" : "done", seconds: e.seconds, data: e.data } },
      };
    }
    case "error":
      return { ...state, phase: "error", error: { stage: e.stage, message: e.message },
        stages: { ...state.stages, [e.stage]: { ...state.stages[e.stage], status: "failed" } } };
    case "done":
      return { ...state, phase: "done", total: e.seconds };
  }
}

export function useRun(url: string | null) {
  const [state, dispatch] = useReducer(reduce, undefined, initial);
  const finished = useRef(false);

  useEffect(() => {
    if (!url) return;
    dispatch({ kind: "reset" });
    finished.current = false;
    const source = new EventSource(url);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as ServerEvent;
      dispatch({ kind: "event", event, at: performance.now() });
      if (event.type === "done" || event.type === "error") {
        finished.current = true;
        source.close();
      }
    };
    source.onerror = () => {
      source.close();
      if (!finished.current) dispatch({ kind: "lost" });
    };
    return () => source.close();
  }, [url]);

  const data = <K extends StageName>(stage: K) => state.stages[stage].data as StageDataMap[K] | undefined;
  return { state, data };
}
