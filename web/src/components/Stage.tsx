import { useEffect, useState, type CSSProperties, type ReactNode } from "react";
import type { StageState } from "../lib/run";

export const LANE_VAR: Record<string, string> = {
  voice: "var(--voice)", words: "var(--words)", face: "var(--face)", fusion: "var(--fusion)",
};

/** Live elapsed time while a stage runs (browser clock since its stage_start arrived);
 *  the server-measured duration once it is done. */
export function StageTimer({ stage }: { stage: StageState }) {
  const [now, setNow] = useState(performance.now());
  useEffect(() => {
    if (stage.status !== "running") return;
    const id = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(id);
  }, [stage.status]);
  if (stage.status === "running" && stage.startedAt !== undefined)
    return <span className="mono">{((now - stage.startedAt) / 1000).toFixed(1)} s</span>;
  if (stage.status === "done" && stage.seconds !== undefined)
    return <span className="mono">{stage.seconds < 0.05 ? "<0.1" : stage.seconds.toFixed(1)} s</span>;
  if (stage.status === "skipped") return <span>skipped</span>;
  if (stage.status === "failed") return <span style={{ color: "var(--danger)" }}>failed</span>;
  return <span className="muted">waiting</span>;
}

interface CardProps {
  lane: keyof typeof LANE_VAR;
  step: string;
  title: string;
  stage: StageState;
  className?: string;
  busyText?: string;
  ready: boolean;        // true when this card's data has arrived
  skippedText?: string;
  children?: ReactNode;
}

/** One stage card: waiting -> running (live timer) -> done (reveals its data). */
export function Card({ lane, step, title, stage, className = "", busyText, ready, skippedText, children }: CardProps) {
  const status = ready ? "done" : stage.status;
  return (
    <section className={`card ${status} ${className}`} style={{ "--lane": LANE_VAR[lane] } as CSSProperties}>
      <header className="card-head">
        <span className="step">{step}</span>
        <h4>{title}</h4>
        <span className="timer"><StageTimer stage={stage} /></span>
      </header>
      {ready ? (
        <div className="card-body">{children}</div>
      ) : stage.status === "skipped" ? (
        <div className="placeholder">{skippedText ?? "skipped"}</div>
      ) : (
        <div className={`placeholder ${stage.status === "running" ? "busy" : ""}`}>
          {stage.status === "running" ? busyText ?? "computing…" : stage.status === "failed" ? "stopped" : "waiting for the previous stage"}
        </div>
      )}
    </section>
  );
}
