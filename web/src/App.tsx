import { useEffect, useState } from "react";
import { Capture, type RunInput } from "./components/Capture";
import { HowItWorks } from "./components/HowItWorks";
import { RunView, type RunSpec } from "./components/RunView";
import { getHealth, sampleEventsUrl, sampleMediaUrl, startJob, type Health } from "./lib/api";
import { PlayheadProvider } from "./lib/playhead";

type Page = "analyze" | "how";
const pageFromHash = (): Page => (window.location.hash === "#/how" ? "how" : "analyze");

function useHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    let stop = false;
    let id = 0;
    const poll = async () => {
      try {
        const h = await getHealth();
        if (stop) return;
        setHealth(h); setDown(false);
        id = window.setTimeout(poll, h.ready ? 15000 : 1500);
      } catch {
        if (stop) return;
        setDown(true);
        id = window.setTimeout(poll, 3000);
      }
    };
    void poll();
    return () => { stop = true; clearTimeout(id); };
  }, []);
  return { health, down };
}

function Status({ health, down }: { health: Health | null; down: boolean }) {
  if (down) return <div className="status"><i className="dot bad" /><span>API offline: start uvicorn on port 8000</span></div>;
  if (!health) return <div className="status"><i className="dot" /><span>connecting…</span></div>;
  if (health.error) return <div className="status"><i className="dot bad" /><span>models failed to load</span></div>;
  if (!health.ready) return <div className="status"><i className="dot busy" /><span>{health.status}…</span></div>;
  return <div className="status"><i className="dot ok" /><span className="mono">models ready · {health.device}{health.queue ? ` · ${health.queue} in queue` : ""}</span></div>;
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash);
  const [run, setRun] = useState<RunSpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { health, down } = useHealth();

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = (p: Page) => { window.location.hash = p === "how" ? "#/how" : "#/"; setPage(p); };

  const start = async (input: RunInput) => {
    setError(null);
    if (input.kind === "sample") {
      const s = input.sample;
      const file = s.video ?? s.audio;
      setRun({ eventsUrl: sampleEventsUrl(s.name), mediaUrl: file ? sampleMediaUrl(s.name, file) : null,
        isVideo: !!s.video, title: s.title, replay: true });
    } else {
      try {
        const eventsUrl = await startJob(input.blob, input.filename);
        setRun({ eventsUrl, mediaUrl: input.mediaUrl, isVideo: input.isVideo, title: input.title, replay: false });
      } catch (e) {
        setError((e as Error).message);
        return;
      }
    }
    window.scrollTo({ top: 0 });
  };

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <i style={{ background: "var(--face)" }} /><i style={{ background: "var(--voice)" }} />
            <i style={{ background: "var(--words)" }} /><i style={{ background: "var(--fusion)" }} />
          </span>
          First Impressions <span className="muted" style={{ fontWeight: 400 }}>· pipeline visualizer</span>
        </div>
        <nav className="nav">
          <button aria-current={page === "analyze" ? "page" : undefined} onClick={() => go("analyze")}>Analyze</button>
          <button aria-current={page === "how" ? "page" : undefined} onClick={() => go("how")}>How it works</button>
        </nav>
        <Status health={health} down={down} />
      </header>
      <main>
        {page === "how" ? <HowItWorks /> : run ? (
          <PlayheadProvider key={run.eventsUrl}>
            <RunView spec={run} onNew={() => setRun(null)} />
          </PlayheadProvider>
        ) : (
          <>
            <div className="hero">
              <span className="upper muted">Multimodal psychometrics · research prototype</span>
              <h1>Watch a machine form a first impression. <em>Then see how unsure it is.</em></h1>
              <p>
                Record your voice or a 15-second video. The trained pipeline runs on it stage by stage (voice, words, face, fusion)
                and every picture and number on the screen comes from your own input. The result is how strangers would
                probably <em>rate</em> you, not who you are.
              </p>
              <div className="lanes-legend">
                <span className="chip"><i className="dot" style={{ background: "var(--voice)" }} />Voice · AUC 0.72</span>
                <span className="chip"><i className="dot" style={{ background: "var(--words)" }} />Words · 0.55</span>
                <span className="chip"><i className="dot" style={{ background: "var(--face)" }} />Face · 0.66</span>
                <span className="chip"><i className="dot" style={{ background: "var(--fusion)" }} />Fusion · 0.73</span>
              </div>
            </div>
            {error && <div className="error-box" style={{ marginBottom: 14 }}>{error}</div>}
            <Capture onStart={start} />
          </>
        )}
      </main>
    </>
  );
}
