import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

// ---- Wire these to your app ------------------------------------------------
const API_URL = import.meta.env.VITE_API_URL;
const LOGIN_PATH = "/login";
const TOKEN_KEY = "insurai_token";
const APP_ROUTE = "/chat";
const DEMO_USER = import.meta.env.VITE_DEMO_USER;
const DEMO_PASS = import.meta.env.VITE_DEMO_PASS;
// ----------------------------------------------------------------------------

const C = {
  navy: "#0a1628", card: "#101c2e", line: "rgba(255,255,255,0.09)",
  gold: "#c9a84c", text: "#e8ecf2", muted: "#8b9ab0", red: "#e05a5a",
};
const TABS = {
  Chat: ["Ask insurance questions in plain language.", "Answers come from insurance textbook content, so staff get a starting point fast.", "Try: How do I file a motor claim?"],
  "Fraud Tool": ["Enter a claim's details and get a risk score with the reasons behind it.", "Flagged claims go to a staff reviewer. The model never decides alone.", "Try: Enter the demo, then open Fraud Tool."],
  Dashboard: ["See completed assessments, the risk mix and flag rates by policy type.", "Open any past assessment to see its form, result and chat again.", "Try: Enter the demo, then open Dashboard."],
};
const STEPS = [
  ["Ask", "Staff type a question in plain language, or fill in a claim in the Fraud Tool."],
  ["Research", "The agent looks up the insurance textbook passages that fit the question."],
  ["Run tools", "When a claim is provided, it runs the fraud model on the claim details."],
  ["Summarise", "It writes the answer, or the score with the reasons behind it."],
  ["Staff decide", "Flagged claims go to a reviewer. The model never decides alone."],
];

function Pipeline() {
  const [active, setActive] = useState(0);
  const [hover, setHover] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [still, setStill] = useState(false);
  useEffect(() => { setStill(window.matchMedia("(prefers-reduced-motion: reduce)").matches); }, []);
  useEffect(() => {
    if (hover || pinned || still) return;
    const t = setInterval(() => setActive((a) => (a + 1) % STEPS.length), 2800);
    return () => clearInterval(t);
  }, [hover, pinned, still]);

  return (
    <div className="rounded-2xl p-6" onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      style={{ background: C.card, border: `1px solid ${C.line}`, boxShadow: "0 24px 60px rgba(0,0,0,.4), 0 0 0 6px rgba(201,168,76,.05)" }}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="serif text-xl font-semibold">How a request moves through InsurAI</h2>
          <p className="mt-1 text-xs" style={{ color: C.muted }}>Click a step to read it.</p>
        </div>
        {pinned && !still && (
          <button onClick={() => setPinned(false)} className="rounded-full px-3 py-1 text-xs font-medium focus:outline-none focus-visible:ring-2"
            style={{ border: `1px solid ${C.gold}`, color: C.gold, "--tw-ring-color": C.gold }}>Play</button>
        )}
      </div>

      <ol className="mt-6 list-none p-0">
        {STEPS.map(([title, text], i) => {
          const on = i === active;
          return (
            <li key={title} className="flex gap-4">
              <div className="flex flex-col items-center">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold"
                  style={{ background: on ? C.gold : "transparent", color: on ? C.navy : i < active ? C.gold : C.muted,
                    border: `1.5px solid ${on || i < active ? C.gold : C.line}`,
                    boxShadow: on ? "0 0 0 6px rgba(201,168,76,.15)" : "none", transition: "all .4s" }}>{i + 1}</span>
                {i < STEPS.length - 1 && <span className="my-1 w-px flex-1" style={{ background: i < active ? C.gold : C.line, transition: "background .4s" }} />}
              </div>
              <button onClick={() => { setActive(i); setPinned(true); }} aria-current={on ? "step" : undefined}
                className="min-h-[44px] flex-1 rounded-lg pb-4 text-left focus:outline-none focus-visible:ring-2" style={{ "--tw-ring-color": C.gold }}>
                <span className="block text-base font-semibold" style={{ color: on ? C.text : C.muted, transition: "color .4s" }}>{title}</span>
                <span className="block overflow-hidden text-sm leading-relaxed" style={{ color: C.muted, maxHeight: on ? 60 : 0, opacity: on ? 1 : 0, transition: "max-height .4s, opacity .4s" }}>{text}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export default function DemoPage() {
  const navigate = useNavigate();
  const [status, setStatus] = useState("idle");
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("Chat");

  useEffect(() => { fetch(API_URL).catch(() => {}); }, []);
  useEffect(() => {
    if (status !== "loading") { setSlow(false); return; }
    const t = setTimeout(() => setSlow(true), 4000);
    return () => clearTimeout(t);
  }, [status]);

  async function enterDemo() {
    setStatus("loading"); setError("");
    sessionStorage.removeItem(TOKEN_KEY);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90000);
    try {
      const res = await fetch(`${API_URL}${LOGIN_PATH}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: DEMO_USER, password: DEMO_PASS }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join(", ") : body.detail;
        throw new Error(detail || `The server returned an error (${res.status}). Try again in a moment.`);
      }
      const data = await res.json();
      sessionStorage.setItem(TOKEN_KEY, data.access_token);
      sessionStorage.setItem("insurai_user", data.username);
      navigate(APP_ROUTE, { replace: true });
    } catch (e) {
      setStatus("error");
      setError(e.name === "AbortError" ? "The server took too long to wake up. Try again." : e.message || "Could not reach the server. Try again.");
    } finally { clearTimeout(timeout); }
  }

  const loading = status === "loading";

  return (
    <div className="min-h-screen" style={{
      background: `radial-gradient(900px 480px at 85% -5%, rgba(201,168,76,.13), transparent 60%), ${C.navy}`,
      color: C.text, fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Playfair+Display:wght@600;700&display=swap');
        .serif{font-family:'Playfair Display',serif}`}</style>

      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl" style={{ background: C.card, border: `1px solid ${C.line}` }}>
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden="true">
              <path d="M12 2L4 5.5V11c0 5 3.5 9.5 8 11 4.5-1.5 8-6 8-11V5.5L12 2z" fill={C.gold} />
              <text x="12" y="15.5" textAnchor="middle" fontSize="7" fontWeight="700" fill={C.navy}>AI</text>
            </svg>
          </div>
          <span className="serif text-xl font-semibold">InsurAI</span>
        </div>
        <Link to="/login" className="rounded text-sm font-medium underline underline-offset-4 focus:outline-none focus-visible:ring-2" style={{ color: C.gold }}>
          Sign in
        </Link>
      </header>

      <main className="mx-auto max-w-6xl px-6 pb-20 pt-6">
        <div className="grid gap-12 lg:grid-cols-2 lg:items-center">
          <section>
            <h1 className="serif max-w-[22ch] text-4xl font-bold leading-[1.12] sm:text-[2.75rem]">
              Review claims faster. Flag the suspicious ones. Investigate faster.
            </h1>
            <p className="mt-6 max-w-[50ch] text-lg leading-relaxed" style={{ color: C.muted }}>
              InsurAI is an AI copilot for claims and underwriting teams in Nigerian insurance. Ask it insurance questions, and run motor claims through a fraud check that hands flagged cases to a human reviewer.
            </p>

            <div className="mt-8">
              <button onClick={enterDemo} disabled={loading}
                className="inline-flex items-center gap-3 rounded-lg px-7 py-3.5 text-base font-semibold focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-wait disabled:opacity-80"
                style={{ background: C.gold, color: C.navy, "--tw-ring-color": C.gold, "--tw-ring-offset-color": C.navy }}>
                {loading && (
                  <svg className="h-4 w-4 motion-safe:animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity=".3" />
                    <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                )}
                {loading ? "Opening the demo" : "Enter the demo"}
              </button>
              <div className="mt-3 min-h-[3rem] text-sm" role="status" aria-live="polite">
                {loading && slow && <p style={{ color: C.muted }}>The server is waking up. This can take up to a minute on the first visit. Please don't refresh.</p>}
                {status === "error" && <p style={{ color: C.red }}>{error}</p>}
              </div>
              <p className="text-sm" style={{ color: C.muted }}>
                Already have an account?{" "}
                <Link to="/login" className="font-medium underline underline-offset-4" style={{ color: C.gold }}>Sign in</Link>
              </p>
            </div>

            <div className="mt-8 max-w-[50ch] border-l-2 pl-5" style={{ borderColor: C.line }}>
              <h2 className="text-sm font-semibold">Before you start</h2>
              <ul className="mt-2 space-y-1.5 text-sm" style={{ color: C.muted }}>
                <li>This is a working prototype, not a production system.</li>
                <li>All claims and policy data in the demo are synthetic.</li>
                <li>Fraud scores prompt a review. A person makes the decision.</li>
              </ul>
            </div>
          </section>

          <Pipeline />
        </div>

        <section className="mt-16" aria-label="What is inside">
          <h2 className="serif text-2xl font-semibold">What's inside the demo</h2>
          <div className="mt-4 flex gap-2" role="tablist">
            {Object.keys(TABS).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
                className="rounded-lg px-4 py-2 text-sm font-medium focus:outline-none focus-visible:ring-2"
                style={{ background: tab === t ? C.card : "transparent", color: tab === t ? C.gold : C.muted,
                  border: `1px solid ${tab === t ? C.gold : C.line}`, "--tw-ring-color": C.gold }}>{t}</button>
            ))}
          </div>
          <div role="tabpanel" className="mt-4 rounded-2xl p-6" style={{ background: C.card, border: `1px solid ${C.line}` }}>
            <p className="text-lg">{TABS[tab][0]}</p>
            <p className="mt-2 max-w-[60ch]" style={{ color: C.muted }}>{TABS[tab][1]}</p>
            <p className="mt-4 text-sm font-medium" style={{ color: C.gold }}>{TABS[tab][2]}</p>
          </div>
        </section>
      </main>
    </div>
  );
}