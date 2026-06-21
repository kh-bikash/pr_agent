import { useState, useEffect, useCallback, useRef } from 'react'

// ─── Constants ───────────────────────────────────────────────────────────────

const API_BASE = ''   // proxied via vite.config.js

const AGENT_META = {
  security:     { icon: '🔒', label: 'Security Agent',      cls: 'security' },
  performance:  { icon: '⚡', label: 'Performance Agent',   cls: 'performance' },
  code_quality: { icon: '✨', label: 'Code Quality Agent',  cls: 'code_quality' },
}

const SEV_ORDER = { critical: 0, warning: 1, info: 2 }

const LOADING_STEPS = [
  { key: 'fetch',    icon: '🔗', label: 'Fetching PR diff from GitHub…' },
  { key: 'security', icon: '🔒', label: 'Running Security Agent…' },
  { key: 'perf',     icon: '⚡', label: 'Running Performance Agent…' },
  { key: 'quality',  icon: '✨', label: 'Running Code Quality Agent…' },
  { key: 'post',     icon: '💬', label: 'Posting review comment to PR…' },
]

// ─── Small reusable components ────────────────────────────────────────────────

function SeverityBadge({ severity }) {
  return <span className={`sev-badge ${severity}`}>{severity}</span>
}

function IssueCard({ issue }) {
  return (
    <div className={`issue-item ${issue.severity}`}>
      <SeverityBadge severity={issue.severity} />
      <div className="issue-meta">
        {issue.file && (
          <span className="meta-chip">
            📄 {issue.file}
          </span>
        )}
        {issue.line_number && (
          <span className="meta-chip">
            L{issue.line_number}
          </span>
        )}
      </div>
      <p className="issue-desc">{issue.description}</p>
    </div>
  )
}

function AgentCard({ agentKey, data, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen ?? false)
  const meta = AGENT_META[agentKey] ?? { icon: '🔍', label: data.agent_name, cls: 'code_quality' }
  const sortedIssues = [...(data.issues ?? [])].sort(
    (a, b) => (SEV_ORDER[a.severity] ?? 3) - (SEV_ORDER[b.severity] ?? 3)
  )
  const hasIssues = sortedIssues.length > 0

  return (
    <div className={`agent-card ${open ? 'open' : ''}`}>
      {/* Header */}
      <div className="agent-header" onClick={() => setOpen(o => !o)}>
        <div className="agent-title-group">
          <div className={`agent-icon-wrap ${meta.cls}`}>{meta.icon}</div>
          <div>
            <div className="agent-name">{data.agent_name}</div>
            <div className="agent-summary-short">{data.summary}</div>
          </div>
        </div>
        <div className="agent-header-right">
          <span className={`issue-count-badge ${hasIssues ? 'has-issues' : 'no-issues'}`}>
            {hasIssues ? `${sortedIssues.length} issue${sortedIssues.length !== 1 ? 's' : ''}` : 'Clean ✓'}
          </span>
          <span className={`chevron ${open ? 'open' : ''}`}>▾</span>
        </div>
      </div>

      {/* Body */}
      {open && (
        <div className="agent-body">
          <div className="summary-box">{data.summary}</div>
          {hasIssues ? (
            <div className="issues-list">
              {sortedIssues.map((issue, i) => (
                <IssueCard key={i} issue={issue} />
              ))}
            </div>
          ) : (
            <div className="no-issues-msg">
              ✅ No issues detected — this section looks good!
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Fallback for older browsers
      const el = document.createElement('textarea')
      el.value = text
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }, [text])

  return (
    <button className={`copy-btn ${copied ? 'copied' : ''}`} onClick={handleCopy}>
      {copied ? '✓ Copied!' : '📋 Copy Markdown'}
    </button>
  )
}

function LoadingCard({ elapsed }) {
  const [activeStep, setActiveStep] = useState(0)

  useEffect(() => {
    // Progress through steps based on elapsed time (approximate)
    const timings = [0, 2, 4, 8, 18]  // seconds when each step becomes active
    let step = 0
    for (let i = 0; i < timings.length; i++) {
      if (elapsed >= timings[i]) step = i
    }
    setActiveStep(step)
  }, [elapsed])

  return (
    <div className="loading-card">
      <div className="spinner-ring" />
      <div className="loading-title">Analysing your Pull Request…</div>
      <div className="loading-steps">
        {LOADING_STEPS.map((step, i) => (
          <div
            key={step.key}
            className={`step-item ${i === activeStep ? 'active' : i < activeStep ? 'done' : ''}`}
          >
            <span className="step-icon">
              {i < activeStep ? '✓' : step.icon}
            </span>
            {step.label}
          </div>
        ))}
      </div>
      <span className="elapsed">⏱ {elapsed}s elapsed</span>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [prUrl, setPrUrl] = useState('')
  const [postComment, setPostComment] = useState(true)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const timerRef = useRef(null)

  const startTimer = useCallback(() => {
    setElapsed(0)
    timerRef.current = setInterval(() => {
      setElapsed(s => s + 1)
    }, 1000)
  }, [])

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  // clean up on unmount
  useEffect(() => () => stopTimer(), [stopTimer])

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault()
    if (!prUrl.trim()) return

    setLoading(true)
    setResult(null)
    setError(null)
    startTimer()

    try {
      const resp = await fetch(`${API_BASE}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pr_url: prUrl.trim(), post_comment: postComment }),
      })

      const data = await resp.json()

      if (!resp.ok) {
        throw new Error(data.detail ?? `Server error ${resp.status}`)
      }

      setResult(data)
    } catch (err) {
      setError(err.message ?? 'An unexpected error occurred.')
    } finally {
      stopTimer()
      setLoading(false)
    }
  }, [prUrl, postComment, startTimer, stopTimer])

  // Derived stats
  const stats = result ? (() => {
    const all = [
      ...(result.security?.issues ?? []),
      ...(result.performance?.issues ?? []),
      ...(result.code_quality?.issues ?? []),
    ]
    return {
      total: all.length,
      critical: all.filter(i => i.severity === 'critical').length,
      warning:  all.filter(i => i.severity === 'warning').length,
      info:     all.filter(i => i.severity === 'info').length,
    }
  })() : null

  return (
    <>
      <div className="bg-gradient" />
      <div className="app-wrapper">

        {/* ── Header ─────────────────────────────────────── */}
        <header className="header">
          <div className="header-badge">
            <span className="dot" />
            Powered by LangGraph + Dough API
          </div>
          <h1>AI Code Review Agent</h1>
          <p>
            Paste a GitHub PR URL and three AI agents will analyse it for
            security vulnerabilities, performance issues, and code quality — in parallel.
          </p>
        </header>

        {/* ── Input card ─────────────────────────────────── */}
        <section className="input-card" aria-label="Review form">
          <form onSubmit={handleSubmit}>
            <label className="input-label" htmlFor="pr-url-input">
              GitHub Pull Request URL
            </label>
            <div className="input-row">
              <input
                id="pr-url-input"
                className="pr-input"
                type="url"
                placeholder="https://github.com/owner/repo/pull/123"
                value={prUrl}
                onChange={e => setPrUrl(e.target.value)}
                disabled={loading}
                required
                aria-describedby="pr-url-hint"
              />
              <button
                type="submit"
                className="submit-btn"
                disabled={loading || !prUrl.trim()}
                id="submit-review-btn"
              >
                {loading ? (
                  <>
                    <span style={{ display:'inline-block', animation:'spin 1s linear infinite', fontSize:'14px' }}>⟳</span>
                    Analysing…
                  </>
                ) : (
                  <>🔍 Review PR</>
                )}
              </button>
            </div>
            <p id="pr-url-hint" className="input-hint">
              <span>Public repos work without a GitHub token</span>
              <span>Token required to post comments</span>
              <span>Uses groq/llama-3.3-70b-versatile via Dough API</span>
            </p>

            {/* Toggle */}
            <div className="toggle-row">
              <label className="toggle" htmlFor="post-comment-toggle">
                <input
                  id="post-comment-toggle"
                  type="checkbox"
                  checked={postComment}
                  onChange={e => setPostComment(e.target.checked)}
                  disabled={loading}
                />
                <span className="toggle-track" />
              </label>
              <span className="toggle-label" onClick={() => !loading && setPostComment(v => !v)}>
                Post review comment to GitHub PR
              </span>
            </div>
          </form>
        </section>

        {/* ── Loading ─────────────────────────────────────── */}
        {loading && <LoadingCard elapsed={elapsed} />}

        {/* ── Error ───────────────────────────────────────── */}
        {error && !loading && (
          <div className="error-banner" role="alert">
            <span className="error-icon">⚠️</span>
            <div>
              <strong>Review failed:</strong> {error}
            </div>
          </div>
        )}

        {/* ── Results ─────────────────────────────────────── */}
        {result && !loading && (
          <div className="results-wrapper">

            {/* Summary bar */}
            <div className="summary-bar">
              <div className="summary-pr-info">
                <h2>{result.pr_title || 'Pull Request Review'}</h2>
                <p>
                  by <strong>@{result.pr_author}</strong>
                  {' · '}
                  <span className="duration-tag">⏱ {result.duration_seconds}s</span>
                </p>
              </div>
              <div className="summary-stats">
                <span className="stat-pill total">📊 {stats.total} total</span>
                {stats.critical > 0 && <span className="stat-pill critical">🔴 {stats.critical} critical</span>}
                {stats.warning  > 0 && <span className="stat-pill warning">🟡 {stats.warning} warnings</span>}
                {stats.info     > 0 && <span className="stat-pill info">🔵 {stats.info} info</span>}
              </div>
            </div>

            {/* Comment notice */}
            {result.comment_posted ? (
              <div className="comment-notice posted">
                💬 Review comment posted to GitHub PR
                {result.comment_url && (
                  <>
                    {' → '}
                    <a href={result.comment_url} target="_blank" rel="noopener noreferrer"
                       style={{ color: 'inherit', textDecoration: 'underline' }}>
                      View on GitHub ↗
                    </a>
                  </>
                )}
              </div>
            ) : (
              <div className="comment-notice skipped">
                ℹ️ Comment not posted (toggle off or GITHUB_TOKEN not set)
              </div>
            )}

            {/* Agent accordions — open first one with issues by default */}
            {[
              { key: 'security',     data: result.security },
              { key: 'performance',  data: result.performance },
              { key: 'code_quality', data: result.code_quality },
            ].map(({ key, data }, idx) => (
              data && (
                <AgentCard
                  key={key}
                  agentKey={key}
                  data={data}
                  defaultOpen={idx === 0 || (data.issues?.length ?? 0) > 0}
                />
              )
            ))}

            {/* Raw markdown panel */}
            <div className="markdown-panel">
              <div className="markdown-header">
                <h3>📝 Generated GitHub Comment (Markdown)</h3>
                <CopyButton text={result.markdown_comment} />
              </div>
              <pre className="markdown-code">{result.markdown_comment}</pre>
            </div>

          </div>
        )}

      </div>
    </>
  )
}
