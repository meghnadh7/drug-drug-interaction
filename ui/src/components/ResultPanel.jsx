import { useEffect, useRef } from 'react'

// Label display order (most clinically significant first)
const LABEL_ORDER = ['effect', 'mechanism', 'advise', 'int', 'negative']

const COLOR_MAP = {
  negative:  'var(--col-negative)',
  mechanism: 'var(--col-mechanism)',
  effect:    'var(--col-effect)',
  advise:    'var(--col-advise)',
  int:       'var(--col-int)',
}

// Highlight [E1] and [E2] markers in the marked sentence
function HighlightedSentence({ text }) {
  if (!text) return null

  const parts = []
  const regex = /(\[E1\]|\[\/E1\]|\[E2\]|\[\/E2\])/g
  let last = 0
  let inE1 = false, inE2 = false
  let key  = 0

  // A simpler pass: split by marker tags, render coloured spans
  const segments = text.split(/(\[E[12]\].*?\[\/E[12]\])/g)

  return (
    <span>
      {text.split(/(\[E1\].*?\[\/E1\]|\[E2\].*?\[\/E2\])/gs).map((part, i) => {
        if (/^\[E1\](.*?)\[\/E1\]$/s.test(part)) {
          const inner = part.replace(/\[E1\]|\[\/E1\]/g, '')
          return (
            <span key={i}>
              <span className="tok-e1">[E1]</span>
              {' '}{inner}{' '}
              <span className="tok-e1">[/E1]</span>
            </span>
          )
        }
        if (/^\[E2\](.*?)\[\/E2\]$/s.test(part)) {
          const inner = part.replace(/\[E2\]|\[\/E2\]/g, '')
          return (
            <span key={i}>
              <span className="tok-e2">[E2]</span>
              {' '}{inner}{' '}
              <span className="tok-e2">[/E2]</span>
            </span>
          )
        }
        return <span key={i}>{part}</span>
      })}
    </span>
  )
}

function ConfidenceBar({ label, pct, color, isTop }) {
  const fillRef = useRef(null)

  useEffect(() => {
    // Animate on mount
    if (fillRef.current) {
      fillRef.current.style.width = '0%'
      requestAnimationFrame(() => {
        setTimeout(() => {
          if (fillRef.current) fillRef.current.style.width = `${pct}%`
        }, 60)
      })
    }
  }, [pct])

  return (
    <div className="bar-row" style={{ opacity: pct < 0.5 ? 0.55 : 1 }}>
      <span className="bar-label" style={{ color: isTop ? color : undefined }}>
        {label}
      </span>
      <div className="bar-track">
        <div
          ref={fillRef}
          className="bar-fill"
          style={{
            width:      `${pct}%`,
            background: isTop
              ? `linear-gradient(90deg, ${color}aa, ${color})`
              : 'var(--border-2)',
          }}
        />
      </div>
      <span className="bar-pct" style={{ color: isTop ? color : undefined }}>
        {pct.toFixed(1)}%
      </span>
    </div>
  )
}

export default function ResultPanel({ result, loading, error }) {
  if (loading) {
    return (
      <div className="card">
        <div className="result-loading">
          <div className="big-spinner" />
          <p>Running BioBERT inference…</p>
          <p style={{ fontSize: 12, color: 'var(--text-dim)' }}>
            Typical latency: ~200 ms
          </p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card">
        <div className="result-error">
          <span style={{ fontSize: 18 }}>⚠</span>
          <div>
            <strong>Error</strong><br />
            {error}
            <br />
            <span style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4, display: 'block' }}>
              Make sure <code style={{ fontFamily: 'monospace' }}>python api.py</code> is running on port 5001.
            </span>
          </div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="card">
        <div className="result-empty">
          <div className="result-empty-icon">🔬</div>
          <p>Enter a clinical sentence and click<br /><strong style={{ color: 'var(--text)' }}>Analyse Interaction</strong> to see the prediction.</p>
          <p style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>
            Or click one of the example cards →
          </p>
        </div>
      </div>
    )
  }

  const { prediction, probabilities, meta, marked_sentence } = result
  const topMeta  = meta[prediction]
  const topColor = COLOR_MAP[prediction]

  return (
    <div className="card">
      <p className="card-title">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 13A6 6 0 1 1 8 2a6 6 0 0 1 0 12z"/>
          <path d="m10.97 4.97-.02.022-3.473 4.425-2.093-2.094a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-1.071-1.05z"/>
        </svg>
        Prediction
      </p>

      {/* Main prediction badge */}
      <div
        className="pred-badge"
        style={{
          background: topMeta.bg,
          color:      topColor,
          borderColor: `${topColor}55`,
        }}
      >
        <span style={{ fontSize: 18 }}>{topMeta.icon}</span>
        {topMeta.title}
        <span style={{ marginLeft: 'auto', fontSize: 12, opacity: 0.8 }}>
          {result.confidence}% confidence
        </span>
      </div>

      {/* Description */}
      <p
        className="pred-description"
        style={{ borderLeftColor: topColor }}
      >
        {topMeta.description}
      </p>

      {/* Confidence bars */}
      <div className="confidence-section">
        <p className="confidence-label">Class probabilities</p>
        {LABEL_ORDER.map(label => (
          <ConfidenceBar
            key={label}
            label={meta[label].title}
            pct={probabilities[label] ?? 0}
            color={COLOR_MAP[label]}
            isTop={label === prediction}
          />
        ))}
      </div>

      {/* Marked sentence */}
      <div className="marked-sentence-section">
        <p className="confidence-label" style={{ marginBottom: 8 }}>
          Entity-marked input
        </p>
        <div className="marked-sentence-box">
          <HighlightedSentence text={marked_sentence} />
        </div>
      </div>
    </div>
  )
}
