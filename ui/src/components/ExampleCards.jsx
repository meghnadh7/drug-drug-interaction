const COLOR_MAP = {
  negative:  'var(--col-negative)',
  mechanism: 'var(--col-mechanism)',
  effect:    'var(--col-effect)',
  advise:    'var(--col-advise)',
  int:       'var(--col-int)',
}

const TYPE_LABELS = {
  negative:  'No Interaction',
  mechanism: 'Pharmacokinetic',
  effect:    'Pharmacodynamic',
  advise:    'Clinical Advisory',
  int:       'General Interaction',
}

export default function ExampleCards({ examples, onSelect }) {
  return (
    <div className="card">
      <p className="card-title">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M5 10.5a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 0 1h-2a.5.5 0 0 1-.5-.5zm0-2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0-2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5zm0-2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5z"/>
          <path d="M3 0h10a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2zm0 1a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H3z"/>
        </svg>
        Example sentences
      </p>

      <div className="examples-grid">
        {examples.map((ex, i) => (
          <button
            key={i}
            className="example-card"
            onClick={() => onSelect(ex)}
            title="Click to load this example"
          >
            <span
              className="example-type-dot"
              style={{ background: COLOR_MAP[ex.label] }}
            />
            <div className="example-content">
              <p
                className="example-type-label"
                style={{ color: COLOR_MAP[ex.label] }}
              >
                {TYPE_LABELS[ex.label]}
              </p>
              <p className="example-sentence">{ex.sentence}</p>
              <p className="example-drugs">
                {ex.drug1} · {ex.drug2}
              </p>
            </div>
            <svg
              width="12"
              height="12"
              viewBox="0 0 16 16"
              fill="var(--text-dim)"
              style={{ flexShrink: 0, marginTop: 4 }}
            >
              <path d="M4 8a.5.5 0 0 1 .5-.5h5.793L8.146 5.354a.5.5 0 1 1 .708-.708l3 3a.5.5 0 0 1 0 .708l-3 3a.5.5 0 0 1-.708-.708L10.293 8.5H4.5A.5.5 0 0 1 4 8z"/>
            </svg>
          </button>
        ))}
      </div>
    </div>
  )
}
