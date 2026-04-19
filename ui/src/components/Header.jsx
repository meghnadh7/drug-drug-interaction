export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">

        <div className="header-brand">
          <div className="header-icon">
            {/* Molecule / interaction icon */}
            <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="7"  cy="13" r="4" fill="#4493f8" opacity="0.9" />
              <circle cx="19" cy="13" r="4" fill="#a855f7" opacity="0.9" />
              <line x1="11" y1="13" x2="15" y2="13" stroke="#e6edf3" strokeWidth="1.5" strokeDasharray="1.5 1" />
              <circle cx="7"  cy="13" r="1.5" fill="white" opacity="0.6" />
              <circle cx="19" cy="13" r="1.5" fill="white" opacity="0.6" />
              <path d="M13 4 Q16 8 13 13 Q10 18 13 22" stroke="#4493f8" strokeWidth="0.9" fill="none" opacity="0.4" strokeDasharray="2 1.5" />
            </svg>
          </div>
          <div>
            <h1 className="header-title">Drug Interaction Extractor</h1>
            <p className="header-subtitle">
              BioBERT · SemEval-2013 Task 9 · 5-class DDI Classification
            </p>
          </div>
        </div>

        <div className="status-pill">
          <span className="status-dot" />
          Model ready
        </div>

      </div>
    </header>
  )
}
