import { useState, useEffect } from 'react'

export default function PredictForm({ onPredict, loading, initialValues, onValuesChange }) {
  const [sentence, setSentence] = useState('')
  const [drug1,    setDrug1]    = useState('')
  const [drug2,    setDrug2]    = useState('')

  // Sync when a parent (e.g. example click) sets new values
  useEffect(() => {
    if (initialValues) {
      setSentence(initialValues.sentence || '')
      setDrug1(initialValues.drug1    || '')
      setDrug2(initialValues.drug2    || '')
    }
  }, [initialValues])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!sentence.trim()) return
    onPredict(sentence, drug1, drug2)
  }

  const handleClear = () => {
    setSentence('')
    setDrug1('')
    setDrug2('')
    if (onValuesChange) onValuesChange({ sentence: '', drug1: '', drug2: '' })
  }

  const canSubmit = sentence.trim().length > 0 && !loading

  return (
    <div className="card">
      <p className="card-title">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M11.5 2a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-1 0v-1a.5.5 0 0 1 .5-.5zm-7 0a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-1 0v-1A.5.5 0 0 1 4.5 2zM2 5.5A1.5 1.5 0 0 1 3.5 4h9A1.5 1.5 0 0 1 14 5.5v7a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 12.5v-7zm1.5-.5a.5.5 0 0 0-.5.5v7a.5.5 0 0 0 .5.5h9a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.5-.5h-9z"/>
        </svg>
        Analyse Interaction
      </p>

      <form onSubmit={handleSubmit}>
        {/* Sentence input */}
        <div className="form-group">
          <label className="form-label" htmlFor="sentence">
            Clinical sentence
          </label>
          <textarea
            id="sentence"
            className="form-textarea"
            placeholder="Enter a sentence that mentions two drugs, e.g. 'Warfarin toxicity is increased when taken with aspirin.'"
            value={sentence}
            onChange={e => setSentence(e.target.value)}
            rows={5}
          />
        </div>

        {/* Drug name inputs */}
        <div className="form-group">
          <label className="form-label">Drug entities</label>
          <div className="form-row">
            <div className="drug-input-wrapper">
              <span className="drug-badge e1">E1</span>
              <input
                type="text"
                className="form-input"
                placeholder="First drug"
                value={drug1}
                onChange={e => setDrug1(e.target.value)}
              />
            </div>
            <div className="drug-input-wrapper">
              <span className="drug-badge e2">E2</span>
              <input
                type="text"
                className="form-input"
                placeholder="Second drug"
                value={drug2}
                onChange={e => setDrug2(e.target.value)}
              />
            </div>
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 7 }}>
            Optional — if left blank, the model will use placeholder names. Adding real drug names improves entity marking accuracy.
          </p>
        </div>

        {/* Actions */}
        <div className="form-actions">
          <button type="submit" className="btn-primary" disabled={!canSubmit}>
            {loading
              ? <><span className="spinner" /> Analysing…</>
              : <>
                  <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M9.405 1.05c-.413-1.4-2.397-1.4-2.81 0l-.1.34a1.464 1.464 0 0 1-2.105.872l-.31-.17c-1.283-.698-2.686.705-1.987 1.987l.169.311c.446.82.023 1.841-.872 2.105l-.34.1c-1.4.413-1.4 2.397 0 2.81l.34.1a1.464 1.464 0 0 1 .872 2.105l-.17.31c-.698 1.283.705 2.686 1.987 1.987l.311-.169a1.464 1.464 0 0 1 2.105.872l.1.34c.413 1.4 2.397 1.4 2.81 0l.1-.34a1.464 1.464 0 0 1 2.105-.872l.31.17c1.283.698 2.686-.705 1.987-1.987l-.169-.311a1.464 1.464 0 0 1 .872-2.105l.34-.1c1.4-.413 1.4-2.397 0-2.81l-.34-.1a1.464 1.464 0 0 1-.872-2.105l.17-.31c.698-1.283-.705-2.686-1.987-1.987l-.311.169a1.464 1.464 0 0 1-2.105-.872l-.1-.34zM8 10.93a2.929 2.929 0 1 1 0-5.86 2.929 2.929 0 0 1 0 5.858z"/>
                  </svg>
                  Analyse Interaction
                </>
            }
          </button>
          <button type="button" className="btn-secondary" onClick={handleClear} disabled={loading}>
            Clear
          </button>
        </div>
      </form>
    </div>
  )
}
