import { useState, useCallback } from 'react'
import Header       from './components/Header'
import PredictForm  from './components/PredictForm'
import ResultPanel  from './components/ResultPanel'
import ExampleCards from './components/ExampleCards'

// ── Example sentences (one per interaction class) ──────────────────────────
const EXAMPLES = [
  {
    sentence: 'Warfarin toxicity is significantly increased when co-administered with aspirin, due to anticoagulant potentiation.',
    drug1:    'warfarin',
    drug2:    'aspirin',
    label:    'effect',
  },
  {
    sentence: 'Rifampin induces hepatic CYP3A4 enzymes and thereby markedly decreases the plasma concentration of clarithromycin.',
    drug1:    'rifampin',
    drug2:    'clarithromycin',
    label:    'mechanism',
  },
  {
    sentence: 'Concomitant use of methotrexate and NSAIDs is not recommended due to potential renal toxicity and increased methotrexate levels.',
    drug1:    'methotrexate',
    drug2:    'NSAIDs',
    label:    'advise',
  },
  {
    sentence: 'There is an interaction between fluoxetine and tramadol that may increase the risk of serotonin syndrome.',
    drug1:    'fluoxetine',
    drug2:    'tramadol',
    label:    'int',
  },
  {
    sentence: 'Paracetamol and ibuprofen can generally be alternated safely when used at recommended doses for short-term pain management.',
    drug1:    'paracetamol',
    drug2:    'ibuprofen',
    label:    'negative',
  },
]

export default function App() {
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const [form,    setForm]    = useState({ sentence: '', drug1: '', drug2: '' })

  const handlePredict = useCallback(async (sentence, drug1, drug2) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch('/api/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ sentence, drug1, drug2 }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.error || `Server returned HTTP ${res.status}`)
      }

      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleExampleSelect = useCallback((ex) => {
    setForm({ sentence: ex.sentence, drug1: ex.drug1, drug2: ex.drug2 })
    setResult(null)
    setError(null)
  }, [])

  return (
    <div className="app-shell">
      <Header />

      <main className="app-main">
        <div className="app-grid">

          {/* ── Left column: form + examples ── */}
          <section className="col-left">
            <PredictForm
              initialValues={form}
              loading={loading}
              onPredict={handlePredict}
              onValuesChange={setForm}
            />
            <ExampleCards examples={EXAMPLES} onSelect={handleExampleSelect} />
          </section>

          {/* ── Right column: results ── */}
          <section className="col-right">
            <ResultPanel result={result} loading={loading} error={error} />
          </section>

        </div>
      </main>

      <footer className="app-footer">
        <span>DDI Extraction System</span>
        <span className="sep">·</span>
        <span>BioBERT (dmis-lab/biobert-v1.1)</span>
        <span className="sep">·</span>
        <span>SemEval-2013 Task 9 · DDI 2013 Corpus</span>
        <span className="sep">·</span>
        <span>Macro-F1 0.63 (partial fine-tune)</span>
      </footer>
    </div>
  )
}
