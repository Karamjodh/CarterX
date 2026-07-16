import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadFile } from '../services/api'
import { useJobPolling } from '../hooks/useJobPolling'

export default function UploadPage() {
  const [jobId,       setJobId]       = useState(null)
  const [uploading,   setUploading]   = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [showDisclaimer, setShowDisclaimer] = useState(false)
  const navigate     = useNavigate()
  const hasNavigated = useRef(false)

  const { job, error: pollError } = useJobPolling(jobId)

  useEffect(() => {
    if (job?.status === 'completed' && !hasNavigated.current) {
      hasNavigated.current = true
      navigate(`/dashboard/${jobId}`, { replace: true })
    }
  }, [job?.status, jobId, navigate])

  const handleFileChange = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    hasNavigated.current = false
    setUploading(true)
    setUploadError(null)
    try {
      const res = await uploadFile(file)
      setJobId(res.data.id)
    } catch (err) {
      setUploadError(err.response?.data?.detail || 'Upload failed. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  const stageIcon  = s => s === 'completed' ? '✓' : s === 'running' ? '⟳' : '○'
  const stageColor = s => s === 'completed' ? '#0F6E56' : s === 'running' ? '#534AB7' : '#B4B2A9'

  return (
    <div style={S.page}>
      <div style={S.container}>

        <div style={S.hero}>
          <h1 style={S.title}>CarterX</h1>
          <p style={S.tagline}>AI-powered customer analytics</p>
          <p style={S.desc}>
            Upload your sales data. Get customer segments, purchase patterns,
            revenue trends, and AI-generated marketing strategy — automatically.
          </p>
        </div>

        {!jobId ? (
          <>
            <div style={S.uploadZone}>
              <div style={S.uploadIcon}>↑</div>
              <p style={S.uploadTitle}>Drop your sales data here</p>
              <p style={S.uploadSub}>CSV or XLSX · max 50MB · min 100 rows</p>
              <label style={S.uploadBtn}>
                Choose file
                <input type="file" accept=".csv,.xlsx" onChange={handleFileChange}
                  disabled={uploading} style={{ display: 'none' }} />
              </label>
              {uploading && <p style={S.info}>Uploading...</p>}
              {uploadError && <p style={S.error}>{uploadError}</p>}

              {/* Data requirements toggle */}
              <button
                onClick={() => setShowDisclaimer(v => !v)}
                style={S.disclaimerToggle}
              >
                {showDisclaimer ? '▲' : '▼'} Data requirements & limitations
              </button>
            </div>

            {/* Disclaimer panel */}
            {showDisclaimer && (
              <div style={S.disclaimerBox}>

                <div style={S.disclaimerHeader}>
                  <span style={S.disclaimerIcon}>⚠</span>
                  <div>
                    <div style={S.disclaimerTitle}>Data Quality & Pipeline Limitations</div>
                    <div style={S.disclaimerSub}>
                      CarterX pipelines are optimised for Amazon-style datasets.
                      Results may vary with other data formats.
                    </div>
                  </div>
                </div>

                <div style={S.sectionGrid}>

                  {/* Required columns */}
                  <div style={S.disclaimerSection}>
                    <div style={S.sectionTitle}>✅ Required column</div>
                    <div style={S.sectionDesc}>
                      Your file must contain at least one customer identifier column.
                      The pipeline accepts any of these names:
                    </div>
                    <div style={S.tagRow}>
                      {['customer_id','user_id','buyer_id','customerid','cust_id','customer_name'].map(t => (
                        <span key={t} style={S.tagGreen}>{t}</span>
                      ))}
                    </div>
                  </div>

                  {/* Recommended columns */}
                  <div style={S.disclaimerSection}>
                    <div style={S.sectionTitle}>📋 Recommended columns</div>
                    <div style={S.sectionDesc}>
                      These unlock the full pipeline — segmentation, forecasting,
                      association rules, and trend analysis:
                    </div>
                    <table style={S.colTable}>
                      <tbody>
                        {[
                          ['date / order_date',       'Enables LSTM forecasting & trend analysis'],
                          ['price / total_sales',     'Revenue-based RFM segmentation'],
                          ['quantity / qty',          'Accurate monetary calculation'],
                          ['product_id / product',    'Association rule mining'],
                          ['category',                'Category-level rules & trends'],
                          ['transaction_id / order_id','Frequency-based RFM'],
                        ].map(([col, desc]) => (
                          <tr key={col}>
                            <td style={S.colName}>{col}</td>
                            <td style={S.colDesc}>{desc}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                </div>

                {/* Known limitations */}
                <div style={S.disclaimerSection}>
                  <div style={S.sectionTitle}>⚡ Known limitations</div>
                  <div style={S.limitGrid}>
                    {[
                      ['Minimum rows',       'At least 100 rows required. Under 500 rows may produce weak segmentation.'],
                      ['Customer names as ID','Using names (e.g. "Emma Clark") as customer IDs causes RFM errors — multiple people share names.'],
                      ['No date column',     'Forecasting tab will show an empty state. Simulation falls back to average daily rate.'],
                      ['Single-purchase data','Association rules will fall back to popular items mode — co-purchase rules need multi-item baskets.'],
                      ['Revenue double-count','If your file has both Price and Total Sales, map only one to avoid double-counting revenue.'],
                      ['Date formats',       'Dates like "14-03-25" may not parse. Use YYYY-MM-DD or DD/MM/YYYY for best results.'],
                    ].map(([title, desc]) => (
                      <div key={title} style={S.limitCard}>
                        <div style={S.limitTitle}>{title}</div>
                        <div style={S.limitDesc}>{desc}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Best dataset */}
                <div style={S.bestDataset}>
                  <span style={S.bestIcon}>🏆</span>
                  <div>
                    <strong>Best results with:</strong> Amazon product review datasets (e.g. amazon.csv with
                    user_id, product_id, rating, discounted_price, rating_count, category).
                    These are the format CarterX was designed and tested against.
                  </div>
                </div>

              </div>
            )}
          </>
        ) : (
          <div style={S.progress}>
            <h2 style={S.progressTitle}>
              {job?.status === 'failed' ? 'Pipeline failed' : 'Running analysis'}
            </h2>
            <p style={S.progressSub}>
              {job?.status === 'failed' ? 'An error occurred.' : 'This takes 10–30 seconds...'}
            </p>
            <div style={S.stages}>
              {Object.entries(job?.stage_status || {}).map(([stage, status]) => (
                <div key={stage} style={S.stageRow}>
                  <span style={{ ...S.stageIcon, color: stageColor(status) }}>
                    {stageIcon(status)}
                  </span>
                  <span style={S.stageName}>{stage.replace(/_/g, ' ')}</span>
                  <span style={{ ...S.stageStatus, color: stageColor(status) }}>
                    {status}
                  </span>
                </div>
              ))}
            </div>
            {job?.status === 'failed' && (
              <p style={S.error}>{job.error_message}</p>
            )}
          </div>
        )}

        <div style={S.features}>
          {[
            ['◉', 'Customer segments',  'GMM clustering on RFM features'],
            ['⇌', 'Purchase patterns',  'FP-Growth association rules'],
            ['▦', 'Revenue forecast',   'LSTM-powered predictions'],
            ['✦', 'AI strategy report', 'LLM-generated recommendations'],
          ].map(([icon, title, sub]) => (
            <div key={title} style={S.feature}>
              <span style={S.featureIcon}>{icon}</span>
              <span style={S.featureTitle}>{title}</span>
              <span style={S.featureSub}>{sub}</span>
            </div>
          ))}
        </div>

      </div>
    </div>
  )
}

const S = {
  // Layout
  page:          { minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem', background: '#F7F6F3' },
  container:     { width: '100%', maxWidth: 580 },
  hero:          { textAlign: 'center', marginBottom: '2rem' },
  title:         { fontSize: 40, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 4 },
  tagline:       { fontSize: 13, color: '#888780', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 12 },
  desc:          { fontSize: 14, color: '#5F5E5A', lineHeight: 1.7 },

  // Upload zone
  uploadZone:    { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 16, padding: '3rem 2rem', textAlign: 'center', marginBottom: 8 },
  uploadIcon:    { fontSize: 32, color: '#B4B2A9', marginBottom: 12 },
  uploadTitle:   { fontSize: 16, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 6 },
  uploadSub:     { fontSize: 12, color: '#888780', marginBottom: 20 },
  uploadBtn:     { display: 'inline-block', padding: '8px 20px', background: '#EEEDFE', color: '#534AB7', borderRadius: 8, fontSize: 13, fontWeight: 500, cursor: 'pointer' },
  info:          { fontSize: 13, color: '#534AB7', marginTop: 12 },
  error:         { fontSize: 13, color: '#993C1D', marginTop: 12 },
  disclaimerToggle: { display: 'inline-block', marginTop: 16, fontSize: 11, color: '#888780', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' },

  // Disclaimer box
  disclaimerBox: { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 14, padding: '1.25rem', marginBottom: 12 },
  disclaimerHeader: { display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 16, paddingBottom: 12, borderBottom: '0.5px solid #F1EFE8' },
  disclaimerIcon:   { fontSize: 18, flexShrink: 0, marginTop: 1 },
  disclaimerTitle:  { fontSize: 14, fontWeight: 600, color: '#1a1a1a', marginBottom: 2 },
  disclaimerSub:    { fontSize: 12, color: '#888780', lineHeight: 1.5 },

  // Sections
  sectionGrid:   { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 },
  disclaimerSection: { marginBottom: 16 },
  sectionTitle:  { fontSize: 12, fontWeight: 600, color: '#1a1a1a', marginBottom: 6 },
  sectionDesc:   { fontSize: 11, color: '#888780', marginBottom: 8, lineHeight: 1.5 },

  // Tags
  tagRow:        { display: 'flex', flexWrap: 'wrap', gap: 4 },
  tagGreen:      { background: '#E1F5EE', color: '#0F6E56', border: '0.5px solid #1D9E7540', padding: '2px 8px', borderRadius: 999, fontSize: 10, fontWeight: 500, fontFamily: 'monospace' },

  // Column table
  colTable:      { width: '100%', borderCollapse: 'collapse', fontSize: 11 },
  colName:       { padding: '4px 8px 4px 0', color: '#534AB7', fontFamily: 'monospace', fontWeight: 500, whiteSpace: 'nowrap', verticalAlign: 'top' },
  colDesc:       { padding: '4px 0', color: '#5F5E5A', lineHeight: 1.4 },

  // Limitations
  limitGrid:     { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 },
  limitCard:     { background: '#F7F6F3', borderRadius: 8, padding: '8px 10px' },
  limitTitle:    { fontSize: 11, fontWeight: 600, color: '#993C1D', marginBottom: 3 },
  limitDesc:     { fontSize: 11, color: '#5F5E5A', lineHeight: 1.4 },

  // Best dataset
  bestDataset:   { background: '#EEEDFE', border: '0.5px solid #AFA9EC', borderRadius: 10, padding: '10px 14px', display: 'flex', gap: 10, alignItems: 'flex-start', fontSize: 12, color: '#534AB7', lineHeight: 1.5 },
  bestIcon:      { fontSize: 14, flexShrink: 0, marginTop: 1 },

  // Progress
  progress:      { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 16, padding: '2rem', marginBottom: '1.5rem' },
  progressTitle: { fontSize: 20, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 4 },
  progressSub:   { fontSize: 13, color: '#888780', marginBottom: '1.5rem' },
  stages:        { display: 'flex', flexDirection: 'column', gap: 2 },
  stageRow:      { display: 'flex', alignItems: 'center', gap: 10, padding: '8px 0', borderBottom: '0.5px solid #F1EFE8' },
  stageIcon:     { fontSize: 14, width: 20, textAlign: 'center', flexShrink: 0 },
  stageName:     { flex: 1, fontSize: 13, color: '#5F5E5A', textTransform: 'capitalize' },
  stageStatus:   { fontSize: 12, fontWeight: 500 },

  // Features
  features:      { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 12 },
  feature:       { background: 'white', border: '0.5px solid #E8E6DF', borderRadius: 10, padding: '1rem', display: 'flex', flexDirection: 'column', gap: 3 },
  featureIcon:   { fontSize: 18, color: '#7F77DD' },
  featureTitle:  { fontSize: 13, fontWeight: 500, color: '#1a1a1a' },
  featureSub:    { fontSize: 12, color: '#888780' },
}