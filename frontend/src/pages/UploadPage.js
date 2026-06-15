import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadFile } from '../services/api'
import { useJobPolling } from '../hooks/useJobPolling'

export default function UploadPage() {
  const [jobId,       setJobId]       = useState(null)
  const [uploading,   setUploading]   = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const navigate = useNavigate()

  const { job, error: pollError } = useJobPolling(jobId)

  if (job?.status === 'completed') navigate(`/dashboard/${jobId}`)

  const handleFileChange = async (e) => {
    const file = e.target.files[0]
    if (!file) return
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

  const stageIcon = (status) =>
    status === 'completed' ? '✓' : status === 'running' ? '⟳' : '○'

  const stageColor = (status) =>
    status === 'completed' ? '#0F6E56' : status === 'running' ? '#534AB7' : '#B4B2A9'

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
          <div style={S.uploadZone}>
            <div style={S.uploadIcon}>↑</div>
            <p style={S.uploadTitle}>Drop your sales data here</p>
            <p style={S.uploadSub}>CSV or XLSX · max 50MB · min 100 rows</p>
            <label style={S.uploadBtn}>
              Choose file
              <input type="file" accept=".csv,.xlsx" onChange={handleFileChange}
                disabled={uploading} style={{display:'none'}} />
            </label>
            {uploading && <p style={S.info}>Uploading...</p>}
            {uploadError && <p style={S.error}>{uploadError}</p>}
          </div>
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
                  <span style={{...S.stageIcon, color: stageColor(status)}}>
                    {stageIcon(status)}
                  </span>
                  <span style={S.stageName}>{stage.replace(/_/g, ' ')}</span>
                  <span style={{...S.stageStatus, color: stageColor(status)}}>
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
            ['◉', 'Customer segments',   'KMeans clustering on RFM features'],
            ['⇌', 'Purchase patterns',   'FP-Growth association rules'],
            ['▦', 'Revenue trends',      'Monthly breakdown & forecasts'],
            ['✦', 'AI strategy report',  'LLM-generated recommendations'],
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
  page:          { minHeight:'100vh', display:'flex', alignItems:'center', justifyContent:'center', padding:'2rem', background:'#F7F6F3' },
  container:     { width:'100%', maxWidth:560 },
  hero:          { textAlign:'center', marginBottom:'2rem' },
  title:         { fontSize:40, fontFamily:"'Instrument Serif', serif", color:'#1a1a1a', marginBottom:4 },
  tagline:       { fontSize:13, color:'#888780', letterSpacing:'0.06em', textTransform:'uppercase', marginBottom:12 },
  desc:          { fontSize:14, color:'#5F5E5A', lineHeight:1.7 },
  uploadZone:    { background:'white', border:'0.5px solid #E8E6DF', borderRadius:16, padding:'3rem 2rem', textAlign:'center', marginBottom:'1.5rem' },
  uploadIcon:    { fontSize:32, color:'#B4B2A9', marginBottom:12 },
  uploadTitle:   { fontSize:16, fontFamily:"'Instrument Serif', serif", color:'#1a1a1a', marginBottom:6 },
  uploadSub:     { fontSize:12, color:'#888780', marginBottom:20 },
  uploadBtn:     { display:'inline-block', padding:'8px 20px', background:'#EEEDFE', color:'#534AB7', borderRadius:8, fontSize:13, fontWeight:500, cursor:'pointer' },
  info:          { fontSize:13, color:'#534AB7', marginTop:12 },
  error:         { fontSize:13, color:'#993C1D', marginTop:12 },
  progress:      { background:'white', border:'0.5px solid #E8E6DF', borderRadius:16, padding:'2rem', marginBottom:'1.5rem' },
  progressTitle: { fontSize:20, fontFamily:"'Instrument Serif', serif", color:'#1a1a1a', marginBottom:4 },
  progressSub:   { fontSize:13, color:'#888780', marginBottom:'1.5rem' },
  stages:        { display:'flex', flexDirection:'column', gap:2 },
  stageRow:      { display:'flex', alignItems:'center', gap:10, padding:'8px 0', borderBottom:'0.5px solid #F1EFE8' },
  stageIcon:     { fontSize:14, width:20, textAlign:'center', flexShrink:0 },
  stageName:     { flex:1, fontSize:13, color:'#5F5E5A', textTransform:'capitalize' },
  stageStatus:   { fontSize:12, fontWeight:500 },
  features:      { display:'grid', gridTemplateColumns:'1fr 1fr', gap:10 },
  feature:       { background:'white', border:'0.5px solid #E8E6DF', borderRadius:10, padding:'1rem', display:'flex', flexDirection:'column', gap:3 },
  featureIcon:   { fontSize:18, color:'#7F77DD' },
  featureTitle:  { fontSize:13, fontWeight:500, color:'#1a1a1a' },
  featureSub:    { fontSize:12, color:'#888780' },
}