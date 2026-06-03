import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadFile } from '../services/api'
import { useJobPolling } from '../hooks/useJobPolling'

export default function UploadPage() {
  const [jobId,    setJobId]    = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const navigate = useNavigate()

  // Start polling once we have a jobId
  const { job, error: pollError } = useJobPolling(jobId)

  // Navigate to dashboard when pipeline completes
  if (job?.status === 'completed') {
    navigate(`/dashboard/${jobId}`)
  }

const handleFileChange = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setUploading(true)
    setUploadError(null)

    try {
      console.log("Uploading file:", file.name)
      const response = await uploadFile(file)
      console.log("Upload response:", response)
      console.log("Job ID:", response.data.id)
      setJobId(response.data.id)
    } catch (err) {
      console.log("Upload error:", err)
      console.log("Error response:", err.response)
      console.log("Error message:", err.message)
      setUploadError(
        err.response?.data?.detail || 'Upload failed. Please try again.'
      )
    } finally {
      setUploading(false)
    }
  }

  const getStageIcon = (status) => {
    if (status === 'completed') return '✓'
    if (status === 'running')   return '⟳'
    return '○'
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>CarterX</h1>
      <p style={styles.subtitle}>Upload your sales data to get AI-powered insights</p>

      {/* Upload area */}
      {!jobId && (
        <div style={styles.uploadBox}>
          <p style={styles.uploadText}>Drop your CSV or XLSX file here</p>
          <input
            type="file"
            accept=".csv,.xlsx"
            onChange={handleFileChange}
            disabled={uploading}
            style={styles.fileInput}
          />
          {uploading && <p style={styles.info}>Uploading...</p>}
          {uploadError && <p style={styles.error}>{uploadError}</p>}
        </div>
      )}

      {/* Pipeline progress */}
      {job && (
        <div style={styles.progressBox}>
          <h2 style={styles.progressTitle}>
            {job.status === 'failed' ? 'Pipeline Failed' : 'Running Pipeline...'}
          </h2>

          {Object.entries(job.stage_status || {}).map(([stage, status]) => (
            <div key={stage} style={styles.stageRow}>
              <span style={{
                ...styles.stageIcon,
                color: status === 'completed' ? '#1D9E75'
                     : status === 'running'   ? '#7F77DD'
                     : '#888'
              }}>
                {getStageIcon(status)}
              </span>
              <span style={styles.stageName}>
                {stage.replace(/_/g, ' ')}
              </span>
              <span style={{
                ...styles.stageStatus,
                color: status === 'completed' ? '#1D9E75'
                     : status === 'running'   ? '#7F77DD'
                     : '#888'
              }}>
                {status}
              </span>
            </div>
          ))}

          {job.status === 'failed' && (
            <p style={styles.error}>{job.error_message}</p>
          )}
        </div>
      )}

      {pollError && <p style={styles.error}>{pollError}</p>}
    </div>
  )
}

const styles = {
  container: {
    maxWidth: '600px',
    margin: '80px auto',
    padding: '0 24px',
    fontFamily: 'system-ui, sans-serif',
  },
  title: {
    fontSize: '32px',
    fontWeight: '600',
    color: '#1a1a1a',
    marginBottom: '8px',
  },
  subtitle: {
    fontSize: '16px',
    color: '#666',
    marginBottom: '40px',
  },
  uploadBox: {
    border: '2px dashed #ccc',
    borderRadius: '12px',
    padding: '48px',
    textAlign: 'center',
    cursor: 'pointer',
  },
  uploadText: {
    fontSize: '16px',
    color: '#444',
    marginBottom: '16px',
  },
  fileInput: {
    fontSize: '14px',
    cursor: 'pointer',
  },
  progressBox: {
    background: '#f9f9f9',
    borderRadius: '12px',
    padding: '24px',
    marginTop: '24px',
  },
  progressTitle: {
    fontSize: '18px',
    fontWeight: '500',
    marginBottom: '20px',
    color: '#1a1a1a',
  },
  stageRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '10px 0',
    borderBottom: '1px solid #eee',
  },
  stageIcon: {
    fontSize: '18px',
    width: '24px',
    textAlign: 'center',
  },
  stageName: {
    flex: 1,
    fontSize: '15px',
    color: '#333',
    textTransform: 'capitalize',
  },
  stageStatus: {
    fontSize: '13px',
    fontWeight: '500',
  },
  info: { color: '#7F77DD', marginTop: '12px' },
  error: { color: '#E24B4A', marginTop: '12px' },
}