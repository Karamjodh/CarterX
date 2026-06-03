import axios from 'axios'

// This is the base URL of your FastAPI backend
// In development it runs on port 8000
const API_BASE = 'http://127.0.0.1:8000/api/v1'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,   // 30 second timeout
})

// ── Jobs ──────────────────────────────────────────────────────────────
export const getJob = (jobId) => api.get(`/jobs/${jobId}`)
export const getAllJobs = () => api.get('/jobs/')

// ── Uploads ───────────────────────────────────────────────────────────
export const uploadFile = (file) => {
  // Files must be sent as FormData — not JSON
  const formData = new FormData()
  formData.append('file', file)
  return api.post('/uploads/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

// ── Insights ──────────────────────────────────────────────────────────
export const getInsights = (jobId) => api.get(`/insights/${jobId}`)

// ── Reports ───────────────────────────────────────────────────────────
export const analyzeReport = (data, model = 'groq', focus = 'general') =>
  api.post('/reports/analyze', { data, model, focus })