import axios from 'axios'

const API_BASE = 'http://localhost:8000/api/v1'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
})

export const getJob     = (jobId) => api.get(`/jobs/${jobId}`)
export const getAllJobs  = ()      => api.get('/jobs/')

export const uploadFile = (file) => {
  const formData = new FormData()
  formData.append('file', file)
  return axios.post(`${API_BASE}/uploads/`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,   // 5 minutes for large files
  })
}

export const getInsights   = (jobId)                          => api.get(`/insights/${jobId}`)
export const analyzeReport = (data, model = 'groq', focus = 'general') =>
  api.post('/reports/analyze', { data, model, focus })