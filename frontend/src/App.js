import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import UploadPage    from './pages/UploadPage'
import DashboardPage from './pages/DashboardPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"                    element={<UploadPage />} />
        <Route path="/dashboard/:jobId"    element={<DashboardPage />} />
        <Route path="*"                    element={<Navigate to="/" />} />
      </Routes>
    </BrowserRouter>
  )
}