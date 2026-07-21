import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getInsights } from '../services/api'
import VisualizationsTab from './dashboard/VisualizationsTab'
import ReportTab         from './dashboard/ReportTab'
import RulesTab          from './dashboard/RulesTab'
import ClustersTab       from './dashboard/ClustersTab'
import StatsTab          from './dashboard/StatsTab'
import SimulationTab     from './dashboard/SimulationTab'
import ForecastTab       from './dashboard/ForecastTab'
import GeoTab            from './dashboard/GeoTab'

// ── Auto-scaling revenue formatter ────────────────────────────────────────────
function fmtRevenue(n) {
  if (n == null || isNaN(n)) return '—'
  if (Math.abs(n) >= 1e12) return `$${(n / 1e12).toFixed(2)}T`
  if (Math.abs(n) >= 1e9)  return `$${(n / 1e9).toFixed(2)}B`
  if (Math.abs(n) >= 1e6)  return `$${(n / 1e6).toFixed(2)}M`
  if (Math.abs(n) >= 1e3)  return `$${(n / 1e3).toFixed(1)}K`
  return `$${Math.round(n).toLocaleString()}`
}

const NAV = [
  { id: 'visualizations', icon: '▦', label: 'Visualizations',  sub: 'Charts & trends'     },
  { id: 'clusters',       icon: '◉', label: 'Segments',         sub: 'Customer groups'     },
  { id: 'simulation',     icon: '⟳', label: 'Simulation',       sub: 'What-if analysis'    },
  { id: 'rules',          icon: '⇌', label: 'Rules',            sub: 'Purchase patterns'   },
  { id: 'forecast',       icon: '⟠', label: 'Forecast',         sub: 'Revenue prediction'  },
  { id: 'report',         icon: '✦', label: 'AI Report',        sub: 'Strategy & insights' },
  { id: 'stats',          icon: '≡', label: 'Statistics',       sub: 'Full data breakdown' },
  { id: 'geo',            icon: '⊕', label: 'Geography',       sub: 'Regional analysis' },
]

export default function DashboardPage() {
  const { jobId }                 = useParams()
  const navigate                  = useNavigate()
  const [insights,  setInsights]  = useState(null)
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState(null)
  const [active,    setActive]    = useState('visualizations')
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    getInsights(jobId)
      .then(r => setInsights(r.data))
      .catch(() => setError('Could not load insights.'))
      .finally(() => setLoading(false))
  }, [jobId])

  if (loading)   return <LoadingScreen />
  if (error)     return <ErrorScreen message={error} />
  if (!insights) return null

  const s = insights.summary || {}

  const tabs = {
    visualizations: <VisualizationsTab insights={insights} />,
    clusters:       <ClustersTab       insights={insights} />,
    simulation:     <SimulationTab     insights={insights} />,
    rules:          <RulesTab          insights={insights} />,
    forecast:       <ForecastTab       insights={insights} />,
    report:         <ReportTab         insights={insights} jobId={jobId} />,
    stats:          <StatsTab          insights={insights} />,
    geo:            <GeoTab            insights={insights} />,
  }

  return (
    <div style={S.shell}>

      {/* ── Sidebar ── */}
      <aside style={{ ...S.sidebar, width: collapsed ? 64 : 220 }}>
        <div style={S.sidebarTop}>
          {!collapsed && (
            <div style={S.brand}>
              <span style={S.brandName}>CarterX</span>
              <span style={S.brandTag}>AI Analytics</span>
            </div>
          )}
          <button onClick={() => setCollapsed(c => !c)} style={S.collapseBtn}>
            {collapsed ? '→' : '←'}
          </button>
        </div>

        <nav style={S.nav}>
          {NAV.map(item => (
            <button
              key={item.id}
              onClick={() => setActive(item.id)}
              style={{
                ...S.navItem,
                background: active === item.id ? '#EEEDFE' : 'transparent',
                color:      active === item.id ? '#534AB7' : '#5F5E5A',
              }}
              title={collapsed ? item.label : ''}
            >
              <span style={{
                ...S.navIcon,
                color: active === item.id ? '#534AB7' : '#888780',
              }}>
                {item.icon}
              </span>
              {!collapsed && (
                <span style={S.navText}>
                  <span style={S.navLabel}>{item.label}</span>
                  <span style={S.navSub}>{item.sub}</span>
                </span>
              )}
            </button>
          ))}
        </nav>

        {!collapsed && (
          <div style={S.sidebarBottom}>
            <button onClick={() => navigate('/')} style={S.newUploadBtn}>
              + New upload
            </button>
            <div style={S.sidebarMeta}>
              <div style={S.metaLine}>{s.total_customers?.toLocaleString()} customers</div>
              <div style={S.metaLine}>{fmtRevenue(s.total_revenue)} revenue</div>
            </div>
          </div>
        )}
      </aside>

      {/* ── Main content ── */}
      <main style={S.main}>

        {/* ── Top header ── */}
        <header style={S.header}>
          <div>
            <h1 style={S.pageTitle}>
              {NAV.find(n => n.id === active)?.label}
            </h1>
            <p style={S.pageSubtitle}>
              {s.date_start && s.date_start !== 'N/A'
                ? `${s.date_start} → ${s.date_end} · `
                : ''}
              {s.total_transactions?.toLocaleString()} transactions
            </p>
          </div>
          <div style={S.headerKPIs}>
            {[
              ['Revenue',   fmtRevenue(s.total_revenue)],
              ['Customers', s.total_customers?.toLocaleString() ?? '—'],
              ['Segments',  insights.n_clusters ?? '—'],
              ['Rules',     insights.association_rules?.length ?? '—'],
            ].map(([label, val]) => (
              <div key={label} style={S.kpi}>
                <div style={S.kpiVal}>{val}</div>
                <div style={S.kpiLabel}>{label}</div>
              </div>
            ))}
          </div>
        </header>

        {/* ── Tab content ── */}
        <div style={S.content}>
          {tabs[active]}
        </div>
      </main>
    </div>
  )
}

function LoadingScreen() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', gap: 16, fontFamily: "'Inter', sans-serif" }}>
      <div style={{ fontSize: 28, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a' }}>CarterX</div>
      <div style={{ fontSize: 13, color: '#888780' }}>Loading your insights...</div>
    </div>
  )
}

function ErrorScreen({ message }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', fontFamily: "'Inter', sans-serif" }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 20, fontFamily: "'Instrument Serif', serif", marginBottom: 8 }}>Something went wrong</div>
        <div style={{ fontSize: 13, color: '#888780' }}>{message}</div>
      </div>
    </div>
  )
}

const S = {
  shell:         { display: 'flex', height: '100vh', overflow: 'hidden', background: '#F7F6F3' },
  sidebar:       { display: 'flex', flexDirection: 'column', background: 'white', borderRight: '0.5px solid #E8E6DF', transition: 'width 0.2s ease', overflow: 'hidden', flexShrink: 0 },
  sidebarTop:    { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '20px 16px 16px', borderBottom: '0.5px solid #E8E6DF' },
  brand:         { display: 'flex', flexDirection: 'column' },
  brandName:     { fontSize: 18, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', lineHeight: 1.2 },
  brandTag:      { fontSize: 10, color: '#888780', letterSpacing: '0.06em', textTransform: 'uppercase' },
  collapseBtn:   { background: '#F7F6F3', border: '0.5px solid #E8E6DF', borderRadius: 6, width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, color: '#888780', cursor: 'pointer', flexShrink: 0 },
  nav:           { flex: 1, padding: '12px 8px', display: 'flex', flexDirection: 'column', gap: 2, overflowY: 'auto' },
  navItem:       { display: 'flex', alignItems: 'center', gap: 10, padding: '8px 10px', borderRadius: 8, border: 'none', cursor: 'pointer', transition: 'all 0.15s', textAlign: 'left', width: '100%' },
  navIcon:       { fontSize: 16, width: 20, textAlign: 'center', flexShrink: 0, lineHeight: 1 },
  navText:       { display: 'flex', flexDirection: 'column', minWidth: 0 },
  navLabel:      { fontSize: 13, fontWeight: 500, color: 'inherit', whiteSpace: 'nowrap' },
  navSub:        { fontSize: 11, color: '#B4B2A9', whiteSpace: 'nowrap' },
  sidebarBottom: { padding: '12px 16px', borderTop: '0.5px solid #E8E6DF' },
  newUploadBtn:  { width: '100%', padding: '8px', background: '#EEEDFE', color: '#534AB7', border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 500, cursor: 'pointer', marginBottom: 12 },
  sidebarMeta:   { display: 'flex', flexDirection: 'column', gap: 3 },
  metaLine:      { fontSize: 11, color: '#888780' },
  main:          { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  header:        { background: 'white', borderBottom: '0.5px solid #E8E6DF', padding: '20px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 },
  pageTitle:     { fontSize: 24, fontFamily: "'Instrument Serif', serif", color: '#1a1a1a', marginBottom: 2 },
  pageSubtitle:  { fontSize: 12, color: '#888780' },
  headerKPIs:    { display: 'flex', gap: 24 },
  kpi:           { textAlign: 'right' },
  kpiVal:        { fontSize: 18, fontWeight: 500, color: '#1a1a1a' },
  kpiLabel:      { fontSize: 11, color: '#888780' },
  content:       { flex: 1, overflowY: 'auto', padding: '28px 32px' },
}