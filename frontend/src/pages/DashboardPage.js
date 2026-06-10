import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getInsights } from '../services/api'
import VisualizationsTab from './dashboard/VisualizationsTab'
import ReportTab         from './dashboard/ReportTab'
import RulesTab          from './dashboard/RulesTab'
import ClustersTab       from './dashboard/ClustersTab'
import StatsTab          from './dashboard/StatsTab'

const TABS = [
  { id:'visualizations', label:'Visualizations' },
  { id:'report',         label:'LLM Report'     },
  { id:'rules',          label:'Rules'          },
  { id:'clusters',       label:'Clusters'       },
  { id:'stats',          label:'Stats'          },
]

export default function DashboardPage() {
  const { jobId }                   = useParams()
  const navigate                    = useNavigate()
  const [insights,  setInsights]    = useState(null)
  const [loading,   setLoading]     = useState(true)
  const [error,     setError]       = useState(null)
  const [activeTab, setActiveTab]   = useState('visualizations')

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const res = await getInsights(jobId)
        setInsights(res.data)
      } catch (e) {
        setError('Could not load insights.')
      } finally {
        setLoading(false)
      }
    }
    fetchInsights()
  }, [jobId])

  if (loading) return <div style={styles.center}>Loading insights...</div>
  if (error)   return <div style={styles.center}>{error}</div>
  if (!insights) return null

  const s = insights.summary || {}

  return (
    <div style={styles.container}>
      <div style={styles.topbar}>
        <div>
          <span style={styles.logo}>CarterX</span>
          <span style={styles.badge}>{insights.job_id?.slice(0,8)}...</span>
        </div>
        <div style={styles.meta}>
          {s.total_customers} customers · {s.total_transactions?.toLocaleString()} transactions · ${s.total_revenue?.toLocaleString()} revenue
        </div>
        <button onClick={() => navigate('/')} style={styles.newBtn}>
          New upload
        </button>
      </div>

      <div style={styles.summaryCards}>
        {[
          ['Total Revenue',  `$${(s.total_revenue/1e6).toFixed(2)}M`, 'total'],
          ['Customers',       s.total_customers,                       'unique'],
          ['Avg Order',      `$${s.avg_order_value}`,                 'per transaction'],
          ['Segments',        insights.n_clusters,                     'found'],
          ['Rules',           insights.association_rules?.length,      'patterns'],
          ['Silhouette',      insights.silhouette_score,               'cluster quality'],
        ].map(([label, val, sub]) => (
          <div key={label} style={styles.card}>
            <div style={styles.cardLabel}>{label}</div>
            <div style={styles.cardVal}>{val}</div>
            <div style={styles.cardSub}>{sub}</div>
          </div>
        ))}
      </div>

      <div style={styles.tabs}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              ...styles.tab,
              color:       activeTab === tab.id ? '#534AB7' : 'var(--color-text-secondary)',
              borderBottom: activeTab === tab.id ? '2px solid #534AB7' : '2px solid transparent',
              fontWeight:  activeTab === tab.id ? 500 : 400,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div style={styles.content}>
        {activeTab === 'visualizations' && <VisualizationsTab insights={insights} />}
        {activeTab === 'report'         && <ReportTab         insights={insights} jobId={jobId} />}
        {activeTab === 'rules'          && <RulesTab          insights={insights} />}
        {activeTab === 'clusters'       && <ClustersTab       insights={insights} />}
        {activeTab === 'stats'          && <StatsTab          insights={insights} />}
      </div>
    </div>
  )
}

const styles = {
  container:    { maxWidth:960, margin:'0 auto', padding:'2rem 1.5rem', fontFamily:'system-ui, sans-serif' },
  center:       { textAlign:'center', padding:80, color:'var(--color-text-secondary)', fontFamily:'system-ui, sans-serif' },
  topbar:       { display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'1.5rem' },
  logo:         { fontSize:20, fontWeight:500, color:'var(--color-text-primary)' },
  badge:        { fontSize:11, background:'#EEEDFE', color:'#534AB7', padding:'2px 8px', borderRadius:999, marginLeft:8, fontWeight:500 },
  meta:         { fontSize:13, color:'var(--color-text-secondary)' },
  newBtn:       { fontSize:13, padding:'6px 14px', background:'var(--color-background-secondary)', border:'0.5px solid var(--color-border-secondary)', borderRadius:'var(--border-radius-md)', cursor:'pointer', color:'var(--color-text-primary)' },
  summaryCards: { display:'grid', gridTemplateColumns:'repeat(6,1fr)', gap:10, marginBottom:'1.5rem' },
  card:         { background:'var(--color-background-secondary)', borderRadius:'var(--border-radius-md)', padding:'0.75rem 1rem' },
  cardLabel:    { fontSize:11, color:'var(--color-text-secondary)', marginBottom:4 },
  cardVal:      { fontSize:18, fontWeight:500, color:'var(--color-text-primary)' },
  cardSub:      { fontSize:10, color:'var(--color-text-tertiary)', marginTop:2 },
  tabs:         { display:'flex', gap:2, borderBottom:'0.5px solid var(--color-border-tertiary)', marginBottom:'1.5rem' },
  tab:          { padding:'8px 16px', fontSize:13, background:'none', border:'none', borderBottom:'2px solid transparent', cursor:'pointer', marginBottom:'-0.5px' },
  content:      {},
}