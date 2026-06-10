import { useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'

const COLORS = ['#7F77DD','#1D9E75','#D85A30','#BA7517','#185FA5']

export default function VisualizationsTab({ insights }) {
  const trend    = insights.trend_data || {}
  const monthly  = trend.monthly_revenue || []
  const catData  = trend.category_monthly || {}
  const topProds = trend.top_products || []
  const clusters = insights.cluster_profiles || []

  const [discount,  setDiscount]  = useState(15)
  const [horizon,   setHorizon]   = useState(90)
  const [targetSeg, setTargetSeg] = useState('all')

  const segmentData = clusters.map(c => ({ name: c.label, value: c.size }))

  const totalRev    = insights.summary?.total_revenue || 0
  const dailyRate   = totalRev / 365
  const targetCusts = targetSeg === 'all'
    ? clusters.reduce((s, c) => s + c.size, 0)
    : clusters.find(c => c.label === targetSeg)?.size || 0
  const totalCusts  = clusters.reduce((s, c) => s + c.size, 0)
  const targetPct   = totalCusts > 0 ? targetCusts / totalCusts : 1
  const volUplift   = Math.abs(-1.5) * (discount / 100)
  const revMulti    = (1 - discount / 100) * (1 + volUplift)
  const baseProj    = Math.round(dailyRate * horizon * targetPct)
  const simProj     = Math.round(baseProj * revMulti)
  const liftPct     = baseProj > 0 ? ((simProj - baseProj) / baseProj * 100).toFixed(1) : 0

  const simChartData = [
    { label: 'Baseline', revenue: baseProj },
    { label: 'Simulated', revenue: simProj },
  ]

  return (
    <div>
      <div style={styles.twoCol}>
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Monthly revenue</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={monthly} margin={{top:5, right:10, bottom:5, left:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
              <XAxis dataKey="month" tick={{fontSize:10}} tickFormatter={m => m.slice(5)} />
              <YAxis tick={{fontSize:10}} tickFormatter={v => `$${(v/1000).toFixed(0)}K`} />
              <Tooltip formatter={v => [`$${v.toLocaleString()}`, 'Revenue']} />
              <Bar dataKey="total_revenue" fill="#7F77DD" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={styles.section}>
          <div style={styles.sectionTitle}>Customer segments</div>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={segmentData} cx="50%" cy="50%" outerRadius={75}
                dataKey="value"
                label={({name, percent}) => `${name} ${(percent*100).toFixed(0)}%`}
                labelLine={false}
              >
                {segmentData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {topProds.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Top 10 products by revenue</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={topProds} layout="vertical"
              margin={{top:5, right:40, bottom:5, left:80}}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
              <XAxis type="number" tick={{fontSize:10}} tickFormatter={v => `$${(v/1000).toFixed(0)}K`} />
              <YAxis type="category" dataKey="product_name" tick={{fontSize:11}} width={75} />
              <Tooltip formatter={v => [`$${v.toLocaleString()}`, 'Revenue']} />
              <Bar dataKey="total_revenue" fill="#1D9E75" radius={[0,3,3,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div style={styles.section}>
        <div style={styles.sectionTitle}>What-if simulation</div>
        <div style={styles.simGrid}>
          <div>
            <div style={styles.simRow}>
              <span style={styles.simLabel}>Target segment</span>
              <select value={targetSeg} onChange={e => setTargetSeg(e.target.value)} style={{flex:1}}>
                <option value="all">All segments</option>
                {clusters.map(c => (
                  <option key={c.cluster_id} value={c.label}>{c.label} ({c.pct_of_customers}%)</option>
                ))}
              </select>
            </div>
            <div style={styles.simRow}>
              <span style={styles.simLabel}>Discount</span>
              <input type="range" min="0" max="50" step="1"
                value={discount} onChange={e => setDiscount(Number(e.target.value))}
                style={{flex:1}} />
              <span style={styles.simVal}>{discount}%</span>
            </div>
            <div style={styles.simRow}>
              <span style={styles.simLabel}>Time horizon</span>
              <input type="range" min="30" max="365" step="30"
                value={horizon} onChange={e => setHorizon(Number(e.target.value))}
                style={{flex:1}} />
              <span style={styles.simVal}>{horizon}d</span>
            </div>
          </div>
          <div style={styles.simResults}>
            <div style={styles.kpiGrid}>
              {[
                ['Targeted customers', targetCusts],
                ['Baseline revenue',   `$${baseProj.toLocaleString()}`],
                ['Projected revenue',  `$${simProj.toLocaleString()}`],
                ['Revenue lift',       `${liftPct}%`],
              ].map(([label, val]) => (
                <div key={label} style={styles.kpiCard}>
                  <div style={styles.kpiLabel}>{label}</div>
                  <div style={styles.kpiVal}>{val}</div>
                </div>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={simChartData} margin={{top:5,right:10,bottom:5,left:10}}>
                <XAxis dataKey="label" tick={{fontSize:12}} />
                <YAxis hide />
                <Tooltip formatter={v => [`$${v.toLocaleString()}`, 'Revenue']} />
                <Bar dataKey="revenue" radius={[4,4,0,0]}>
                  <Cell fill="#AFA9EC" />
                  <Cell fill="#7F77DD" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  twoCol:     { display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1rem', marginBottom:'1rem' },
  section:    { background:'var(--color-background-primary)', border:'0.5px solid var(--color-border-tertiary)', borderRadius:'var(--border-radius-lg)', padding:'1.25rem', marginBottom:'1rem' },
  sectionTitle:{ fontSize:14, fontWeight:500, color:'var(--color-text-primary)', marginBottom:'1rem' },
  simGrid:    { display:'grid', gridTemplateColumns:'1fr 1fr', gap:24 },
  simRow:     { display:'flex', alignItems:'center', gap:12, marginBottom:16 },
  simLabel:   { fontSize:13, color:'var(--color-text-secondary)', width:120, flexShrink:0 },
  simVal:     { fontSize:13, fontWeight:500, color:'var(--color-text-primary)', width:40, textAlign:'right' },
  simResults: {},
  kpiGrid:    { display:'grid', gridTemplateColumns:'1fr 1fr', gap:8, marginBottom:12 },
  kpiCard:    { background:'var(--color-background-secondary)', borderRadius:'var(--border-radius-md)', padding:'10px 12px' },
  kpiLabel:   { fontSize:11, color:'var(--color-text-secondary)', marginBottom:4 },
  kpiVal:     { fontSize:18, fontWeight:500, color:'var(--color-text-primary)' },
}