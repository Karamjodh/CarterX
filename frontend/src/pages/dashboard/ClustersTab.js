const COLORS = ['#7F77DD','#1D9E75','#D85A30','#BA7517','#185FA5']
const BG     = ['#EEEDFE','#E1F5EE','#FAECE7','#FAEEDA','#E6F1FB']

export default function ClustersTab({ insights }) {
  const clusters = insights.cluster_profiles || []
  const total    = clusters.reduce((s, c) => s + c.size, 0)

  return (
    <div>
      <div style={styles.grid}>
        {clusters.map((c, i) => (
          <div key={c.cluster_id} style={{...styles.card, borderTop:`3px solid ${COLORS[i % COLORS.length]}`}}>
            <div style={{...styles.ring, background:BG[i % BG.length], color:COLORS[i % COLORS.length]}}>
              {c.pct_of_customers}%
            </div>
            <div style={styles.label}>{c.label}</div>
            <div style={styles.size}>{c.size} customers</div>
            <div style={styles.divider} />
            <div style={styles.metaRow}>
              <span style={styles.metaLabel}>Avg spend</span>
              <span style={styles.metaVal}>${c.avg_monetary.toLocaleString()}</span>
            </div>
            <div style={styles.metaRow}>
              <span style={styles.metaLabel}>Last bought</span>
              <span style={styles.metaVal}>{c.avg_recency_days}d ago</span>
            </div>
            <div style={styles.metaRow}>
              <span style={styles.metaLabel}>Frequency</span>
              <span style={styles.metaVal}>{c.avg_frequency}x</span>
            </div>
            <div style={styles.barWrap}>
              <div style={{...styles.bar, width:`${c.pct_of_customers}%`, background:COLORS[i % COLORS.length]}} />
            </div>
          </div>
        ))}
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Segment comparison</div>
        <div style={styles.compareGrid}>
          {['avg_monetary','avg_frequency','avg_recency_days'].map(metric => {
            const max = Math.max(...clusters.map(c => c[metric]))
            const labels = { avg_monetary:'Avg spend ($)', avg_frequency:'Buy frequency', avg_recency_days:'Recency (days)' }
            return (
              <div key={metric}>
                <div style={styles.metricLabel}>{labels[metric]}</div>
                {clusters.map((c,i) => (
                  <div key={c.cluster_id} style={{marginBottom:8}}>
                    <div style={{display:'flex', justifyContent:'space-between', fontSize:12, marginBottom:3}}>
                      <span style={{color:'var(--color-text-secondary)'}}>{c.label}</span>
                      <span style={{fontWeight:500, color:'var(--color-text-primary)'}}>
                        {metric === 'avg_monetary' ? `$${c[metric].toLocaleString()}` : c[metric]}
                      </span>
                    </div>
                    <div style={styles.trackWrap}>
                      <div style={{
                        height:'100%',
                        width:`${(c[metric]/max)*100}%`,
                        background:COLORS[i % COLORS.length],
                        borderRadius:4,
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

const styles = {
  grid:        { display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(200px, 1fr))', gap:12, marginBottom:'1.5rem' },
  card:        { background:'var(--color-background-primary)', border:'0.5px solid var(--color-border-tertiary)', borderRadius:'var(--border-radius-lg)', padding:'1.25rem' },
  ring:        { width:56, height:56, borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center', fontSize:14, fontWeight:500, marginBottom:12 },
  label:       { fontSize:15, fontWeight:500, color:'var(--color-text-primary)', marginBottom:2 },
  size:        { fontSize:12, color:'var(--color-text-secondary)', marginBottom:12 },
  divider:     { height:'0.5px', background:'var(--color-border-tertiary)', marginBottom:12 },
  metaRow:     { display:'flex', justifyContent:'space-between', fontSize:12, marginBottom:6 },
  metaLabel:   { color:'var(--color-text-secondary)' },
  metaVal:     { fontWeight:500, color:'var(--color-text-primary)' },
  barWrap:     { height:4, background:'var(--color-background-secondary)', borderRadius:4, marginTop:12 },
  bar:         { height:'100%', borderRadius:4 },
  section:     { background:'var(--color-background-primary)', border:'0.5px solid var(--color-border-tertiary)', borderRadius:'var(--border-radius-lg)', padding:'1.25rem' },
  sectionTitle:{ fontSize:14, fontWeight:500, color:'var(--color-text-primary)', marginBottom:'1rem' },
  compareGrid: { display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:24 },
  metricLabel: { fontSize:12, fontWeight:500, color:'var(--color-text-secondary)', marginBottom:12 },
  trackWrap:   { height:8, background:'var(--color-background-secondary)', borderRadius:4 },
}