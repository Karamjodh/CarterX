const COLORS = ['#7F77DD','#1D9E75','#D85A30','#BA7517','#185FA5']
const LIGHTS  = ['#EEEDFE','#E1F5EE','#FAECE7','#FAEEDA','#E6F1FB']

export default function ClustersTab({ insights }) {
  const clusters = insights.cluster_profiles || []

  return (
    <div>
      <div style={S.grid}>
        {clusters.map((c, i) => (
          <div key={c.cluster_id} style={{
            ...S.card,
            borderTop: `3px solid ${COLORS[i % COLORS.length]}`
          }}>
            <div style={{
              ...S.ring,
              background: LIGHTS[i % LIGHTS.length],
              color:      COLORS[i % COLORS.length],
            }}>
              {c.pct_of_customers}%
            </div>
            <div style={S.clusterName}>{c.label}</div>
            <div style={S.clusterSize}>{c.size} customers</div>
            <div style={S.divider} />
            {[
              ['Avg spend',    `$${c.avg_monetary.toLocaleString()}`],
              ['Last bought',  `${c.avg_recency_days}d ago`],
              ['Frequency',    `${c.avg_frequency}× per period`],
            ].map(([k,v]) => (
              <div key={k} style={S.metaRow}>
                <span style={S.metaKey}>{k}</span>
                <span style={S.metaVal}>{v}</span>
              </div>
            ))}
            <div style={{...S.barTrack, marginTop:12}}>
              <div style={{
                height:'100%',
                width:`${c.pct_of_customers}%`,
                background: COLORS[i % COLORS.length],
                borderRadius:4,
              }} />
            </div>
          </div>
        ))}
      </div>

      <div style={S.section}>
        <div style={S.secTitle}>Segment comparison</div>
        <div style={S.compareGrid}>
          {[
            { key:'avg_monetary',    label:'Average spend',    fmt: v => `$${v.toLocaleString()}` },
            { key:'avg_frequency',   label:'Purchase frequency', fmt: v => `${v}×` },
            { key:'avg_recency_days',label:'Days since purchase', fmt: v => `${v}d` },
          ].map(metric => {
            const max = Math.max(...clusters.map(c => c[metric.key]))
            return (
              <div key={metric.key}>
                <div style={S.metricLabel}>{metric.label}</div>
                {clusters.map((c,i) => (
                  <div key={c.cluster_id} style={{marginBottom:10}}>
                    <div style={{display:'flex', justifyContent:'space-between', fontSize:12, marginBottom:3}}>
                      <span style={{color:'#5F5E5A'}}>{c.label}</span>
                      <span style={{fontWeight:500}}>{metric.fmt(c[metric.key])}</span>
                    </div>
                    <div style={S.barTrack}>
                      <div style={{
                        height:'100%',
                        width:`${(c[metric.key]/max)*100}%`,
                        background: COLORS[i % COLORS.length],
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

const S = {
  grid:        { display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(200px, 1fr))', gap:12, marginBottom:12 },
  card:        { background:'white', border:'0.5px solid #E8E6DF', borderRadius:12, padding:'1.25rem' },
  ring:        { width:52, height:52, borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center', fontSize:14, fontWeight:500, marginBottom:12 },
  clusterName: { fontSize:16, fontFamily:"'Instrument Serif', serif", color:'#1a1a1a', marginBottom:2 },
  clusterSize: { fontSize:12, color:'#888780', marginBottom:12 },
  divider:     { height:'0.5px', background:'#E8E6DF', marginBottom:12 },
  metaRow:     { display:'flex', justifyContent:'space-between', fontSize:12, marginBottom:6 },
  metaKey:     { color:'#888780' },
  metaVal:     { fontWeight:500, color:'#1a1a1a' },
  barTrack:    { height:6, background:'#F1EFE8', borderRadius:4, overflow:'hidden' },
  section:     { background:'white', border:'0.5px solid #E8E6DF', borderRadius:12, padding:'1.25rem' },
  secTitle:    { fontSize:16, fontFamily:"'Instrument Serif', serif", color:'#1a1a1a', marginBottom:16 },
  compareGrid: { display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:24 },
  metricLabel: { fontSize:12, fontWeight:500, color:'#5F5E5A', marginBottom:12 },
}