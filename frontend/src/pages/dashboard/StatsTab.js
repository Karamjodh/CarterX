export default function StatsTab({ insights }) {
  const s = insights.summary || {}
  const rules = insights.association_rules || []
  const clusters = insights.cluster_profiles || []

  const rows = [
    ["Total customers",       s.total_customers?.toLocaleString()],
    ["Total transactions",    s.total_transactions?.toLocaleString()],
    ["Total revenue",         `$${s.total_revenue?.toLocaleString()}`],
    ["Average order value",   `$${s.avg_order_value}`],
    ["Date range",            `${s.date_start} → ${s.date_end}`],
    ["Rows removed (cleaning)", s.rows_removed],
    ["Customer segments",     insights.n_clusters],
    ["Silhouette score",      insights.silhouette_score],
    ["Association rules found", rules.length],
    ["Months of data",        insights.trend_data?.monthly_revenue?.length || 0],
  ]

  return (
    <div>
      <div style={styles.grid}>
        {[
          ["Total Revenue",    `$${(s.total_revenue/1e6).toFixed(2)}M`, "Jan–Dec"],
          ["Unique Customers", s.total_customers,                       "analyzed"],
          ["Transactions",     s.total_transactions?.toLocaleString(), "total"],
          ["Avg Order Value",  `$${s.avg_order_value}`,                "per order"],
          ["Segments Found",   insights.n_clusters,                    "clusters"],
          ["Rules Discovered", rules.length,                           "patterns"],
        ].map(([label, val, sub]) => (
          <div key={label} style={styles.card}>
            <div style={styles.cardLabel}>{label}</div>
            <div style={styles.cardVal}>{val}</div>
            <div style={styles.cardSub}>{sub}</div>
          </div>
        ))}
      </div>

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Full dataset statistics</div>
        <table style={styles.table}>
          <tbody>
            {rows.map(([label, val]) => (
              <tr key={label} style={styles.tr}>
                <td style={styles.tdLabel}>{label}</td>
                <td style={styles.tdVal}>{val}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {s.top_categories && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Revenue by category</div>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Category</th>
                <th style={{...styles.th, textAlign:'right'}}>Revenue</th>
                <th style={{...styles.th, textAlign:'right'}}>Share</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(s.top_categories)
                .sort((a,b) => b[1]-a[1])
                .map(([cat, rev]) => (
                  <tr key={cat} style={styles.tr}>
                    <td style={styles.tdLabel}>{cat}</td>
                    <td style={{...styles.tdVal, textAlign:'right'}}>${rev.toLocaleString()}</td>
                    <td style={{...styles.tdVal, textAlign:'right', color:'#7F77DD'}}>
                      {((rev / s.total_revenue) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      <div style={styles.section}>
        <div style={styles.sectionTitle}>Segment summary</div>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Segment</th>
              <th style={{...styles.th,textAlign:'right'}}>Customers</th>
              <th style={{...styles.th,textAlign:'right'}}>Avg spend</th>
              <th style={{...styles.th,textAlign:'right'}}>Avg recency</th>
              <th style={{...styles.th,textAlign:'right'}}>Avg frequency</th>
            </tr>
          </thead>
          <tbody>
            {clusters.map(c => (
              <tr key={c.cluster_id} style={styles.tr}>
                <td style={styles.tdLabel}>{c.label}</td>
                <td style={{...styles.tdVal,textAlign:'right'}}>{c.size} ({c.pct_of_customers}%)</td>
                <td style={{...styles.tdVal,textAlign:'right'}}>${c.avg_monetary.toLocaleString()}</td>
                <td style={{...styles.tdVal,textAlign:'right'}}>{c.avg_recency_days} days</td>
                <td style={{...styles.tdVal,textAlign:'right'}}>{c.avg_frequency}x</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const styles = {
  grid:       { display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12, marginBottom:'1.5rem' },
  card:       { background:'var(--color-background-secondary)', borderRadius:'var(--border-radius-md)', padding:'1rem' },
  cardLabel:  { fontSize:12, color:'var(--color-text-secondary)', marginBottom:6 },
  cardVal:    { fontSize:22, fontWeight:500, color:'var(--color-text-primary)' },
  cardSub:    { fontSize:11, color:'var(--color-text-tertiary)', marginTop:3 },
  section:    { background:'var(--color-background-primary)', border:'0.5px solid var(--color-border-tertiary)', borderRadius:'var(--border-radius-lg)', padding:'1.25rem', marginBottom:'1rem' },
  sectionTitle:{ fontSize:14, fontWeight:500, color:'var(--color-text-primary)', marginBottom:'1rem' },
  table:      { width:'100%', borderCollapse:'collapse', fontSize:13 },
  th:         { textAlign:'left', fontWeight:500, fontSize:12, color:'var(--color-text-secondary)', padding:'6px 0', borderBottom:'0.5px solid var(--color-border-tertiary)' },
  tr:         { borderBottom:'0.5px solid var(--color-border-tertiary)' },
  tdLabel:    { padding:'8px 0', color:'var(--color-text-secondary)' },
  tdVal:      { padding:'8px 0', fontWeight:500, color:'var(--color-text-primary)' },
}