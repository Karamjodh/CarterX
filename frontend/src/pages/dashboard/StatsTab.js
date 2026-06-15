export default function StatsTab({ insights }) {
  const s        = insights.summary || {}
  const rules    = insights.association_rules || []
  const clusters = insights.cluster_profiles || []
  const trend    = insights.trend_data || {}

  const statRows = [
    ['Total customers',         s.total_customers?.toLocaleString()],
    ['Total transactions',      s.total_transactions?.toLocaleString()],
    ['Total revenue',           `$${s.total_revenue?.toLocaleString()}`],
    ['Average order value',     `$${s.avg_order_value}`],
    ['Date range',              `${s.date_start} → ${s.date_end}`],
    ['Rows cleaned',            s.rows_removed],
    ['Segments found',          insights.n_clusters],
    ['Silhouette score',        `${insights.silhouette_score} (${
      insights.silhouette_score >= 0.5 ? 'strong' :
      insights.silhouette_score >= 0.3 ? 'moderate' : 'weak'
    })`],
    ['Association rules',       rules.length],
    ['Months of data',          trend.monthly_revenue?.length || 0],
    ['Month-over-month growth', trend.mom_growth_pct ? `${trend.mom_growth_pct > 0 ? '+' : ''}${trend.mom_growth_pct}%` : 'N/A'],
  ]

  return (
    <div>
      <div style={S.topCards}>
        {[
          ['Total Revenue',   `$${(s.total_revenue/1e6)?.toFixed(2)}M`, '#534AB7', '#EEEDFE'],
          ['Customers',        s.total_customers,                        '#0F6E56', '#E1F5EE'],
          ['Avg Order',       `$${s.avg_order_value}`,                  '#993C1D', '#FAECE7'],
          ['Rules Found',      rules.length,                             '#534AB7', '#EEEDFE'],
          ['Segments',         insights.n_clusters,                      '#0F6E56', '#E1F5EE'],
          ['Silhouette',       insights.silhouette_score,                '#BA7517', '#FAEEDA'],
        ].map(([label, val, color, bg]) => (
          <div key={label} style={{...S.kpiCard, background:bg}}>
            <div style={S.kpiLabel}>{label}</div>
            <div style={{...S.kpiVal, color}}>{val}</div>
          </div>
        ))}
      </div>

      <div style={S.twoCol}>
        <div style={S.section}>
          <div style={S.secTitle}>Dataset statistics</div>
          <table style={S.table}>
            <tbody>
              {statRows.map(([label, val]) => (
                <tr key={label} style={S.tr}>
                  <td style={S.tdLabel}>{label}</td>
                  <td style={S.tdVal}>{val}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          {s.top_categories && (
            <div style={{...S.section, marginBottom:12}}>
              <div style={S.secTitle}>Revenue by category</div>
              <table style={S.table}>
                <thead>
                  <tr>
                    <th style={S.th}>Category</th>
                    <th style={{...S.th, textAlign:'right'}}>Revenue</th>
                    <th style={{...S.th, textAlign:'right'}}>Share</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(s.top_categories)
                    .sort((a,b) => b[1]-a[1])
                    .map(([cat, rev]) => (
                      <tr key={cat} style={S.tr}>
                        <td style={S.tdLabel}>{cat}</td>
                        <td style={{...S.tdVal, textAlign:'right'}}>${rev.toLocaleString()}</td>
                        <td style={{...S.tdVal, textAlign:'right', color:'#534AB7'}}>
                          {((rev/s.total_revenue)*100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}

          <div style={S.section}>
            <div style={S.secTitle}>Segment breakdown</div>
            <table style={S.table}>
              <thead>
                <tr>
                  <th style={S.th}>Segment</th>
                  <th style={{...S.th, textAlign:'right'}}>Size</th>
                  <th style={{...S.th, textAlign:'right'}}>Avg spend</th>
                  <th style={{...S.th, textAlign:'right'}}>Recency</th>
                </tr>
              </thead>
              <tbody>
                {clusters.map(c => (
                  <tr key={c.cluster_id} style={S.tr}>
                    <td style={S.tdLabel}>{c.label}</td>
                    <td style={{...S.tdVal, textAlign:'right'}}>{c.size} ({c.pct_of_customers}%)</td>
                    <td style={{...S.tdVal, textAlign:'right'}}>${c.avg_monetary.toLocaleString()}</td>
                    <td style={{...S.tdVal, textAlign:'right'}}>{c.avg_recency_days}d</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

const S = {
  topCards: { display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:10, marginBottom:'1.5rem' },
  kpiCard:  { borderRadius:10, padding:'1rem' },
  kpiLabel: { fontSize:11, color:'#5F5E5A', marginBottom:4 },
  kpiVal:   { fontSize:20, fontWeight:500 },
  twoCol:   { display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 },
  section:  { background:'white', border:'0.5px solid #E8E6DF', borderRadius:12, padding:'1.25rem' },
  secTitle: { fontSize:16, fontFamily:"'Instrument Serif', serif", color:'#1a1a1a', marginBottom:12 },
  table:    { width:'100%', borderCollapse:'collapse', fontSize:13 },
  th:       { textAlign:'left', fontWeight:500, fontSize:11, color:'#888780', padding:'5px 0', borderBottom:'0.5px solid #E8E6DF' },
  tr:       { borderBottom:'0.5px solid #E8E6DF' },
  tdLabel:  { padding:'8px 0', color:'#5F5E5A' },
  tdVal:    { padding:'8px 0', fontWeight:500, color:'#1a1a1a' },
}