import { useState } from 'react'

export default function RulesTab({ insights }) {
  const rules = insights.association_rules || []
  const [sortBy, setSortBy] = useState('lift')
  const [search, setSearch] = useState('')

  const filtered = [...rules]
    .filter(r => {
      const text = [...r.antecedents, ...r.consequents].join(' ').toLowerCase()
      return text.includes(search.toLowerCase())
    })
    .sort((a, b) => b[sortBy] - a[sortBy])

  return (
    <div>
      <div style={styles.toolbar}>
        <input
          placeholder="Search products..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={styles.search}
        />
        <div style={styles.sortGroup}>
          <span style={styles.sortLabel}>Sort by:</span>
          {['lift','confidence','support'].map(s => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              style={{
                ...styles.sortBtn,
                background: sortBy === s ? '#EEEDFE' : 'transparent',
                color:      sortBy === s ? '#534AB7' : 'var(--color-text-secondary)',
                borderColor: sortBy === s ? '#AFA9EC' : 'var(--color-border-tertiary)',
              }}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      <div style={styles.section}>
        <div style={styles.countRow}>
          <span style={styles.count}>{filtered.length} rules</span>
        </div>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>If customer buys</th>
              <th style={styles.th}>They also buy</th>
              <th style={{...styles.th, textAlign:'right'}}>Confidence</th>
              <th style={{...styles.th, textAlign:'right'}}>Lift</th>
              <th style={{...styles.th, textAlign:'right'}}>Support</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((rule, i) => (
              <tr key={i} style={styles.tr}>
                <td style={styles.td}>
                  <div style={styles.pills}>
                    {rule.antecedents.map(a => (
                      <span key={a} style={styles.pill}>{a}</span>
                    ))}
                  </div>
                </td>
                <td style={styles.td}>
                  <div style={styles.pills}>
                    {rule.consequents.map(c => (
                      <span key={c} style={{...styles.pill, background:'#E1F5EE', color:'#0F6E56'}}>{c}</span>
                    ))}
                  </div>
                </td>
                <td style={{...styles.td, textAlign:'right', fontWeight:500}}>
                  {(rule.confidence * 100).toFixed(0)}%
                </td>
                <td style={{...styles.td, textAlign:'right', fontWeight:500, color:'#534AB7'}}>
                  {rule.lift.toFixed(2)}x
                </td>
                <td style={{...styles.td, textAlign:'right', color:'var(--color-text-secondary)'}}>
                  {(rule.support * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const styles = {
  toolbar:    { display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'1rem', gap:12 },
  search:     { flex:1, maxWidth:280 },
  sortGroup:  { display:'flex', alignItems:'center', gap:6 },
  sortLabel:  { fontSize:12, color:'var(--color-text-secondary)' },
  sortBtn:    { padding:'4px 10px', borderRadius:'var(--border-radius-md)', border:'0.5px solid', fontSize:12, cursor:'pointer' },
  section:    { background:'var(--color-background-primary)', border:'0.5px solid var(--color-border-tertiary)', borderRadius:'var(--border-radius-lg)', padding:'1.25rem' },
  countRow:   { marginBottom:12 },
  count:      { fontSize:12, color:'var(--color-text-tertiary)' },
  table:      { width:'100%', borderCollapse:'collapse', fontSize:13 },
  th:         { textAlign:'left', fontWeight:500, fontSize:12, color:'var(--color-text-secondary)', padding:'6px 0', borderBottom:'0.5px solid var(--color-border-tertiary)' },
  tr:         { borderBottom:'0.5px solid var(--color-border-tertiary)' },
  td:         { padding:'10px 0', verticalAlign:'middle' },
  pills:      { display:'flex', gap:4, flexWrap:'wrap' },
  pill:       { background:'#EEEDFE', color:'#534AB7', padding:'2px 8px', borderRadius:999, fontSize:11, fontWeight:500 },
}