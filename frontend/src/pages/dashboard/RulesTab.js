import { useState } from 'react'

export default function RulesTab({ insights }) {
  const rules    = insights.association_rules || []
  const [sortBy, setSortBy] = useState('lift')
  const [search, setSearch] = useState('')

  const filtered = [...rules]
    .filter(r => [...r.antecedents, ...r.consequents].join(' ').toLowerCase().includes(search.toLowerCase()))
    .sort((a,b) => b[sortBy] - a[sortBy])

  return (
    <div>
      <div style={S.toolbar}>
        <div style={S.searchWrap}>
          <input placeholder="Search products..." value={search}
            onChange={e => setSearch(e.target.value)} />
        </div>
        <div style={S.sortGroup}>
          <span style={S.sortLabel}>Sort by</span>
          {['lift','confidence','support'].map(s => (
            <button key={s} onClick={() => setSortBy(s)} style={{
              ...S.sortBtn,
              background:  sortBy===s ? '#EEEDFE' : 'white',
              color:       sortBy===s ? '#534AB7' : '#5F5E5A',
              borderColor: sortBy===s ? '#AFA9EC' : '#D3D1C7',
            }}>
              {s}
            </button>
          ))}
        </div>
      </div>

      <div style={S.section}>
        <div style={S.tableCount}>{filtered.length} rules found</div>
        <table style={S.table}>
          <thead>
            <tr>
              <th style={S.th}>If customer buys</th>
              <th style={S.th}>They also buy</th>
              <th style={{...S.th, textAlign:'right'}}>Confidence</th>
              <th style={{...S.th, textAlign:'right'}}>Lift</th>
              <th style={{...S.th, textAlign:'right'}}>Support</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((rule, i) => (
              <tr key={i} style={S.tr}>
                <td style={S.td}>
                  <div style={S.pills}>
                    {rule.antecedents.map(a => <span key={a} style={S.pillPurple}>{a}</span>)}
                  </div>
                </td>
                <td style={S.td}>
                  <div style={S.pills}>
                    {rule.consequents.map(c => <span key={c} style={S.pillGreen}>{c}</span>)}
                  </div>
                </td>
                <td style={{...S.td, textAlign:'right', fontWeight:500}}>
                  {(rule.confidence*100).toFixed(0)}%
                </td>
                <td style={{...S.td, textAlign:'right', fontWeight:500, color:'#534AB7'}}>
                  {rule.lift.toFixed(2)}×
                </td>
                <td style={{...S.td, textAlign:'right', color:'#888780'}}>
                  {(rule.support*100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const S = {
  toolbar:    { display:'flex', alignItems:'center', gap:12, marginBottom:12 },
  searchWrap: { flex:1, maxWidth:280 },
  sortGroup:  { display:'flex', alignItems:'center', gap:6 },
  sortLabel:  { fontSize:12, color:'#888780' },
  sortBtn:    { padding:'5px 12px', borderRadius:8, border:'0.5px solid', fontSize:12, cursor:'pointer', transition:'all 0.15s' },
  section:    { background:'white', border:'0.5px solid #E8E6DF', borderRadius:12, padding:'1.25rem' },
  tableCount: { fontSize:12, color:'#888780', marginBottom:12 },
  table:      { width:'100%', borderCollapse:'collapse', fontSize:13 },
  th:         { textAlign:'left', fontWeight:500, fontSize:12, color:'#888780', padding:'6px 0', borderBottom:'0.5px solid #E8E6DF' },
  tr:         { borderBottom:'0.5px solid #E8E6DF' },
  td:         { padding:'10px 0', verticalAlign:'middle' },
  pills:      { display:'flex', gap:4, flexWrap:'wrap' },
  pillPurple: { background:'#EEEDFE', color:'#534AB7', padding:'2px 8px', borderRadius:999, fontSize:11, fontWeight:500 },
  pillGreen:  { background:'#E1F5EE', color:'#0F6E56', padding:'2px 8px', borderRadius:999, fontSize:11, fontWeight:500 },
}