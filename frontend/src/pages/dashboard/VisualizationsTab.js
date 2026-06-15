import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts'

const COLORS = ['#7F77DD','#1D9E75','#D85A30','#BA7517','#185FA5']

export default function VisualizationsTab({ insights }) {
  const trend    = insights.trend_data || {}
  const monthly  = trend.monthly_revenue || []
  const topProds = trend.top_products || []
  const clusters = insights.cluster_profiles || []
  const segData  = clusters.map(c => ({ name: c.label, value: c.size }))

  return (
    <div>
      <div style={S.twoCol}>
        <div style={S.section}>
          <div style={S.secTitle}>Monthly revenue</div>
          <div style={S.secSub}>Full year trend</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={monthly} margin={{top:8, right:8, bottom:0, left:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" vertical={false} />
              <XAxis dataKey="month" tick={{fontSize:10, fill:'#888780'}}
                tickFormatter={m => m.slice(5)} axisLine={false} tickLine={false} />
              <YAxis tick={{fontSize:10, fill:'#888780'}} axisLine={false} tickLine={false}
                tickFormatter={v => `$${(v/1000).toFixed(0)}K`} />
              <Tooltip
                contentStyle={{border:'0.5px solid #E8E6DF', borderRadius:8, fontSize:12}}
                formatter={v => [`$${v.toLocaleString()}`, 'Revenue']}
              />
              <Bar dataKey="total_revenue" fill="#7F77DD" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={S.section}>
          <div style={S.secTitle}>Customer segments</div>
          <div style={S.secSub}>By size</div>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={segData} cx="50%" cy="50%" outerRadius={80} innerRadius={40}
                dataKey="value" label={({name, percent}) => `${name} ${(percent*100).toFixed(0)}%`}
                labelLine={false} fontSize={11}
              >
                {segData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{border:'0.5px solid #E8E6DF', borderRadius:8, fontSize:12}} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {topProds.length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>Top products by revenue</div>
          <div style={S.secSub}>Ranked by total sales value</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={topProds} layout="vertical"
              margin={{top:8, right:40, bottom:0, left:90}}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" horizontal={false} />
              <XAxis type="number" tick={{fontSize:10, fill:'#888780'}} axisLine={false} tickLine={false}
                tickFormatter={v => `$${(v/1000).toFixed(0)}K`} />
              <YAxis type="category" dataKey="product_name" tick={{fontSize:11, fill:'#5F5E5A'}}
                width={85} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{border:'0.5px solid #E8E6DF', borderRadius:8, fontSize:12}}
                formatter={v => [`$${v.toLocaleString()}`, 'Revenue']} />
              <Bar dataKey="total_revenue" fill="#1D9E75" radius={[0,4,4,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {Object.keys(trend.category_monthly || {}).length > 0 && (
        <div style={S.section}>
          <div style={S.secTitle}>Category trends</div>
          <div style={S.secSub}>Monthly revenue by product category</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart
              data={monthly.map(m => {
                const row = { month: m.month }
                Object.entries(trend.category_monthly || {}).forEach(([cat, data]) => {
                  const match = data.find(d => d.month === m.month)
                  row[cat] = match?.revenue || 0
                })
                return row
              })}
              margin={{top:8, right:8, bottom:0, left:0}}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#F1EFE8" vertical={false} />
              <XAxis dataKey="month" tick={{fontSize:10, fill:'#888780'}}
                tickFormatter={m => m.slice(5)} axisLine={false} tickLine={false} />
              <YAxis tick={{fontSize:10, fill:'#888780'}} axisLine={false} tickLine={false}
                tickFormatter={v => `$${(v/1000).toFixed(0)}K`} />
              <Tooltip contentStyle={{border:'0.5px solid #E8E6DF', borderRadius:8, fontSize:12}}
                formatter={v => [`$${v.toLocaleString()}`, '']} />
              {Object.keys(trend.category_monthly || {}).map((cat, i) => (
                <Line key={cat} type="monotone" dataKey={cat}
                  stroke={COLORS[i % COLORS.length]} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

const S = {
  twoCol:   { display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginBottom:12 },
  section:  { background:'white', border:'0.5px solid #E8E6DF', borderRadius:12, padding:'1.25rem', marginBottom:12 },
  secTitle: { fontSize:15, fontFamily:"'Instrument Serif', serif", color:'#1a1a1a', marginBottom:2 },
  secSub:   { fontSize:12, color:'#888780', marginBottom:12 },
}