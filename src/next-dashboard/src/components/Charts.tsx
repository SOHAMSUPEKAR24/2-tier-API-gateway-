"use client";

import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, 
  PieChart, Pie, Cell, BarChart, Bar, Legend 
} from 'recharts';

export function SystemVitalsCharts({ data }: { data: any[] }) {
  if (!data || data.length === 0) return <p style={{color: '#8a8d91', fontStyle: 'italic'}}>awaiting telemetry data...</p>;

  const chartData = data.slice(-50);

  return (
    <div className="grid-1-1">
      <div>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px'}}>
          <p style={{fontSize: '0.85em', color: '#66fcf1', margin: 0, fontWeight: 600}}>Tier 1 Edge Accuracy (%)</p>
          <span style={{fontSize: '0.75em', color: '#8a8d91'}}>Adapting: 52% ➔ 99%</span>
        </div>
        <div style={{ width: '100%', height: 200 }}>
          <ResponsiveContainer>
            <LineChart data={chartData}>
              <XAxis dataKey="timestamp" hide />
              <YAxis domain={[50.0, 100.0]} tick={{fontSize: 10, fill: '#8a8d91'}} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{backgroundColor: 'rgba(0,0,0,0.85)', border: '1px solid rgba(102, 252, 241, 0.4)', borderRadius: '8px', color: '#fff', fontSize: '12px'}} />
              <Line type="monotone" dataKey="accuracy" stroke="#66fcf1" strokeWidth={3} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px'}}>
          <p style={{fontSize: '0.85em', color: '#ff4b4b', margin: 0, fontWeight: 600}}>Tier 2 Sandbox Escalation (%)</p>
          <span style={{fontSize: '0.75em', color: '#8a8d91'}}>Narrowing: 16% ➔ &lt;1%</span>
        </div>
        <div style={{ width: '100%', height: 200 }}>
          <ResponsiveContainer>
            <LineChart data={chartData}>
              <XAxis dataKey="timestamp" hide />
              <YAxis domain={[0.0, 20.0]} tick={{fontSize: 10, fill: '#8a8d91'}} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{backgroundColor: 'rgba(0,0,0,0.85)', border: '1px solid rgba(255, 75, 75, 0.4)', borderRadius: '8px', color: '#fff', fontSize: '12px'}} />
              <Line type="monotone" dataKey="escalation_rate" stroke="#ff4b4b" strokeWidth={3} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export function LatencyComparisonChart({ latency }: { latency?: { tier1: number; tier2: number; blended: number; savings: string } }) {
  const t1 = latency?.tier1 ?? 8.85;
  const blended = latency?.blended ?? 12.83;
  const t2 = latency?.tier2 ?? 59.45;

  const data = [
    { name: 'Tier 1 Edge', latency: Number(t1.toFixed(2)), fill: '#66fcf1' },
    { name: 'Two-Tier Gateway', latency: Number(blended.toFixed(2)), fill: '#f2a900' },
    { name: 'Monolithic Deep Verif', latency: Number(t2.toFixed(2)), fill: '#ff4b4b' },
  ];

  return (
    <div style={{ width: '100%', height: 220 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 60, bottom: 5 }}>
          <XAxis type="number" unit=" ms" tick={{ fill: '#8a8d91', fontSize: 11 }} axisLine={false} />
          <YAxis type="category" dataKey="name" tick={{ fill: '#c5c6c7', fontSize: 11 }} axisLine={false} width={100} />
          <Tooltip 
            contentStyle={{ backgroundColor: 'rgba(0,0,0,0.85)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', color: '#fff' }}
            formatter={(value: any) => [`${value} ms`, 'Processing Latency']}
          />
          <Bar dataKey="latency" radius={[0, 6, 6, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ textAlign: 'center', fontSize: '0.8em', color: '#66fcf1', marginTop: '4px', fontWeight: 600 }}>
        ⚡ {latency?.savings || '78.4'}% Latency Overhead Reduction vs Monolithic Verification
      </div>
    </div>
  );
}

export function RadarFeatureChart() {
  const radarData = [
    { subject: 'Fwd Pkt Len Mean', A: 35, fullMark: 40 },
    { subject: 'Bwd Pkt Len Max', A: 25, fullMark: 40 },
    { subject: 'Flow Bytes/s', A: 15, fullMark: 40 },
    { subject: 'Flow Pkts/s', A: 10, fullMark: 40 },
    { subject: 'Fwd IAT Mean', A: 10, fullMark: 40 },
    { subject: 'ACK Flag Count', A: 5, fullMark: 40 },
  ];

  return (
    <div style={{ width: '100%', height: 250 }}>
      <ResponsiveContainer>
        <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" />
          <PolarAngleAxis dataKey="subject" tick={{fill: '#c5c6c7', fontSize: 10}} />
          <PolarRadiusAxis angle={30} domain={[0, 40]} tick={false} axisLine={false} />
          <Radar name="Importance" dataKey="A" stroke="#f2a900" fill="#f2a900" fillOpacity={0.4} strokeWidth={2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrafficDistributionChart({ traffic }: { traffic: any[] }) {
  if (!traffic || traffic.length === 0) return null;

  let forwarded = 0;
  let blocked = 0;
  let escalated = 0;

  traffic.forEach(t => {
    if (t.status === 'forwarded') forwarded++;
    if (t.status === 'blocked') blocked++;
    if (t.status === 'escalated') escalated++;
  });

  const data = [
    { name: 'Forwarded', value: forwarded || 1, color: '#66fcf1' },
    { name: 'Blocked', value: blocked || 1, color: '#ff4b4b' },
    { name: 'Escalated', value: escalated || 1, color: '#f2a900' },
  ];

  return (
    <div style={{ width: '100%', height: 210, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={70}
            paddingAngle={5}
            dataKey="value"
            stroke="none"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip contentStyle={{backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#fff', fontSize: '12px'}} />
        </PieChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: '15px', fontSize: '0.8em', color: '#8a8d91', marginTop: '-10px' }}>
        <span style={{color: '#66fcf1'}}>■ Forwarded ({forwarded})</span>
        <span style={{color: '#ff4b4b'}}>■ Blocked ({blocked})</span>
        <span style={{color: '#f2a900'}}>■ Escalated ({escalated})</span>
      </div>
    </div>
  );
}
