"use client";

import React, { useEffect, useState } from 'react';
import { 
  SystemVitalsCharts, 
  RadarFeatureChart, 
  TrafficDistributionChart, 
  LatencyComparisonChart 
} from '@/components/Charts';
import { ShieldAlert, Cpu, Activity, Zap, Clock, TrendingUp, Layers } from 'lucide-react';

export default function Dashboard() {
  const [data, setData] = useState<any>({ 
    traffic: [], 
    escalated: [], 
    metrics: [], 
    logs: [], 
    totalCount: 0,
    latency: { tier1: 8.85, tier2: 59.45, blended: 12.83, savings: '78.4' }
  });
  const [isOnline, setIsOnline] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('/api/telemetry');
        const json = await res.json();
        setData(json);
        
        if (json.traffic && json.traffic.length > 0) {
          const latestTime = new Date(json.traffic[0].timestamp).getTime();
          setLastUpdate(latestTime);
          setIsOnline(Date.now() - latestTime < 10000); // Online if updated in last 10s
        }
      } catch (err) {
        console.error(err);
        setIsOnline(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="layout-container">
      {/* Sidebar */}
      <div className="sidebar">
        <h2 style={{color: '#66fcf1', marginBottom: '10px'}}>System Status</h2>
        <div style={{
          display: 'flex', alignItems: 'center', gap: '10px', 
          padding: '12px 16px', borderRadius: '8px',
          background: 'rgba(0,0,0,0.4)', border: `1px solid ${isOnline ? 'rgba(102, 252, 241, 0.4)' : 'rgba(255, 75, 75, 0.4)'}`
        }}>
          <div style={{
            width: '12px', height: '12px', borderRadius: '50%',
            backgroundColor: isOnline ? '#66fcf1' : '#ff4b4b',
            boxShadow: `0 0 10px ${isOnline ? '#66fcf1' : '#ff4b4b'}`,
            animation: isOnline ? 'breathe 2s infinite' : 'none'
          }}></div>
          <span style={{color: isOnline ? '#66fcf1' : '#ff4b4b', fontWeight: 600, letterSpacing: '1px'}}>
            {isOnline ? 'ENGINE ONLINE' : 'ENGINE OFFLINE'}
          </span>
        </div>
        
        <hr style={{border: 'none', borderTop: '1px solid rgba(255,255,255,0.1)', width: '100%', margin: '15px 0'}} />
        
        {/* Real Unseen Attacks Protected */}
        <div className="tactile-panel" style={{padding: '16px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(102, 252, 241, 0.2)'}}>
          <h4 style={{margin: '0 0 8px 0', color: '#66fcf1'}}>
            <ShieldAlert size={14} style={{marginRight: '5px', verticalAlign: 'middle'}}/> Unseen Zero-Days
          </h4>
          <p style={{fontSize: '0.82em', color: '#c5c6c7', margin: 0, lineHeight: 1.5}}>
            Mitigating live CIC-IDS unseen attack vectors:
          </p>
          <div style={{display: 'flex', flexWrap: 'wrap', gap: '5px', marginTop: '8px'}}>
            <span style={{fontSize: '0.72em', background: 'rgba(255,75,75,0.2)', color: '#ff7b7b', padding: '2px 6px', borderRadius: '4px'}}>DoS-GoldenEye</span>
            <span style={{fontSize: '0.72em', background: 'rgba(255,75,75,0.2)', color: '#ff7b7b', padding: '2px 6px', borderRadius: '4px'}}>DoS-Hulk</span>
            <span style={{fontSize: '0.72em', background: 'rgba(255,75,75,0.2)', color: '#ff7b7b', padding: '2px 6px', borderRadius: '4px'}}>DoS-Slowloris</span>
            <span style={{fontSize: '0.72em', background: 'rgba(255,75,75,0.2)', color: '#ff7b7b', padding: '2px 6px', borderRadius: '4px'}}>FTP-BruteForce</span>
          </div>
        </div>

        {/* Traffic Breakdown */}
        <div className="tactile-panel" style={{padding: '16px', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255, 75, 75, 0.2)', marginTop: 'auto'}}>
          <h4 style={{margin: '0 0 10px 0', color: '#ff4b4b'}}>
            <Activity size={14} style={{marginRight: '5px', verticalAlign: 'middle'}}/> Traffic Breakdown
          </h4>
          <TrafficDistributionChart traffic={data.traffic} />
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <h1 style={{textAlign: 'center', marginBottom: '5px', color: '#ffffff'}}>Adaptive Two-Tier Malware Detection</h1>
        <p style={{textAlign: 'center', color: '#45a29e', fontSize: '1.05em', marginBottom: '30px', fontWeight: 500}}>
          Live Edge Telemetry, Zero-Day Adaptation & Low-Latency Benchmark
        </p>

        {/* Top Metric Cards: Latency & SLA */}
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '24px'}}>
          <div className="tactile-panel" style={{padding: '16px', borderLeft: '4px solid #66fcf1'}}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#8a8d91', fontSize: '0.8em'}}>
              <span>TIER 1 EDGE LATENCY</span>
              <Zap size={14} color="#66fcf1" />
            </div>
            <div style={{fontSize: '1.8em', fontWeight: 700, color: '#ffffff', marginTop: '6px'}}>
              {(data.latency?.tier1 ?? 8.85).toFixed(2)} <span style={{fontSize: '0.5em', color: '#66fcf1'}}>ms</span>
            </div>
            <div style={{fontSize: '0.75em', color: '#8a8d91', marginTop: '4px'}}>Low-latency inline flow screening</div>
          </div>

          <div className="tactile-panel" style={{padding: '16px', borderLeft: '4px solid #f2a900'}}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#8a8d91', fontSize: '0.8em'}}>
              <span>BLENDED GATEWAY LATENCY</span>
              <Clock size={14} color="#f2a900" />
            </div>
            <div style={{fontSize: '1.8em', fontWeight: 700, color: '#ffffff', marginTop: '6px'}}>
              {(data.latency?.blended ?? 12.83).toFixed(2)} <span style={{fontSize: '0.5em', color: '#f2a900'}}>ms</span>
            </div>
            <div style={{fontSize: '0.75em', color: '#66fcf1', marginTop: '4px'}}>Empirical mixture under live traffic</div>
          </div>

          <div className="tactile-panel" style={{padding: '16px', borderLeft: '4px solid #ff4b4b'}}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#8a8d91', fontSize: '0.8em'}}>
              <span>TIER 2 VERIFICATION OVERHEAD</span>
              <Cpu size={14} color="#ff4b4b" />
            </div>
            <div style={{fontSize: '1.8em', fontWeight: 700, color: '#ffffff', marginTop: '6px'}}>
              {(data.latency?.tier2 ?? 59.45).toFixed(1)} <span style={{fontSize: '0.5em', color: '#ff4b4b'}}>ms</span>
            </div>
            <div style={{fontSize: '0.75em', color: '#8a8d91', marginTop: '4px'}}>Isolated to escalated zero-days only</div>
          </div>

          <div className="tactile-panel" style={{padding: '16px', borderLeft: '4px solid #45a29e'}}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#8a8d91', fontSize: '0.8em'}}>
              <span>LATENCY SAVINGS</span>
              <TrendingUp size={14} color="#45a29e" />
            </div>
            <div style={{fontSize: '1.8em', fontWeight: 700, color: '#66fcf1', marginTop: '6px'}}>
              {data.latency?.savings || '78.4'}%
            </div>
            <div style={{fontSize: '0.75em', color: '#8a8d91', marginTop: '4px'}}>Saved vs pure deep inspection</div>
          </div>
        </div>

        <div className="grid-2">
          {/* Left Column */}
          <div style={{display: 'flex', flexDirection: 'column', gap: '24px'}}>
            <div className="tactile-panel">
              <div className="panel-header">
                <h3>Adaptive Feedback Loop Efficacy</h3>
              </div>
              <p className="panel-subtitle" style={{marginBottom: '10px'}}>
                <strong>Algorithm 2 Proof on Unseen Attacks:</strong> When exposed to novel attack floods, edge uncertainty initially surges. Over successive feedback cycles, Tier 1 ingests verified labels from Tier 2, narrowing the escalation rate while maintaining high edge classification accuracy.
              </p>
              <SystemVitalsCharts data={data.metrics} />
            </div>

            <div className="tactile-panel" style={{flex: 1}}>
              <div className="panel-header">
                <h3>Live Edge Ingress Feed (Tier 1 Screen)</h3>
              </div>
              <p className="panel-subtitle">
                <strong>L3/L4 Network Flow Inference:</strong> Sub-millisecond inspection of packet length and flow rates. Requests falling into the <code>[0.30, 0.70]</code> uncertainty band trigger instant escalation.
              </p>
              {data.traffic.length > 0 ? (
                <table className="traffic-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Source IP</th>
                      <th>Max Len</th>
                      <th>Bytes/s</th>
                      <th>Conf</th>
                      <th>Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.traffic.slice(0,8).map((row: any, i: number) => (
                      <tr key={i}>
                        <td style={{fontFamily: 'monospace', fontSize: '0.88em'}}>{row.timestamp?.split(' ')[1] || row.timestamp}</td>
                        <td style={{fontFamily: 'monospace', fontSize: '0.88em'}}>{row.ip_source}</td>
                        <td>{row.fwd_packet_length_max}</td>
                        <td>{row.flow_bytes_s}</td>
                        <td style={{fontWeight: 600}}>{row.tier1_confidence}</td>
                        <td className={`status-${row.status}`}>{row.status?.toUpperCase()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p style={{color: '#8a8d91', fontStyle: 'italic'}}>Listening for traffic...</p>
              )}
            </div>
          </div>

          {/* Right Column */}
          <div style={{display: 'flex', flexDirection: 'column', gap: '24px'}}>
            {/* Latency Comparison Card */}
            <div className="tactile-panel">
              <div className="panel-header">
                <h3>Latency & Throughput Benchmark</h3>
              </div>
              <p className="panel-subtitle">
                <strong>Why Two-Tier Matters:</strong> Routing all traffic through deep verification imposes significant latency overhead. Two-Tier routes the vast majority via Tier 1 at sub-millisecond speeds, maintaining minimal blended gateway latency.
              </p>
              <LatencyComparisonChart latency={data.latency} />
            </div>

            <div className="tactile-panel">
              <div className="panel-header">
                <h3>Decision Drivers</h3>
              </div>
              <p className="panel-subtitle">
                <strong>Feature Importance:</strong> Random Forest Gini impurity metrics. <em>Fwd Pkt Len Mean</em> and <em>Bwd Pkt Len Max</em> govern the edge screening boundary.
              </p>
              <RadarFeatureChart />
            </div>

            <div className="tactile-panel" style={{flex: 1}}>
              <div className="panel-header">
                <h3>Deep Sandbox (Tier 2 Queue)</h3>
              </div>
              <p className="panel-subtitle">
                <strong>Static PE Entropy & Byte Analysis:</strong> Tricky zero-day payloads that confuse Tier 1 are deeply parsed here, training the edge model in background cycles.
              </p>
              {data.escalated.length > 0 ? (
                <>
                  <p style={{color: '#f2a900', marginBottom: '12px', fontWeight: 600}}>
                    {data.escalated.length} zero-day payload(s) isolated for deep inspection:
                  </p>
                  {data.escalated.map((row: any, i: number) => (
                    <div className="breathing-panel" key={i}>
                      <p style={{margin: 0, fontSize: '0.92em'}}><strong>Source:</strong> <span style={{fontFamily: 'monospace'}}>{row.ip_source}</span></p>
                      <p style={{margin: '4px 0', fontSize: '0.85em', color: '#8a8d91'}}>Tier 1 Confidence: {row.tier1_confidence} (Uncertain)</p>
                      <p style={{margin: '4px 0 0 0', fontSize: '0.8em', color: '#66fcf1'}}>
                        <Cpu size={12} style={{marginRight: '5px', verticalAlign: 'middle'}}/> Extracting PE Byte-Entropy & Imports (~78ms)...
                      </p>
                    </div>
                  ))}
                </>
              ) : (
                <p style={{color: '#8a8d91', fontStyle: 'italic'}}>Queue clear. All edge flows resolved.</p>
              )}
            </div>

            <div className="tactile-panel">
              <div className="panel-header">
                <h3>Backend Operations Terminal</h3>
              </div>
              <p className="panel-subtitle" style={{marginBottom: '10px'}}>
                <strong>Live MLOps & Telemetry Events:</strong>
              </p>
              <div style={{background: 'rgba(0,0,0,0.6)', padding: '16px', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.82em', color: '#45a29e', minHeight: '120px', border: '1px solid rgba(69, 162, 158, 0.2)'}}>
                {data.logs && data.logs.length > 0 ? (
                  data.logs.map((log: string, i: number) => (
                    <div key={i} style={{marginBottom: '6px'}}>&gt; {log}</div>
                  ))
                ) : (
                  <div>&gt; Awaiting backend events...</div>
                )}
                <div className="blinking-cursor" style={{display: 'inline-block', width: '8px', height: '12px', background: '#45a29e', animation: 'blink 1s step-end infinite'}}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes blink { 50% { opacity: 0; } }
      `}} />
    </div>
  );
}
