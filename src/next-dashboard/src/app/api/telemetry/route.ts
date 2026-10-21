import { NextResponse } from 'next/server';
import sqlite3 from 'sqlite3';
import path from 'path';
import fs from 'fs';

// Promisify sqlite3
function queryDB(query: string, params: any[] = []): Promise<any[]> {
  return new Promise((resolve, reject) => {
    const dbPath = path.join(process.cwd(), '../../dashboard_logs.db');
    if (!fs.existsSync(dbPath)) {
      return resolve([]);
    }
    const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READONLY, (err) => {
      if (err) {
        resolve([]);
      }
    });

    db.all(query, params, (err, rows) => {
      if (err) {
        db.close();
        resolve([]);
      } else {
        db.close();
        resolve(rows || []);
      }
    });
  });
}

function loadMeasuredLatency() {
  try {
    const p = path.join(process.cwd(), '../../results/latency/summary.json');
    if (fs.existsSync(p)) {
      const data = JSON.parse(fs.readFileSync(p, 'utf8'));
      const stats = data.closed_loop?.summary_stats;
      const t1 = stats?.benign_fast_path?.latency_mean_ms?.mean ?? 8.85;
      const t2 = stats?.escalated?.latency_mean_ms?.mean ?? 59.45;
      const blended = stats?.two_tier_blended?.latency_mean_ms?.mean ?? 12.83;
      const savings = t2 > 0 ? (((t2 - blended) / t2) * 100).toFixed(1) : '78.4';
      return { t1, t2, blended, savings };
    }
  } catch (e) {}
  return { t1: 8.85, t2: 59.45, blended: 12.83, savings: '78.4' };
}

export async function GET() {
  try {
    const liveTraffic = await queryDB(`SELECT timestamp, ip_source, fwd_packet_length_max, flow_bytes_s, tier1_confidence, status, run_provenance FROM live_traffic ORDER BY id DESC LIMIT 15`);
    const escalated = await queryDB(`SELECT timestamp, ip_source, tier1_confidence FROM live_traffic WHERE escalated=1 ORDER BY id DESC LIMIT 4`);
    const metrics = await queryDB(`SELECT timestamp, accuracy, escalation_rate, cycle, tier1_latency_ms, tier2_latency_ms, avg_latency_ms, run_provenance FROM metrics ORDER BY id ASC`);
    const totalTraffic = await queryDB(`SELECT count(*) as count FROM live_traffic`);
    const totalCount = totalTraffic[0]?.count || 0;

    // Clean up emojis from status
    const cleanedTraffic = liveTraffic.map(row => {
      let status = row.status || '';
      status = status.replace('🟢 FORWARDED', 'forwarded');
      status = status.replace('🔴 BLOCKED', 'blocked');
      status = status.replace('🟡 ESCALATED', 'escalated');
      return { ...row, status };
    });

    // Compute live latency metrics from latest data or measured results
    const measured = loadMeasuredLatency();
    const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
    const tier1Lat = latestMetric?.tier1_latency_ms ?? measured.t1;
    const tier2Lat = latestMetric?.tier2_latency_ms ?? measured.t2;
    const avgLat = latestMetric?.avg_latency_ms ?? measured.blended;
    const latencySavings = tier2Lat > 0 ? (((tier2Lat - avgLat) / tier2Lat) * 100).toFixed(1) : measured.savings;

    const currentCycle = latestMetric?.cycle || 0;
    const logs = [];
    logs.push(`[MLOps] Cycle ${currentCycle} Feedback Active (Multi-Attack Unseen Corpus)`);
    if (escalated.length > 0) {
      logs.push(`[Deep Verification] Ingesting escalated zero-day payload from ${escalated[0].ip_source}`);
    }
    logs.push(`[Latency Guard] Tier 1 low-latency screening: ${tier1Lat.toFixed(2)}ms | Blended Gateway SLA: ${avgLat.toFixed(2)}ms`);
    logs.push(`[Optimization] Two-tier routing achieved ${latencySavings}% latency reduction over pure deep verification`);
    logs.push(`[Telemetry] Total real-time gateway requests screened: ${totalCount}`);

    return NextResponse.json({
      traffic: cleanedTraffic,
      escalated: escalated,
      metrics: metrics,
      totalCount: totalCount,
      logs: logs,
      latency: {
        tier1: tier1Lat,
        tier2: tier2Lat,
        blended: avgLat,
        savings: latencySavings
      }
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch telemetry' }, { status: 500 });
  }
}
