import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

type SwipeEventRow = {
  action: 'like' | 'dislike' | 'save' | 'block';
  final_score: number | null;
  created_at: string;
};

export async function GET() {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();
  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data, error } = await supabase
    .from('analytics_swipe_events')
    .select('action, final_score, created_at')
    .eq('user_id', user.id)
    .order('created_at', { ascending: true });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  const events = (data as SwipeEventRow[]) ?? [];
  const counts = { like: 0, dislike: 0, save: 0, block: 0 };
  let sum = 0;
  const timeline: Array<{ label: string; matches: number; connected: number; skipped: number }> = [];

  let i = 0;
  for (const ev of events) {
    counts[ev.action] += 1;
    sum += Number(ev.final_score ?? 0);
    i += 1;
    if (i % 5 === 0 || i === events.length) {
      timeline.push({
        label: `#${i}`,
        matches: i,
        connected: counts.like,
        skipped: counts.dislike,
      });
    }
  }
  if (timeline.length === 0) {
    timeline.push({ label: 'Start', matches: 0, connected: 0, skipped: 0 });
  }

  const scores = events.map((e) => Number(e.final_score ?? 0));
  const scoreDistribution = [
    { range: '90-100', count: scores.filter((s) => s >= 90).length, fill: '#10b981' },
    { range: '80-89', count: scores.filter((s) => s >= 80 && s < 90).length, fill: '#06b6d4' },
    { range: '70-79', count: scores.filter((s) => s >= 70 && s < 80).length, fill: '#f59e0b' },
    { range: '<70', count: scores.filter((s) => s < 70).length, fill: '#ef4444' },
  ];

  return NextResponse.json({
    totalMatches: events.length,
    connected: counts.like,
    skipped: counts.dislike,
    saved: counts.save,
    blocked: counts.block,
    avgScore: events.length > 0 ? sum / events.length : 0,
    timeline,
    scoreDistribution,
  });
}
