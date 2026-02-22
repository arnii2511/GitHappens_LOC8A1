import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

export async function POST(request: Request) {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();
  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = (await request.json()) as {
    action?: 'like' | 'dislike' | 'save' | 'block';
    finalScore?: number;
    matchId?: string;
  };

  const action = body.action;
  if (!action || !['like', 'dislike', 'save', 'block'].includes(action)) {
    return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
  }

  const finalScore = Number(body.finalScore ?? 0);
  const { error } = await supabase.from('analytics_swipe_events').insert({
    user_id: user.id,
    action,
    final_score: Number.isFinite(finalScore) ? finalScore : 0,
    match_id: body.matchId ?? null,
  });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json({ success: true });
}
