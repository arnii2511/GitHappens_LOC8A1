import { NextRequest, NextResponse } from 'next/server';
import type { SwipeResponse } from '@/lib/types';

/**
 * POST /api/matches/:matchId/skip
 *
 * Records a 'dislike' swipe.
 *
 * In production this would:
 *  - Insert into swipes(user_id, match_id, action='dislike')
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  const { matchId } = await params;

  const response: SwipeResponse = {
    success: true,
    action: 'dislike',
    matchId,
  };

  return NextResponse.json(response);
}
