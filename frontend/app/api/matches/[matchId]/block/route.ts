import { NextRequest, NextResponse } from 'next/server';
import type { SwipeResponse } from '@/lib/types';

/**
 * POST /api/matches/:matchId/block
 *
 * Records a 'block' swipe and updates match status.
 *
 * In production this would:
 *  - Insert into swipes(user_id, match_id, action='block')
 *  - Update matches.status = 'blocked' so it never reappears
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  const { matchId } = await params;

  const response: SwipeResponse = {
    success: true,
    action: 'block',
    matchId,
  };

  return NextResponse.json(response);
}
