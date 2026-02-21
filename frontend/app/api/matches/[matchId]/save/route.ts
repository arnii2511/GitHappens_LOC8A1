import { NextRequest, NextResponse } from 'next/server';
import type { SwipeResponse } from '@/lib/types';

/**
 * POST /api/matches/:matchId/save
 *
 * Records a 'save' swipe (bookmark for later).
 *
 * In production this would:
 *  - Insert into swipes(user_id, match_id, action='save')
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  const { matchId } = await params;

  const response: SwipeResponse = {
    success: true,
    action: 'save',
    matchId,
  };

  return NextResponse.json(response);
}
