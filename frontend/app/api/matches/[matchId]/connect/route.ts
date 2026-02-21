import { NextRequest, NextResponse } from 'next/server';
import type { SwipeResponse } from '@/lib/types';

/**
 * POST /api/matches/:matchId/connect
 *
 * Records a 'like' swipe and optionally generates an outreach draft.
 *
 * In production this would:
 *  - Insert into swipes(user_id, match_id, action='like')
 *  - Generate an outreach_drafts row with channel='email', personalized message
 *  - Return the outreach_draft_id
 */
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ matchId: string }> }
) {
  const { matchId } = await params;

  // Simulate outreach draft generation
  const outreachDraftId = `draft-${matchId}-${Date.now()}`;

  const response: SwipeResponse = {
    success: true,
    action: 'like',
    matchId,
    outreachDraftId,
  };

  return NextResponse.json(response);
}
