import { NextRequest, NextResponse } from 'next/server';
import type { SwipeResponse } from '@/lib/types';

const ML_API_BASE_URL =
  process.env.ML_API_BASE_URL ||
  process.env.NEXT_PUBLIC_ML_API_BASE_URL ||
  'http://127.0.0.1:8000';

type SwipeBridgeBody = {
  buyerId?: string;
  buyer_id?: string;
  exporterId?: string;
  exporter_id?: string;
};

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
  const body = (await request.json().catch(() => ({}))) as SwipeBridgeBody;

  const buyerId = String(body.buyerId || body.buyer_id || '').trim();
  const exporterId = String(body.exporterId || body.exporter_id || matchId).trim();

  if (!buyerId || !exporterId) {
    return NextResponse.json(
      { success: false, error: 'buyerId and exporterId are required for swipe forwarding.' },
      { status: 400 }
    );
  }

  const upstream = await fetch(`${ML_API_BASE_URL}/swipe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      buyer_id: buyerId,
      exporter_id: exporterId,
      action: 'right',
    }),
    cache: 'no-store',
  });

  if (!upstream.ok) {
    const details = await upstream.text();
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to persist swipe in ML backend.',
        details: details || upstream.statusText,
      },
      { status: 502 }
    );
  }

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
