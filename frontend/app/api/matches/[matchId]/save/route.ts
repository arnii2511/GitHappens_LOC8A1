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
  const body = (await request.json().catch(() => ({}))) as SwipeBridgeBody;

  const buyerId = String(body.buyerId || body.buyer_id || '').trim();
  const exporterId = String(body.exporterId || body.exporter_id || matchId).trim();

  if (!buyerId || !exporterId) {
    return NextResponse.json(
      { success: false, error: 'buyerId and exporterId are required for swipe forwarding.' },
      { status: 400 }
    );
  }

  // Backend currently supports only left/right actions.
  const upstream = await fetch(`${ML_API_BASE_URL}/swipe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      buyer_id: buyerId,
      exporter_id: exporterId,
      action: 'left',
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

  const response: SwipeResponse = {
    success: true,
    action: 'save',
    matchId,
  };

  return NextResponse.json(response);
}
