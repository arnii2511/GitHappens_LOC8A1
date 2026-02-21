import { NextResponse } from 'next/server';

// POST /api/connections/:connectionId/respond - Accept or decline a connection
export async function POST(
  request: Request,
  { params }: { params: Promise<{ connectionId: string }> }
) {
  const { connectionId } = await params;
  const body = await request.json();
  const { action } = body as { action: 'accepted' | 'declined' };

  if (!['accepted', 'declined'].includes(action)) {
    return NextResponse.json(
      { success: false, error: 'Invalid action. Must be "accepted" or "declined".' },
      { status: 400 }
    );
  }

  return NextResponse.json({
    success: true,
    connectionId,
    status: action,
    respondedAt: new Date().toISOString(),
  });
}
