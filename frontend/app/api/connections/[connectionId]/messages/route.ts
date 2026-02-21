import { NextResponse } from 'next/server';
import { mockMessages } from '@/lib/mock-data';

// GET /api/connections/:connectionId/messages - Get all messages for a connection
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ connectionId: string }> }
) {
  const { connectionId } = await params;
  const messages = mockMessages.filter((m) => m.connectionId === connectionId);

  return NextResponse.json({
    messages,
    total: messages.length,
  });
}

// POST /api/connections/:connectionId/messages - Send a message
export async function POST(
  request: Request,
  { params }: { params: Promise<{ connectionId: string }> }
) {
  const { connectionId } = await params;
  const body = await request.json();
  const { content, senderId, senderName } = body;

  const newMessage = {
    id: `msg-${Date.now()}`,
    connectionId,
    senderId,
    senderName,
    senderRole: 'exporter' as const,
    content,
    createdAt: new Date().toISOString(),
    read: false,
  };

  return NextResponse.json({
    success: true,
    message: newMessage,
  });
}
