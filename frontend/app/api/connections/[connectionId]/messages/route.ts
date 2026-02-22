import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

type ConnectionRow = {
  id: string;
  exporter_user_id: string;
  importer_user_id: string | null;
};

type MessageRow = {
  id: string;
  connection_id: string;
  sender_user_id: string;
  sender_name: string;
  sender_role: 'exporter' | 'importer';
  content: string;
  read: boolean;
  created_at: string;
};

function toMessageDTO(row: MessageRow) {
  return {
    id: row.id,
    connectionId: row.connection_id,
    senderId: row.sender_user_id,
    senderName: row.sender_name,
    senderRole: row.sender_role,
    content: row.content,
    read: row.read,
    createdAt: row.created_at,
  };
}

// GET /api/connections/:connectionId/messages - Get all messages for a connection
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ connectionId: string }> }
) {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();
  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { connectionId } = await params;
  const { data: conn, error: connErr } = await supabase
    .from('connections')
    .select('id, exporter_user_id, importer_user_id')
    .eq('id', connectionId)
    .maybeSingle();
  if (connErr) return NextResponse.json({ error: connErr.message }, { status: 500 });
  if (!conn) return NextResponse.json({ error: 'Connection not found' }, { status: 404 });

  const c = conn as ConnectionRow;
  if (c.exporter_user_id !== user.id && c.importer_user_id !== user.id) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  const { data, error } = await supabase
    .from('connection_messages')
    .select('*')
    .eq('connection_id', connectionId)
    .order('created_at', { ascending: true });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  const messages = ((data as MessageRow[]) ?? []).map(toMessageDTO);

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
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();
  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { connectionId } = await params;
  const body = await request.json();
  const { content } = body as { content?: string };
  if (!content || !content.trim()) {
    return NextResponse.json({ error: 'content is required' }, { status: 400 });
  }

  const { data: conn, error: connErr } = await supabase
    .from('connections')
    .select('id, exporter_user_id, importer_user_id')
    .eq('id', connectionId)
    .maybeSingle();
  if (connErr) return NextResponse.json({ error: connErr.message }, { status: 500 });
  if (!conn) return NextResponse.json({ error: 'Connection not found' }, { status: 404 });

  const c = conn as ConnectionRow;
  if (c.exporter_user_id !== user.id && c.importer_user_id !== user.id) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }
  const senderRole: 'exporter' | 'importer' =
    c.exporter_user_id === user.id ? 'exporter' : 'importer';

  const { data: profile } = await supabase
    .from('profiles')
    .select('full_name')
    .eq('id', user.id)
    .maybeSingle();

  const { data, error } = await supabase
    .from('connection_messages')
    .insert({
      connection_id: connectionId,
      sender_user_id: user.id,
      sender_name: profile?.full_name || 'User',
      sender_role: senderRole,
      content: content.trim(),
      read: false,
    })
    .select('*')
    .single();
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  const receiverUserId =
    senderRole === 'exporter' ? c.importer_user_id : c.exporter_user_id;
  if (receiverUserId) {
    await supabase.from('app_notifications').insert({
      user_id: receiverUserId,
      type: 'new_message',
      title: 'New Message',
      description: `${profile?.full_name || 'User'} sent you a message.`,
      connection_id: connectionId,
      read: false,
    });
  }

  return NextResponse.json({
    success: true,
    message: toMessageDTO(data as MessageRow),
  });
}
