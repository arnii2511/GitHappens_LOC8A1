import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

type NotificationRow = {
  id: string;
  type: 'connection_sent' | 'connection_accepted' | 'connection_declined' | 'new_message';
  title: string;
  description: string;
  connection_id: string;
  read: boolean;
  created_at: string;
};

function toDTO(row: NotificationRow) {
  return {
    id: row.id,
    type: row.type,
    title: row.title,
    description: row.description,
    connectionId: row.connection_id,
    read: row.read,
    createdAt: row.created_at,
  };
}

export async function GET() {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();
  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { data, error } = await supabase
    .from('app_notifications')
    .select('*')
    .eq('user_id', user.id)
    .order('created_at', { ascending: false })
    .limit(200);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  const notifications = ((data as NotificationRow[]) ?? []).map(toDTO);
  return NextResponse.json({ notifications, total: notifications.length });
}

export async function PATCH(request: Request) {
  const supabase = await createClient();
  const {
    data: { user },
    error: authErr,
  } = await supabase.auth.getUser();
  if (authErr || !user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = (await request.json()) as {
    ids?: string[];
    connectionId?: string;
    read?: boolean;
  };
  const read = body.read ?? true;

  let q = supabase.from('app_notifications').update({ read }).eq('user_id', user.id);
  if (Array.isArray(body.ids) && body.ids.length > 0) {
    q = q.in('id', body.ids);
  } else if (body.connectionId) {
    q = q.eq('connection_id', body.connectionId);
  } else {
    return NextResponse.json(
      { error: 'Provide ids or connectionId to update notifications.' },
      { status: 400 }
    );
  }

  const { error } = await q;
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ success: true });
}
