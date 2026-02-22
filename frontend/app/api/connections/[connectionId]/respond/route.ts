import { NextResponse } from 'next/server';
import { createClient } from '@/lib/supabase/server';

// POST /api/connections/:connectionId/respond - Accept or decline a connection
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
  const { action } = body as { action: 'accepted' | 'declined' };

  if (!['accepted', 'declined'].includes(action)) {
    return NextResponse.json(
      { success: false, error: 'Invalid action. Must be "accepted" or "declined".' },
      { status: 400 }
    );
  }

  const { data: existing, error: findErr } = await supabase
    .from('connections')
    .select('id, importer_user_id, exporter_user_id, importer_org_id, importer_org_name')
    .eq('id', connectionId)
    .maybeSingle();
  if (findErr) return NextResponse.json({ error: findErr.message }, { status: 500 });
  if (!existing) return NextResponse.json({ error: 'Connection not found' }, { status: 404 });

  let resolvedImporterUserId = existing.importer_user_id as string | null;
  if (!resolvedImporterUserId) {
    const mlId = String(existing.importer_org_id || existing.importer_org_name || '').trim();
    if (mlId) {
      const { data: mapped } = await supabase
        .from('buyer_directory')
        .select('user_id')
        .eq('ml_buyer_id', mlId)
        .maybeSingle();
      resolvedImporterUserId = mapped?.user_id ?? null;
    }
  }

  const canRespond = resolvedImporterUserId === user.id;
  if (!canRespond) {
    return NextResponse.json({ error: 'Only the importer can respond.' }, { status: 403 });
  }

  const respondedAt = new Date().toISOString();
  const { data, error } = await supabase
    .from('connections')
    .update({ status: action, responded_at: respondedAt, importer_user_id: resolvedImporterUserId })
    .eq('id', connectionId)
    .select('*')
    .single();
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  await supabase.from('app_notifications').insert({
    user_id: existing.exporter_user_id,
    type: action === 'accepted' ? 'connection_accepted' : 'connection_declined',
    title: action === 'accepted' ? 'Connection Accepted' : 'Connection Declined',
    description:
      action === 'accepted'
        ? 'Your connection request was accepted.'
        : 'Your connection request was declined.',
    connection_id: connectionId,
    read: false,
  });

  return NextResponse.json({ success: true, connection: data });
}
