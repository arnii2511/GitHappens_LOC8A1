-- Dynamic connections, messaging, and notifications

create table if not exists public.connections (
  id uuid primary key default gen_random_uuid(),
  match_id text,
  exporter_user_id uuid not null references auth.users(id) on delete cascade,
  importer_user_id uuid references auth.users(id) on delete set null,
  exporter_org_id text,
  exporter_org_name text not null,
  importer_org_id text,
  importer_org_name text not null,
  importer_country text,
  importer_industry text,
  final_score numeric(6,2) not null default 0,
  status text not null default 'pending' check (status in ('pending', 'accepted', 'declined')),
  note text,
  created_at timestamptz not null default now(),
  responded_at timestamptz
);

alter table public.connections enable row level security;

create policy "connections_select_participants"
  on public.connections
  for select
  using (auth.uid() = exporter_user_id or auth.uid() = importer_user_id);

create policy "connections_insert_exporter"
  on public.connections
  for insert
  with check (auth.uid() = exporter_user_id);

create policy "connections_update_importer_response"
  on public.connections
  for update
  using (auth.uid() = importer_user_id)
  with check (auth.uid() = importer_user_id);

create table if not exists public.connection_messages (
  id uuid primary key default gen_random_uuid(),
  connection_id uuid not null references public.connections(id) on delete cascade,
  sender_user_id uuid not null references auth.users(id) on delete cascade,
  sender_name text not null,
  sender_role text not null check (sender_role in ('exporter', 'importer')),
  content text not null,
  read boolean not null default false,
  created_at timestamptz not null default now()
);

alter table public.connection_messages enable row level security;

create policy "messages_select_participants"
  on public.connection_messages
  for select
  using (
    exists (
      select 1
      from public.connections c
      where c.id = connection_id
        and (auth.uid() = c.exporter_user_id or auth.uid() = c.importer_user_id)
    )
  );

create policy "messages_insert_participant_sender"
  on public.connection_messages
  for insert
  with check (
    auth.uid() = sender_user_id
    and exists (
      select 1
      from public.connections c
      where c.id = connection_id
        and (auth.uid() = c.exporter_user_id or auth.uid() = c.importer_user_id)
    )
  );

create policy "messages_update_participants"
  on public.connection_messages
  for update
  using (
    exists (
      select 1
      from public.connections c
      where c.id = connection_id
        and (auth.uid() = c.exporter_user_id or auth.uid() = c.importer_user_id)
    )
  );

create table if not exists public.app_notifications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  type text not null check (type in ('connection_sent', 'connection_accepted', 'connection_declined', 'new_message')),
  title text not null,
  description text not null,
  connection_id uuid references public.connections(id) on delete cascade,
  read boolean not null default false,
  created_at timestamptz not null default now()
);

alter table public.app_notifications enable row level security;

create policy "notifications_select_own"
  on public.app_notifications
  for select
  using (auth.uid() = user_id);

create policy "notifications_insert_own"
  on public.app_notifications
  for insert
  with check (auth.uid() = user_id);

create policy "notifications_update_own"
  on public.app_notifications
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
