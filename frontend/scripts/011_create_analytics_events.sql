-- Persistent per-user analytics events for dashboard metrics

create table if not exists public.analytics_swipe_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  action text not null check (action in ('like', 'dislike', 'save', 'block')),
  final_score numeric(6,2) not null default 0,
  match_id text,
  created_at timestamptz not null default now()
);

alter table public.analytics_swipe_events enable row level security;

create policy "analytics_events_select_own"
  on public.analytics_swipe_events
  for select
  using (auth.uid() = user_id);

create policy "analytics_events_insert_own"
  on public.analytics_swipe_events
  for insert
  with check (auth.uid() = user_id);
