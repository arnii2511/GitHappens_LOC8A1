-- Map ML buyer IDs (e.g. BUY_69687) to app users so connection requests can be routed.

create table if not exists public.buyer_directory (
  id uuid primary key default gen_random_uuid(),
  ml_buyer_id text not null unique,
  user_id uuid not null references auth.users(id) on delete cascade,
  org_id uuid references public.organizations(id) on delete set null,
  org_name text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.buyer_directory enable row level security;

create policy "buyer_directory_select_own"
  on public.buyer_directory
  for select
  using (auth.uid() = user_id);

create policy "buyer_directory_insert_own"
  on public.buyer_directory
  for insert
  with check (auth.uid() = user_id);

create policy "buyer_directory_update_own"
  on public.buyer_directory
  for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
