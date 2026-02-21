
create table if not exists public.kyc_documents (
  id uuid primary key default gen_random_uuid(),
  org_id uuid not null references public.organizations(id) on delete cascade,
  uploaded_by uuid not null references auth.users(id),
  doc_type text not null check (doc_type in ('company_registration', 'tax_vat_gst', 'trade_license', 'brochure_catalog', 'certification')),
  file_name text not null,
  file_path text not null,
  file_size integer,
  mime_type text,
  status text not null default 'pending' check (status in ('pending', 'verified', 'rejected')),
  created_at timestamptz not null default now()
);

alter table public.kyc_documents enable row level security;


create policy "kyc_docs_select" on public.kyc_documents
  for select using (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = kyc_documents.org_id
        and organization_members.user_id = auth.uid()
    )
  );


create policy "kyc_docs_insert" on public.kyc_documents
  for insert with check (
    exists (
      select 1 from public.organization_members
      where organization_members.org_id = kyc_documents.org_id
        and organization_members.user_id = auth.uid()
    )
  );


create policy "kyc_docs_delete" on public.kyc_documents
  for delete using (
    auth.uid() = uploaded_by
  );
