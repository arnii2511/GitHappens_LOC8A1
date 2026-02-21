-- Create kyc-docs storage bucket (private)
insert into storage.buckets (id, name, public)
values ('kyc-docs', 'kyc-docs', false)
on conflict (id) do nothing;

-- Storage policy: org members can upload files to their org folder
create policy "kyc_storage_insert" on storage.objects
  for insert with check (
    bucket_id = 'kyc-docs'
    and auth.role() = 'authenticated'
    and exists (
      select 1 from public.organization_members
      where organization_members.org_id = (storage.foldername(name))[1]::uuid
        and organization_members.user_id = auth.uid()
    )
  );

-- Storage policy: org members can read files from their org folder
create policy "kyc_storage_select" on storage.objects
  for select using (
    bucket_id = 'kyc-docs'
    and exists (
      select 1 from public.organization_members
      where organization_members.org_id = (storage.foldername(name))[1]::uuid
        and organization_members.user_id = auth.uid()
    )
  );

-- Storage policy: org members can delete files from their org folder
create policy "kyc_storage_delete" on storage.objects
  for delete using (
    bucket_id = 'kyc-docs'
    and exists (
      select 1 from public.organization_members
      where organization_members.org_id = (storage.foldername(name))[1]::uuid
        and organization_members.user_id = auth.uid()
    )
  );
