create extension if not exists pgcrypto;

create table if not exists public.meetings (
  id uuid primary key,
  user_id uuid null,
  title text,
  started_at timestamptz not null default now(),
  ended_at timestamptz null,
  status text not null default 'active',
  created_at timestamptz not null default now()
);

create table if not exists public.frame_events (
  id uuid primary key default gen_random_uuid(),
  meeting_id uuid not null references public.meetings(id) on delete cascade,
  timestamp double precision not null,
  mode text not null,
  mode_confidence double precision null,
  visual_change boolean not null default false,
  processor_used text null,
  summary text null,
  payload_json jsonb not null default '{}'::jsonb,
  debug_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.transcript_segments (
  id uuid primary key default gen_random_uuid(),
  meeting_id uuid not null references public.meetings(id) on delete cascade,
  start_time double precision not null,
  end_time double precision not null,
  speaker text null,
  text text not null,
  is_final boolean not null default true,
  created_at timestamptz not null default now()
);

create table if not exists public.meeting_documents (
  id uuid primary key default gen_random_uuid(),
  meeting_id uuid not null references public.meetings(id) on delete cascade,
  markdown text null,
  json jsonb null,
  created_at timestamptz not null default now()
);

alter table public.meetings enable row level security;
alter table public.frame_events enable row level security;
alter table public.transcript_segments enable row level security;
alter table public.meeting_documents enable row level security;

drop policy if exists "anon can read meetings" on public.meetings;
create policy "anon can read meetings"
on public.meetings
for select
to anon
using (true);

drop policy if exists "anon can insert meetings" on public.meetings;
create policy "anon can insert meetings"
on public.meetings
for insert
to anon
with check (true);

drop policy if exists "anon can update meetings" on public.meetings;
create policy "anon can update meetings"
on public.meetings
for update
to anon
using (true)
with check (true);

drop policy if exists "anon can read frame_events" on public.frame_events;
create policy "anon can read frame_events"
on public.frame_events
for select
to anon
using (true);

drop policy if exists "anon can insert frame_events" on public.frame_events;
create policy "anon can insert frame_events"
on public.frame_events
for insert
to anon
with check (true);

drop policy if exists "anon can read transcript_segments" on public.transcript_segments;
create policy "anon can read transcript_segments"
on public.transcript_segments
for select
to anon
using (true);

drop policy if exists "anon can insert transcript_segments" on public.transcript_segments;
create policy "anon can insert transcript_segments"
on public.transcript_segments
for insert
to anon
with check (true);

drop policy if exists "anon can read meeting_documents" on public.meeting_documents;
create policy "anon can read meeting_documents"
on public.meeting_documents
for select
to anon
using (true);

drop policy if exists "anon can insert meeting_documents" on public.meeting_documents;
create policy "anon can insert meeting_documents"
on public.meeting_documents
for insert
to anon
with check (true);

create index if not exists idx_frame_events_meeting_id_timestamp
on public.frame_events (meeting_id, "timestamp");

create index if not exists idx_transcript_segments_meeting_id_start_time
on public.transcript_segments (meeting_id, start_time);
