'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import {
  Globe,
  Loader2,
  User,
  Building2,
  Package,
  ShieldCheck,
  Pencil,
  X,
  Check,
  ArrowLeft,
  Mail,
  Phone,
  MapPin,
  Calendar,
  Users,
  DollarSign,
  ExternalLink,
  FileText,
  LogOut,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

/* ─── Constants (same as onboarding) ─── */

const COUNTRIES = [
  'United States', 'United Kingdom', 'China', 'India', 'Germany', 'Japan',
  'Brazil', 'South Korea', 'France', 'Canada', 'Australia', 'Mexico',
  'Indonesia', 'Turkey', 'Saudi Arabia', 'Nigeria', 'South Africa',
  'United Arab Emirates', 'Vietnam', 'Thailand', 'Malaysia', 'Philippines',
  'Egypt', 'Kenya', 'Colombia', 'Chile', 'Bangladesh', 'Pakistan', 'Other',
]

const PRODUCT_CATEGORIES = [
  'Agriculture & Food', 'Textiles & Apparel', 'Electronics & Electrical',
  'Machinery & Equipment', 'Chemicals & Petrochemicals', 'Construction Materials',
  'Metals & Mining', 'Automotive & Transport', 'Consumer Goods',
  'Pharmaceuticals & Medical', 'Energy & Renewables', 'Other',
]

const CERTIFICATIONS = [
  'ISO 9001', 'ISO 14001', 'ISO 22000', 'HACCP', 'CE Marking', 'FDA Approved',
  'Fair Trade', 'Organic Certified', 'GMP', 'BRC', 'BSCI', 'SA8000', 'Other',
]

const PAYMENT_TERMS = [
  'Letter of Credit (L/C)', 'Telegraphic Transfer (T/T)', 'Documentary Collection (D/P)',
  'Open Account', 'Cash in Advance', 'Consignment',
]

const INCOTERMS = [
  'EXW', 'FCA', 'FAS', 'FOB', 'CFR', 'CIF', 'CPT', 'CIP', 'DAP', 'DPU', 'DDP',
]

const LOGISTICS = [
  'Sea Freight (FCL)', 'Sea Freight (LCL)', 'Air Freight', 'Road Transport',
  'Rail Freight', 'Multimodal', 'Warehousing', 'Cold Chain',
]

/* ─── Types ─── */

interface ProfileData {
  id: string
  account_type: string
  full_name: string
  phone: string
  onboarding_completed: boolean
  created_at: string
  updated_at: string
}

interface OrgData {
  id: string
  user_id: string
  company_name: string
  registration_number: string | null
  country: string
  city: string | null
  address: string | null
  website: string | null
  year_established: string | null
  number_of_employees: string | null
  annual_revenue: string | null
  description: string | null
}

interface TradeData {
  id: string
  user_id: string
  product_categories: string[]
  primary_markets: string[]
  certifications: string[]
  min_order_value: string | null
  preferred_payment_terms: string[]
  preferred_incoterms: string[]
  logistics_capabilities: string[]
}

interface DocData {
  id: string
  user_id: string
  document_type: string
  file_name: string
  file_path: string
  file_size: number | null
  mime_type: string | null
  created_at: string
}

type EditingSection = 'profile' | 'organization' | 'trade' | null

/* ─── Page ─── */

export default function ProfilePage() {
  const router = useRouter()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [editing, setEditing] = useState<EditingSection>(null)

  const [email, setEmail] = useState('')
  const [profile, setProfile] = useState<ProfileData | null>(null)
  const [org, setOrg] = useState<OrgData | null>(null)
  const [trade, setTrade] = useState<TradeData | null>(null)
  const [documents, setDocuments] = useState<DocData[]>([])

  // Edit state copies
  const [editProfile, setEditProfile] = useState<Partial<ProfileData>>({})
  const [editOrg, setEditOrg] = useState<Partial<OrgData>>({})
  const [editTrade, setEditTrade] = useState<Partial<TradeData>>({})

  const fetchProfile = useCallback(async () => {
    try {
      const res = await fetch('/api/profile')
      if (!res.ok) {
        if (res.status === 401) {
          router.push('/auth/login')
          return
        }
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.error || `Failed to fetch profile (${res.status})`)
      }
      const data = await res.json()
      setEmail(data.email ?? '')
      setProfile(data.profile ?? null)
      setOrg(data.organization ?? null)
      setTrade(data.tradeInfo ?? null)
      setDocuments(data.documents ?? [])
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load profile')
    } finally {
      setLoading(false)
    }
  }, [router])

  useEffect(() => {
    fetchProfile()
  }, [fetchProfile])

  function startEdit(section: EditingSection) {
    setError(null)
    setSuccess(null)
    if (section === 'profile' && profile) {
      setEditProfile({ ...profile })
    }
    if (section === 'organization' && org) {
      setEditOrg({ ...org })
    }
    if (section === 'trade' && trade) {
      setEditTrade({ ...trade })
    }
    setEditing(section)
  }

  function cancelEdit() {
    setEditing(null)
    setError(null)
  }

  async function saveSection() {
    if (!editing) return
    setSaving(true)
    setError(null)
    setSuccess(null)

    try {
      let payload: Record<string, unknown> = {}

      if (editing === 'profile') payload = editProfile
      if (editing === 'organization') payload = editOrg
      if (editing === 'trade') payload = editTrade

      const res = await fetch('/api/profile', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ section: editing, data: payload }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.error || 'Update failed')
      }

      setSuccess('Changes saved successfully')
      setEditing(null)
      await fetchProfile()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save changes')
    } finally {
      setSaving(false)
    }
  }

  async function handleLogout() {
    const supabase = createClient()
    await supabase.auth.signOut()
    router.push('/auth/login')
  }

  function toggleTradeArray(field: keyof TradeData, item: string) {
    setEditTrade((prev) => {
      const arr = (prev[field] as string[]) ?? []
      const updated = arr.includes(item)
        ? arr.filter((x) => x !== item)
        : [...arr, item]
      return { ...prev, [field]: updated }
    })
  }

  function formatFileSize(bytes: number | null) {
    if (!bytes) return 'Unknown size'
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  function getInitials(name: string) {
    return name
      .split(' ')
      .map((w) => w[0])
      .join('')
      .toUpperCase()
      .slice(0, 2)
  }

  /* ─── Render ─── */

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push('/')}
              className="text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="w-4 h-4 mr-1" />
              Back
            </Button>
            <Separator orientation="vertical" className="h-6 bg-border" />
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-primary to-secondary rounded-lg blur opacity-50" />
              <div className="relative bg-card px-3 py-2 rounded-lg flex items-center gap-2">
                <Globe className="w-5 h-5 text-secondary" />
                <span className="font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  NexPort
                </span>
              </div>
            </div>
          </div>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleLogout}
                  className="text-muted-foreground hover:text-destructive"
                >
                  <LogOut className="w-4 h-4 mr-1" />
                  <span className="hidden sm:inline">Logout</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent className="bg-card border-border text-card-foreground">
                <p className="text-xs">Sign out of your account</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-8 max-w-3xl">
        {/* Status messages */}
        {error && (
          <div className="mb-6 bg-destructive/10 border border-destructive/30 text-destructive-foreground text-sm rounded-lg p-3">
            {error}
          </div>
        )}
        {success && (
          <div className="mb-6 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm rounded-lg p-3">
            {success}
          </div>
        )}

        {/* Profile Hero */}
        <div className="bg-card border border-border rounded-xl p-6 mb-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-primary/20 border-2 border-primary flex items-center justify-center text-primary text-xl font-bold shrink-0">
              {profile?.full_name ? getInitials(profile.full_name) : 'U'}
            </div>
            <div className="flex-1 min-w-0">
              <h1 className="text-2xl font-bold text-foreground text-balance">
                {profile?.full_name || 'User'}
              </h1>
              <div className="flex flex-wrap items-center gap-3 mt-1">
                <span className="text-sm text-muted-foreground flex items-center gap-1.5">
                  <Mail className="w-3.5 h-3.5" />
                  {email}
                </span>
                {profile?.account_type && (
                  <Badge
                    variant="secondary"
                    className="text-xs capitalize bg-primary/15 text-primary border-primary/30"
                  >
                    {profile.account_type}
                  </Badge>
                )}
              </div>
              {org?.company_name && (
                <p className="text-sm text-muted-foreground mt-1 flex items-center gap-1.5">
                  <Building2 className="w-3.5 h-3.5" />
                  {org.company_name}
                  {org.country && <span className="text-border">|</span>}
                  {org.country}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Personal Information Section */}
        <SectionCard
          icon={<User className="w-5 h-5" />}
          title="Personal Information"
          isEditing={editing === 'profile'}
          onEdit={() => startEdit('profile')}
          onCancel={cancelEdit}
          onSave={saveSection}
          saving={saving}
        >
          {editing === 'profile' ? (
            <div className="space-y-4">
              <div>
                <Label className="text-muted-foreground text-xs mb-1.5 block">Full Name</Label>
                <Input
                  value={editProfile.full_name ?? ''}
                  onChange={(e) => setEditProfile((p) => ({ ...p, full_name: e.target.value }))}
                  className="bg-background border-border text-foreground"
                />
              </div>
              <div>
                <Label className="text-muted-foreground text-xs mb-1.5 block">Phone</Label>
                <Input
                  value={editProfile.phone ?? ''}
                  onChange={(e) => setEditProfile((p) => ({ ...p, phone: e.target.value }))}
                  className="bg-background border-border text-foreground"
                />
              </div>
              <div>
                <Label className="text-muted-foreground text-xs mb-1.5 block">Account Type</Label>
                <Select
                  value={editProfile.account_type ?? 'exporter'}
                  onValueChange={(v) => setEditProfile((p) => ({ ...p, account_type: v }))}
                >
                  <SelectTrigger className="bg-background border-border text-foreground">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    <SelectItem value="exporter">Exporter</SelectItem>
                    <SelectItem value="buyer">Buyer</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <InfoRow icon={<User className="w-4 h-4" />} label="Full Name" value={profile?.full_name} />
              <InfoRow icon={<Mail className="w-4 h-4" />} label="Email" value={email} />
              <InfoRow icon={<Phone className="w-4 h-4" />} label="Phone" value={profile?.phone} />
              <InfoRow
                icon={<ShieldCheck className="w-4 h-4" />}
                label="Account Type"
                value={profile?.account_type}
                capitalize
              />
            </div>
          )}
        </SectionCard>

        {/* Organization Section */}
        <SectionCard
          icon={<Building2 className="w-5 h-5" />}
          title="Organization Details"
          isEditing={editing === 'organization'}
          onEdit={() => startEdit('organization')}
          onCancel={cancelEdit}
          onSave={saveSection}
          saving={saving}
        >
          {editing === 'organization' ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <Label className="text-muted-foreground text-xs mb-1.5 block">Company Name</Label>
                  <Input
                    value={editOrg.company_name ?? ''}
                    onChange={(e) => setEditOrg((p) => ({ ...p, company_name: e.target.value }))}
                    className="bg-background border-border text-foreground"
                  />
                </div>
                <div>
                  <Label className="text-muted-foreground text-xs mb-1.5 block">Registration Number</Label>
                  <Input
                    value={editOrg.registration_number ?? ''}
                    onChange={(e) => setEditOrg((p) => ({ ...p, registration_number: e.target.value }))}
                    className="bg-background border-border text-foreground"
                  />
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <Label className="text-muted-foreground text-xs mb-1.5 block">Country</Label>
                  <Select
                    value={editOrg.country ?? ''}
                    onValueChange={(v) => setEditOrg((p) => ({ ...p, country: v }))}
                  >
                    <SelectTrigger className="bg-background border-border text-foreground">
                      <SelectValue placeholder="Select country" />
                    </SelectTrigger>
                    <SelectContent className="bg-card border-border max-h-60">
                      {COUNTRIES.map((c) => (
                        <SelectItem key={c} value={c}>{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label className="text-muted-foreground text-xs mb-1.5 block">City</Label>
                  <Input
                    value={editOrg.city ?? ''}
                    onChange={(e) => setEditOrg((p) => ({ ...p, city: e.target.value }))}
                    className="bg-background border-border text-foreground"
                  />
                </div>
              </div>
              <div>
                <Label className="text-muted-foreground text-xs mb-1.5 block">Address</Label>
                <Input
                  value={editOrg.address ?? ''}
                  onChange={(e) => setEditOrg((p) => ({ ...p, address: e.target.value }))}
                  className="bg-background border-border text-foreground"
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <Label className="text-muted-foreground text-xs mb-1.5 block">Website</Label>
                  <Input
                    value={editOrg.website ?? ''}
                    onChange={(e) => setEditOrg((p) => ({ ...p, website: e.target.value }))}
                    className="bg-background border-border text-foreground"
                    placeholder="https://"
                  />
                </div>
                <div>
                  <Label className="text-muted-foreground text-xs mb-1.5 block">Year Established</Label>
                  <Input
                    value={editOrg.year_established ?? ''}
                    onChange={(e) =>
                      setEditOrg((p) => ({
                        ...p,
                        year_established: e.target.value || null,
                      }))
                    }
                    className="bg-background border-border text-foreground"
                    placeholder="e.g. 2010"
                  />
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <Label className="text-muted-foreground text-xs mb-1.5 block">Number of Employees</Label>
                  <Input
                    value={editOrg.number_of_employees ?? ''}
                    onChange={(e) => setEditOrg((p) => ({ ...p, number_of_employees: e.target.value }))}
                    className="bg-background border-border text-foreground"
                  />
                </div>
                <div>
                  <Label className="text-muted-foreground text-xs mb-1.5 block">Annual Revenue</Label>
                  <Input
                    value={editOrg.annual_revenue ?? ''}
                    onChange={(e) => setEditOrg((p) => ({ ...p, annual_revenue: e.target.value }))}
                    className="bg-background border-border text-foreground"
                  />
                </div>
              </div>
              <div>
                <Label className="text-muted-foreground text-xs mb-1.5 block">Description</Label>
                <Textarea
                  value={editOrg.description ?? ''}
                  onChange={(e) => setEditOrg((p) => ({ ...p, description: e.target.value }))}
                  className="bg-background border-border text-foreground min-h-[80px]"
                />
              </div>
            </div>
          ) : org ? (
            <div className="space-y-3">
              <InfoRow icon={<Building2 className="w-4 h-4" />} label="Company Name" value={org?.company_name} />
              <InfoRow icon={<FileText className="w-4 h-4" />} label="Registration No." value={org?.registration_number} />
              <InfoRow icon={<MapPin className="w-4 h-4" />} label="Location" value={[org?.city, org?.country].filter(Boolean).join(', ') || null} />
              <InfoRow icon={<MapPin className="w-4 h-4" />} label="Address" value={org?.address} />
              {org?.website && (
                <div className="flex items-start gap-3 py-1">
                  <ExternalLink className="w-4 h-4 text-muted-foreground mt-0.5 shrink-0" />
                  <div>
                    <p className="text-xs text-muted-foreground">Website</p>
                    <a
                      href={org.website.startsWith('http') ? org.website : `https://${org.website}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-primary hover:underline"
                    >
                      {org.website}
                    </a>
                  </div>
                </div>
              )}
              <InfoRow icon={<Calendar className="w-4 h-4" />} label="Year Established" value={org?.year_established} />
              <InfoRow icon={<Users className="w-4 h-4" />} label="Employees" value={org?.number_of_employees} />
              <InfoRow icon={<DollarSign className="w-4 h-4" />} label="Annual Revenue" value={org?.annual_revenue} />
              {org?.description && (
                <div className="pt-2">
                  <p className="text-xs text-muted-foreground mb-1">Description</p>
                  <p className="text-sm text-foreground/90 leading-relaxed">{org.description}</p>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No organization details added yet.</p>
          )}
        </SectionCard>

        {/* Trade Information Section */}
        <SectionCard
          icon={<Package className="w-5 h-5" />}
          title="Trade Information"
          isEditing={editing === 'trade'}
          onEdit={() => startEdit('trade')}
          onCancel={cancelEdit}
          onSave={saveSection}
          saving={saving}
        >
          {editing === 'trade' ? (
            <div className="space-y-5">
              <MultiSelectEdit
                label="Product Categories"
                options={PRODUCT_CATEGORIES}
                selected={(editTrade.product_categories as string[]) ?? []}
                onToggle={(item) => toggleTradeArray('product_categories', item)}
              />
              <MultiSelectEdit
                label="Primary Markets"
                options={COUNTRIES}
                selected={(editTrade.primary_markets as string[]) ?? []}
                onToggle={(item) => toggleTradeArray('primary_markets', item)}
              />
              <MultiSelectEdit
                label="Certifications"
                options={CERTIFICATIONS}
                selected={(editTrade.certifications as string[]) ?? []}
                onToggle={(item) => toggleTradeArray('certifications', item)}
              />
              <div>
                <Label className="text-muted-foreground text-xs mb-1.5 block">Min Order Value</Label>
                <Input
                  value={(editTrade.min_order_value as string) ?? ''}
                  onChange={(e) => setEditTrade((p) => ({ ...p, min_order_value: e.target.value }))}
                  className="bg-background border-border text-foreground"
                  placeholder="e.g. $5,000"
                />
              </div>
              <MultiSelectEdit
                label="Payment Terms"
                options={PAYMENT_TERMS}
                selected={(editTrade.preferred_payment_terms as string[]) ?? []}
                onToggle={(item) => toggleTradeArray('preferred_payment_terms', item)}
              />
              <MultiSelectEdit
                label="Incoterms"
                options={INCOTERMS}
                selected={(editTrade.preferred_incoterms as string[]) ?? []}
                onToggle={(item) => toggleTradeArray('preferred_incoterms', item)}
              />
              <MultiSelectEdit
                label="Logistics Capabilities"
                options={LOGISTICS}
                selected={(editTrade.logistics_capabilities as string[]) ?? []}
                onToggle={(item) => toggleTradeArray('logistics_capabilities', item)}
              />
            </div>
          ) : trade ? (
            <div className="space-y-4">
              <TagRow label="Product Categories" tags={trade?.product_categories} />
              <TagRow label="Primary Markets" tags={trade?.primary_markets} />
              <TagRow label="Certifications" tags={trade?.certifications} />
              <InfoRow icon={<DollarSign className="w-4 h-4" />} label="Min Order Value" value={trade?.min_order_value} />
              <TagRow label="Payment Terms" tags={trade?.preferred_payment_terms} />
              <TagRow label="Incoterms" tags={trade?.preferred_incoterms} />
              <TagRow label="Logistics" tags={trade?.logistics_capabilities} />
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No trade information added yet.</p>
          )}
        </SectionCard>

        {/* Documents Section (read-only) */}
        <SectionCard
          icon={<ShieldCheck className="w-5 h-5" />}
          title="Uploaded Documents"
          readOnly
        >
          {documents.length > 0 ? (
            <div className="space-y-2">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-center gap-3 p-3 bg-background/50 rounded-lg border border-border/50"
                >
                  <FileText className="w-4 h-4 text-muted-foreground shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-foreground truncate">{doc.file_name}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(doc.file_size)}
                      {doc.mime_type && ` -- ${doc.mime_type}`}
                    </p>
                  </div>
                  <Badge variant="outline" className="text-[10px] capitalize border-border text-muted-foreground shrink-0">
                    {doc.document_type}
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No documents uploaded.</p>
          )}
        </SectionCard>

        {/* Account Meta */}
        <div className="mt-4 mb-8 text-center text-xs text-muted-foreground">
          Member since{' '}
          {profile?.created_at
            ? new Date(profile.created_at).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })
            : 'N/A'}
        </div>
      </main>
    </div>
  )
}

/* ─── Sub-components ─── */

function SectionCard({
  icon,
  title,
  isEditing = false,
  readOnly = false,
  onEdit,
  onCancel,
  onSave,
  saving = false,
  children,
}: {
  icon: React.ReactNode
  title: string
  isEditing?: boolean
  readOnly?: boolean
  onEdit?: () => void
  onCancel?: () => void
  onSave?: () => void
  saving?: boolean
  children: React.ReactNode
}) {
  return (
    <div className="bg-card border border-border rounded-xl mb-4 overflow-hidden">
      <div className="flex items-center justify-between px-6 py-4 border-b border-border/50">
        <div className="flex items-center gap-2 text-foreground">
          <span className="text-primary">{icon}</span>
          <h2 className="font-semibold text-base">{title}</h2>
        </div>
        {!readOnly && (
          <div className="flex items-center gap-2">
            {isEditing ? (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onCancel}
                  className="text-muted-foreground hover:text-foreground"
                >
                  <X className="w-4 h-4 mr-1" />
                  Cancel
                </Button>
                <Button
                  size="sm"
                  onClick={onSave}
                  disabled={saving}
                  className="bg-primary hover:bg-primary/90 text-primary-foreground"
                >
                  {saving ? (
                    <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                  ) : (
                    <Check className="w-4 h-4 mr-1" />
                  )}
                  Save
                </Button>
              </>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                onClick={onEdit}
                className="text-muted-foreground hover:text-foreground"
              >
                <Pencil className="w-4 h-4 mr-1" />
                Edit
              </Button>
            )}
          </div>
        )}
      </div>
      <div className="px-6 py-4">{children}</div>
    </div>
  )
}

function InfoRow({
  icon,
  label,
  value,
  capitalize = false,
}: {
  icon: React.ReactNode
  label: string
  value?: string | null
  capitalize?: boolean
}) {
  if (!value) return null
  return (
    <div className="flex items-start gap-3 py-1">
      <span className="text-muted-foreground mt-0.5 shrink-0">{icon}</span>
      <div>
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={`text-sm text-foreground ${capitalize ? 'capitalize' : ''}`}>{value}</p>
      </div>
    </div>
  )
}

function TagRow({ label, tags }: { label: string; tags?: string[] | null }) {
  if (!tags || tags.length === 0) return null
  return (
    <div>
      <p className="text-xs text-muted-foreground mb-1.5">{label}</p>
      <div className="flex flex-wrap gap-1.5">
        {tags.map((tag) => (
          <Badge
            key={tag}
            variant="secondary"
            className="text-xs bg-primary/10 text-primary border-primary/20"
          >
            {tag}
          </Badge>
        ))}
      </div>
    </div>
  )
}

function MultiSelectEdit({
  label,
  options,
  selected,
  onToggle,
}: {
  label: string
  options: string[]
  selected: string[]
  onToggle: (item: string) => void
}) {
  return (
    <div>
      <Label className="text-muted-foreground text-xs mb-2 block">{label}</Label>
      <div className="flex flex-wrap gap-1.5">
        {options.map((opt) => {
          const isSelected = selected.includes(opt)
          return (
            <button
              key={opt}
              type="button"
              onClick={() => onToggle(opt)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
                isSelected
                  ? 'bg-primary/20 text-primary border-primary/40'
                  : 'bg-background text-muted-foreground border-border hover:border-primary/30 hover:text-foreground'
              }`}
            >
              {opt}
            </button>
          )
        })}
      </div>
    </div>
  )
}
