'use client'

import { Check, X } from 'lucide-react'

interface PasswordStrengthMeterProps {
  password: string
}

const rules = [
  { label: 'At least 8 characters', test: (p: string) => p.length >= 8 },
  { label: 'Uppercase letter', test: (p: string) => /[A-Z]/.test(p) },
  { label: 'Lowercase letter', test: (p: string) => /[a-z]/.test(p) },
  { label: 'Number', test: (p: string) => /\d/.test(p) },
  { label: 'Special character', test: (p: string) => /[^A-Za-z0-9]/.test(p) },
]

export default function PasswordStrengthMeter({ password }: PasswordStrengthMeterProps) {
  const passed = rules.filter((r) => r.test(password)).length
  const strength = passed <= 2 ? 'Weak' : passed <= 4 ? 'Fair' : 'Strong'
  const strengthColor =
    passed <= 2
      ? 'bg-destructive'
      : passed <= 4
        ? 'bg-yellow-500'
        : 'bg-emerald-500'
  const pct = (passed / rules.length) * 100

  if (!password) return null

  return (
    <div className="space-y-3">
      {/* Strength bar */}
      <div className="flex items-center gap-3">
        <div className="flex-1 h-1.5 rounded-full bg-border overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${strengthColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span
          className={`text-xs font-medium ${
            passed <= 2
              ? 'text-destructive-foreground'
              : passed <= 4
                ? 'text-yellow-500'
                : 'text-emerald-500'
          }`}
        >
          {strength}
        </span>
      </div>

      {/* Rule checklist */}
      <ul className="space-y-1">
        {rules.map((rule) => {
          const ok = rule.test(password)
          return (
            <li key={rule.label} className="flex items-center gap-2 text-xs">
              {ok ? (
                <Check className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
              ) : (
                <X className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
              )}
              <span className={ok ? 'text-emerald-500' : 'text-muted-foreground'}>
                {rule.label}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
