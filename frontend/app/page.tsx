"use client"

import LoginPage from "./auth/login/page"
import { Globe } from 'lucide-react'

export default function Page() {
  return (
    <div className="min-h-screen flex bg-background text-foreground">

      {/* LEFT SIDE */}
      <div className="hidden lg:flex lg:w-1/2 relative flex-col justify-between p-16 bg-linear-to-r from-blue-800 to-cyan-700 rounded-none border-r border-border">

        {/* Soft Accent Glow */}
        <div className="absolute top-0 left-0 w-72 h-72" />

        {/* Logo */}
        <div className="flex flex-col items-start gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-linear-to-r from-blue-600 to-cyan-500 rounded-lg blur opacity-50" />
            <div className="relative bg-card px-3 py-2 rounded-lg flex items-center gap-2">
              <Globe className="w-6 h-6 text-secondary" />
              <span className="font-bold text-lg bg-linear-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                NexPort
              </span>
            </div>
          </div>
        </div>

        {/* Hero Content */}
        <div className="max-w-xl space-y-14">

          <div>
            <h2 className="text-5xl font-bold leading-tight tracking-tight mb-6">
              AI-Powered Export
              <br />
              Growth Engine
            </h2>

            <p className="text-lg text-muted-foreground leading-relaxed max-w-lg">
              Scale your manufacturing exports with intelligent lead generation
              and automated outreach designed for modern trade teams.
            </p>
          </div>

          {/* Steps */}
          <div className="space-y-10">

            {[
              {
                number: "1",
                title: "Approve Curated Leads",
                desc: "Swipe through AI-selected export opportunities tailored to your ICP",
              },
              {
                number: "2",
                title: "AI Handles Outreach",
                desc: "Omni-channel automation across LinkedIn, Email, WhatsApp, and Calls",
              },
              {
                number: "3",
                title: "Attend Scheduled Meetings",
                desc: "Focus on closing deals while AI manages the pipeline",
              },
            ].map((item) => (
              <div key={item.number} className="flex gap-5 items-start group">
                <div className="w-11 h-11 flex items-center justify-center rounded-xl bg-background border border-border font-semibold text-primary group-hover:bg-primary/10 transition">
                  {item.number}
                </div>
                <div>
                  <h3 className="font-semibold text-lg mb-1">
                    {item.title}
                  </h3>
                  <p className="text-muted-foreground text-sm leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              </div>
            ))}

          </div>
        </div>

        {/* Footer */}
        <div className="text-sm text-muted-foreground">
          © 2025 NexPort. All rights reserved.
        </div>
      </div>


      {/* RIGHT SIDE */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          <LoginPage />
        </div>
      </div>

    </div>
  )
}
