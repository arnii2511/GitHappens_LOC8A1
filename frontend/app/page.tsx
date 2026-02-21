"use client"

import LoginPage from "./auth/login/page"

export default function Page() {
  return (
    <div className="min-h-screen flex bg-background text-foreground">

      {/* LEFT SIDE */}
      <div className="hidden lg:flex lg:w-1/2 relative flex-col justify-between p-16 bg-muted/40 border-r border-border">

        {/* Soft Accent Glow */}
        <div className="absolute top-0 left-0 w-72 h-72 bg-primary/10 blur-3xl rounded-full -z-10" />

        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary rounded-xl shadow-sm" />
          <h1 className="text-2xl font-semibold tracking-tight">
            NexPort
          </h1>
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
