'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Mail, Share2, Download, Copy, Check } from 'lucide-react';

interface Template {
  id: string;
  name: string;
  subject: string;
  body: string;
  type: 'email' | 'linkedin';
}

const templates: Template[] = [
  {
    id: 'intro',
    name: 'Professional Introduction',
    subject: 'Exploring Partnership Opportunities - [Your Company]',
    body: `Dear [Buyer Name],

I hope this email finds you well. I've identified your organization as an ideal partner based on your product interests and market reach.

We specialize in [Product Category] and have been delivering high-quality solutions to companies across [Regions] with proven track records of:
• Consistent quality and on-time delivery
• Competitive pricing with volume discounts
• Flexible minimum order quantities
• Complete supply chain transparency

I'd welcome the opportunity to discuss how we can support your growth objectives.

Best regards,
[Your Name]`,
    type: 'email',
  },
  {
    id: 'proposal',
    name: 'Detailed Proposal',
    subject: 'Strategic Partnership Proposal - Joint Growth Opportunity',
    body: `Dear [Buyer Name],

Following our discussion, I'm excited to present a tailored partnership proposal designed to maximize value for both organizations.

Key Benefits:
• Cost savings of 15-25% vs current market rates
• Dedicated account management
• 30-day payment terms (eligible accounts)
• Custom packaging and labeling
• Priority fulfillment window

Timeline: We can commence supply within 4 weeks of order confirmation.

I've attached our product specifications and certification documents. Please let me know your preferred next steps.

Best regards,
[Your Name]`,
    type: 'email',
  },
  {
    id: 'followup',
    name: 'Strategic Follow-up',
    subject: 'Let\'s Connect - [Product] Partnership',
    body: `Hi [Buyer Name],

I noticed your company has been expanding in [Market/Category]. I think we could create significant value together given your growth trajectory.

Quick question: What's your current strategy for [Product Category] sourcing? I'd love to explore if we could be a better partner than your current suppliers.

Happy to jump on a call this week if you're interested.

Cheers,
[Your Name]`,
    type: 'linkedin',
  },
];

export default function Outreach() {
  const [selectedTemplate, setSelectedTemplate] = useState<Template>(templates[0]);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const content = `Subject: ${selectedTemplate.subject}\n\n${selectedTemplate.body}`;
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleExport = () => {
    const csv = templates.map(t => 
      `"${t.name}","${t.type}","${t.subject.replace(/"/g, '""')}","${t.body.replace(/"/g, '""')}"`
    ).join('\n');
    
    const element = document.createElement('a');
    element.setAttribute('href', `data:text/csv;charset=utf-8,${encodeURIComponent('Name,Type,Subject,Body\n' + csv)}`);
    element.setAttribute('download', 'outreach-templates.csv');
    element.click();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600/20 to-cyan-500/20 border border-blue-500/30 rounded-xl p-6">
        <h2 className="text-2xl font-bold text-white mb-2">Outreach Center</h2>
        <p className="text-slate-300">
          Use AI-optimized templates to reach out to qualified buyers. Personalize and export for your CRM.
        </p>
      </div>

      {/* Template Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {templates.map((template) => (
          <button
            key={template.id}
            onClick={() => setSelectedTemplate(template)}
            className={`text-left p-4 rounded-lg border-2 transition-all ${
              selectedTemplate.id === template.id
                ? 'bg-blue-600/20 border-blue-500 shadow-lg shadow-blue-500/20'
                : 'bg-slate-800/50 border-slate-700/50 hover:border-slate-600'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              {template.type === 'email' ? (
                <Mail className="w-4 h-4 text-cyan-400" />
              ) : (
                <Share2 className="w-4 h-4 text-cyan-400" />
              )}
              <span className="text-xs font-semibold text-slate-400 uppercase">
                {template.type}
              </span>
            </div>
            <h3 className="font-bold text-white text-sm">{template.name}</h3>
            <p className="text-xs text-slate-400 mt-1 line-clamp-2">{template.subject}</p>
          </button>
        ))}
      </div>

      {/* Template Editor */}
      <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-6 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-slate-300 mb-2">Subject Line</label>
          <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600/50">
            <p className="text-white">{selectedTemplate.subject}</p>
          </div>
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-300 mb-2">Email Body</label>
          <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/50 min-h-64">
            <p className="text-slate-200 whitespace-pre-wrap text-sm leading-relaxed">
              {selectedTemplate.body}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4">
          <Button
            onClick={handleCopy}
            className="flex-1 bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-700 hover:to-cyan-600 text-white font-semibold"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4 mr-2" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="w-4 h-4 mr-2" />
                Copy to Clipboard
              </>
            )}
          </Button>
          <Button
            onClick={handleExport}
            variant="outline"
            className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:text-slate-100 hover:border-slate-500"
          >
            <Download className="w-4 h-4 mr-2" />
            Export CSV
          </Button>
        </div>
      </Card>

      {/* Personalization Tips */}
      <Card className="bg-gradient-to-br from-blue-900/20 to-cyan-900/20 border border-blue-500/30 p-6">
        <h3 className="text-lg font-bold text-white mb-4">Personalization Tips</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-semibold text-cyan-400 mb-2">Variables to Replace:</h4>
            <ul className="space-y-1 text-sm text-slate-300">
              <li>• [Buyer Name] - Use their first name</li>
              <li>• [Company] - Their organization name</li>
              <li>• [Product Category] - Your relevant products</li>
              <li>• [Regions] - Where you serve</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-cyan-400 mb-2">Best Practices:</h4>
            <ul className="space-y-1 text-sm text-slate-300">
              <li>✓ Research the buyer before sending</li>
              <li>✓ Reference recent company news/achievements</li>
              <li>✓ Keep subject lines under 50 characters</li>
              <li>✓ Include clear call-to-action</li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-4 text-center">
          <p className="text-2xl font-bold text-cyan-400">48%</p>
          <p className="text-xs text-slate-400 mt-1">Avg. Open Rate</p>
        </Card>
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-4 text-center">
          <p className="text-2xl font-bold text-cyan-400">18%</p>
          <p className="text-xs text-slate-400 mt-1">Click Rate</p>
        </Card>
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-4 text-center">
          <p className="text-2xl font-bold text-cyan-400">8.2 hrs</p>
          <p className="text-xs text-slate-400 mt-1">Avg. Response Time</p>
        </Card>
        <Card className="bg-gradient-to-br from-slate-800 to-slate-900 border-slate-700/50 p-4 text-center">
          <p className="text-2xl font-bold text-cyan-400">3.4x</p>
          <p className="text-xs text-slate-400 mt-1">Deal Value vs Cold</p>
        </Card>
      </div>
    </div>
  );
}
