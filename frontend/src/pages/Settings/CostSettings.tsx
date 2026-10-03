import { useState } from "react";
import { DollarSign, TrendingUp, AlertTriangle, Bell, BarChart3, ArrowUpRight, ArrowDownRight, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const USAGE_DATA = [
  { date: "Jan 14", cost: 2.34, tokens: 156800 },
  { date: "Jan 15", cost: 3.12, tokens: 208000 },
  { date: "Jan 16", cost: 1.89, tokens: 126000 },
  { date: "Jan 17", cost: 4.56, tokens: 304000 },
  { date: "Jan 18", cost: 2.78, tokens: 185600 },
  { date: "Jan 19", cost: 5.23, tokens: 348800 },
  { date: "Jan 20", cost: 3.45, tokens: 230000 },
];

const PROVIDER_COSTS = [
  { provider: "NVIDIA NIM", cost: 18.45, percentage: 78, color: "bg-green-500" },
  { provider: "OpenAI (Fallback)", cost: 5.23, percentage: 22, color: "bg-slate-400" },
];

const MODEL_COSTS = [
  { model: "deepseek-ai/deepseek-v4-pro", role: "Chief", provider: "NVIDIA NIM", inputTokens: 456000, outputTokens: 123000, cost: 8.34, trend: "up" },
  { model: "qwen/qwen3.5-122b-a10b", role: "Chief (Fallback)", provider: "NVIDIA NIM", inputTokens: 0, outputTokens: 0, cost: 0, trend: "stable" },
  { model: "meta/llama-3.3-70b-instruct", role: "Coder", provider: "NVIDIA NIM", inputTokens: 234000, outputTokens: 89000, cost: 5.12, trend: "down" },
  { model: "meta/llama-3.1-70b-instruct", role: "Coder (Fallback)", provider: "NVIDIA NIM", inputTokens: 0, outputTokens: 0, cost: 0, trend: "stable" },
  { model: "qwen/qwen3.5-397b-a17b", role: "Research", provider: "NVIDIA NIM", inputTokens: 189000, outputTokens: 56000, cost: 3.12, trend: "up" },
  { model: "qwen/qwen3-next-80b-a3b-instruct", role: "Research (Fallback)", provider: "NVIDIA NIM", inputTokens: 0, outputTokens: 0, cost: 0, trend: "stable" },
  { model: "meta/llama-3.2-90b-vision-instruct", role: "Vision", provider: "NVIDIA NIM", inputTokens: 123000, outputTokens: 45000, cost: 0.89, trend: "stable" },
  { model: "meta/llama-3.2-11b-vision-instruct", role: "Vision (Fallback)", provider: "NVIDIA NIM", inputTokens: 0, outputTokens: 0, cost: 0, trend: "stable" },
];

const maxBarValue = Math.max(...USAGE_DATA.map((d) => d.cost));

export default function CostSettings() {
  const [monthlyBudget, setMonthlyBudget] = useState(50);
  const [dailyLimit, setDailyLimit] = useState(10);
  const [alertThreshold, setAlertThreshold] = useState(80);
  const [hardLimit, setHardLimit] = useState(true);
  const [emailAlerts, setEmailAlerts] = useState(true);
  const [slackAlerts, setSlackAlerts] = useState(false);
  const [autoScaleDown, setAutoScaleDown] = useState(false);
  const [currency, setCurrency] = useState("USD");
  const [billingCycle, setBillingCycle] = useState("monthly");
  const [budgetInput, setBudgetInput] = useState("50");
  const [dailyInput, setDailyInput] = useState("10");

  const totalSpent = PROVIDER_COSTS.reduce((a, b) => a + b.cost, 0);
  const budgetPercent = Math.round((totalSpent / monthlyBudget) * 100);
  const isOverAlert = budgetPercent >= alertThreshold;
  const isOverBudget = totalSpent >= monthlyBudget;

  const formatCost = (n: number) => `$${n.toFixed(2)}`;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Cost Management</h2>
        <p className="text-muted-foreground mt-1">Monitor spending and configure budget limits.</p>
      </div>

      {isOverAlert && (
        <Alert className={`${isOverBudget ? "border-red-300 bg-red-50 dark:bg-red-900/10 dark:border-red-800" : "border-amber-200 bg-amber-50 dark:bg-amber-900/10 dark:border-amber-800"}`}>
          <AlertTriangle className={`h-4 w-4 ${isOverBudget ? "text-red-500" : "text-amber-500"}`} />
          <AlertDescription className={`ml-2 text-sm ${isOverBudget ? "text-red-700 dark:text-red-400" : "text-amber-700 dark:text-amber-400"}`}>
            {isOverBudget
              ? `Budget exceeded! You've spent $${totalSpent.toFixed(2)} of your $${monthlyBudget} limit.`
              : `Warning: You've used ${budgetPercent}% of your monthly budget.`}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: "This Month", value: formatCost(totalSpent), sub: `of $${monthlyBudget} budget`, icon: DollarSign, color: "text-blue-500 bg-blue-50 dark:bg-blue-900/20" },
          { label: "Today", value: formatCost(USAGE_DATA[USAGE_DATA.length - 1].cost), sub: `of $${dailyLimit} limit`, icon: Calendar, color: "text-emerald-500 bg-emerald-50 dark:bg-emerald-900/20" },
          { label: "Total Tokens", value: "1.36M", sub: "this month", icon: BarChart3, color: "text-purple-500 bg-purple-50 dark:bg-purple-900/20" },
          { label: "Avg per Day", value: formatCost(totalSpent / 7), sub: "last 7 days", icon: TrendingUp, color: "text-orange-500 bg-orange-50 dark:bg-orange-900/20" },
        ].map((stat) => (
          <Card key={stat.label}>
            <CardContent className="p-4">
              <div className={`inline-flex p-2 rounded-lg mb-3 ${stat.color}`}>
                <stat.icon className="h-4 w-4" />
              </div>
              <div className="text-2xl font-bold mb-0.5">{stat.value}</div>
              <div className="text-xs text-muted-foreground">{stat.label}</div>
              <div className="text-xs text-muted-foreground">{stat.sub}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Tabs defaultValue="overview">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="models">By Model</TabsTrigger>
          <TabsTrigger value="limits">Limits</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4 mt-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Daily Spending (Last 7 Days)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-end gap-2 h-32">
                {USAGE_DATA.map((d) => (
                  <div key={d.date} className="flex-1 flex flex-col items-center gap-1">
                    <span className="text-xs text-muted-foreground hidden sm:block">${d.cost.toFixed(2)}</span>
                    <div className="w-full flex items-end justify-center" style={{ height: "80px" }}>
                      <div
                        className="w-full max-w-10 rounded-t-md bg-primary/80 hover:bg-primary transition-colors"
                        style={{ height: `${(d.cost / maxBarValue) * 80}px` }}
                        title={`${d.date}: $${d.cost.toFixed(2)}`}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground">{d.date.split(" ")[1]}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Cost by Provider</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {PROVIDER_COSTS.map((p) => (
                <div key={p.provider} className="space-y-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{p.provider}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">{p.percentage}%</span>
                      <span className="font-semibold">{formatCost(p.cost)}</span>
                    </div>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className={`h-full rounded-full ${p.color} transition-all`}
                      style={{ width: `${p.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
              <Separator className="my-2" />
              <div className="flex items-center justify-between font-semibold">
                <span>Total</span>
                <span>{formatCost(totalSpent)}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Monthly Budget Progress</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span>{formatCost(totalSpent)} spent</span>
                <span>{formatCost(monthlyBudget - totalSpent)} remaining</span>
              </div>
              <Progress
                value={Math.min(budgetPercent, 100)}
                className={`h-3 ${isOverBudget ? "[&>div]:bg-red-500" : isOverAlert ? "[&>div]:bg-amber-500" : ""}`}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>$0</span>
                <span className={`font-medium ${isOverAlert ? "text-amber-600" : ""}`}>{budgetPercent}% used</span>
                <span>{formatCost(monthlyBudget)}</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="models" className="mt-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Cost by Model</CardTitle>
              <CardDescription>Token usage and costs per model this month.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {MODEL_COSTS.map((model) => (
                  <div key={model.model} className="flex flex-col sm:flex-row sm:items-center gap-3 p-3 rounded-lg border bg-card hover:bg-muted/30 transition-colors">
                    <div className="flex-1 min-w-0">
                      <div className="flex flex-wrap items-center gap-2 mb-1">
                        <span className="font-mono text-sm font-medium">{model.model}</span>
                        <Badge variant="outline" className="text-xs">{model.role}</Badge>
                      </div>
                      <div className="flex flex-wrap gap-x-4 text-xs text-muted-foreground">
                        <span>Input: {(model.inputTokens / 1000).toFixed(0)}K tokens</span>
                        <span>Output: {(model.outputTokens / 1000).toFixed(0)}K tokens</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1">
                        {model.trend === "up" && <ArrowUpRight className="h-3.5 w-3.5 text-red-500" />}
                        {model.trend === "down" && <ArrowDownRight className="h-3.5 w-3.5 text-emerald-500" />}
                        {model.trend === "stable" && <span className="h-3.5 w-3.5 flex items-center justify-center text-slate-400">—</span>}
                      </div>
                      <span className="font-bold text-sm">{formatCost(model.cost)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="limits" className="space-y-4 mt-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Budget Configuration</CardTitle>
              <CardDescription>Set spending limits and alert thresholds.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div className="space-y-1.5">
                  <Label>Monthly Budget</Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">$</span>
                    <Input
                      type="number"
                      value={budgetInput}
                      onChange={(e) => { setBudgetInput(e.target.value); setMonthlyBudget(Number(e.target.value) || 0); }}
                      className="pl-7"
                      min={0}
                    />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <Label>Daily Limit</Label>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">$</span>
                    <Input
                      type="number"
                      value={dailyInput}
                      onChange={(e) => { setDailyInput(e.target.value); setDailyLimit(Number(e.target.value) || 0); }}
                      className="pl-7"
                      min={0}
                    />
                  </div>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Alert at % of budget</Label>
                  <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{alertThreshold}%</span>
                </div>
                <Slider min={10} max={100} step={5} value={[alertThreshold]} onValueChange={([v]: number[]) => setAlertThreshold(v)} />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>10%</span>
                  <span>100%</span>
                </div>
              </div>
              <Separator />
              <div className="space-y-4">
                {[
                  { label: "Hard Limit", desc: "Block requests when budget is reached.", value: hardLimit, onChange: setHardLimit },
                  { label: "Auto Scale Down", desc: "Switch to cheaper models as budget runs low.", value: autoScaleDown, onChange: setAutoScaleDown },
                ].map((item) => (
                  <div key={item.label} className="flex items-center justify-between">
                    <div>
                      <Label>{item.label}</Label>
                      <p className="text-xs text-muted-foreground mt-0.5">{item.desc}</p>
                    </div>
                    <Switch checked={item.value} onCheckedChange={item.onChange} />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Bell className="h-4 w-4" />
                Alert Notifications
              </CardTitle>
              <CardDescription>Configure how you receive spending alerts.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                { label: "Email Alerts", desc: "Receive alerts via email.", value: emailAlerts, onChange: setEmailAlerts },
                { label: "Slack Notifications", desc: "Send alerts to a Slack channel.", value: slackAlerts, onChange: setSlackAlerts },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between">
                  <div>
                    <Label>{item.label}</Label>
                    <p className="text-xs text-muted-foreground mt-0.5">{item.desc}</p>
                  </div>
                  <Switch checked={item.value} onCheckedChange={item.onChange} />
                </div>
              ))}
              <Separator />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Currency</Label>
                  <Select value={currency} onValueChange={setCurrency}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="USD">USD ($)</SelectItem>
                      <SelectItem value="EUR">EUR (€)</SelectItem>
                      <SelectItem value="GBP">GBP (£)</SelectItem>
                      <SelectItem value="JPY">JPY (¥)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Billing Cycle</Label>
                  <Select value={billingCycle} onValueChange={setBillingCycle}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="flex justify-end gap-3">
        <Button variant="outline">Reset Limits</Button>
        <Button>Save Cost Settings</Button>
      </div>
    </div>
  );
}
