import React, { useState } from "react";
import { Globe, ExternalLink, RefreshCw, AlertCircle, Wifi, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface Provider {
  id: string;
  name: string;
  logo: string;
  enabled: boolean;
  status: "connected" | "disconnected" | "error";
  baseUrl: string;
  models: string[];
  rateLimit: number;
  timeout: number;
  retries: number;
  priority: number;
  features: string[];
}

const PROVIDERS: Provider[] = [
  {
    id: "nvidia_nim",
    name: "NVIDIA NIM",
    logo: "🟢",
    enabled: true,
    status: "connected",
    baseUrl: "https://integrate.api.nvidia.com/v1",
    models: [
      "deepseek-ai/deepseek-v4-pro",
      "qwen/qwen3.5-122b-a10b",
      "meta/llama-3.3-70b-instruct",
      "meta/llama-3.1-70b-instruct",
      "qwen/qwen3.5-397b-a17b",
      "qwen/qwen3-next-80b-a3b-instruct",
      "meta/llama-3.2-90b-vision-instruct",
      "meta/llama-3.2-11b-vision-instruct",
      "nvidia/nv-embed-v1",
      "openai/whisper-large-v3",
      "parakeet-1.1b-rnnt-multilingual-asr",
      "magpie-tts-multilingual",
      "chatterbox-multilingual-tts",
    ],
    rateLimit: 10000,
    timeout: 60,
    retries: 3,
    priority: 1,
    features: ["text", "vision", "embeddings", "stt", "tts"],
  },
  {
    id: "openai",
    name: "OpenAI (Fallback)",
    logo: "🤖",
    enabled: false,
    status: "disconnected",
    baseUrl: "https://api.openai.com/v1",
    models: ["gpt-4o"],
    rateLimit: 10000,
    timeout: 30,
    retries: 3,
    priority: 2,
    features: ["text", "vision", "function-calling", "streaming"],
  },
];

const statusColors: Record<string, string> = {
  connected: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
  disconnected: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
  error: "bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400",
};

const StatusIcon = ({ status }: { status: string }) => {
  if (status === "connected") return <Wifi className="h-3.5 w-3.5 text-emerald-500" />;
  if (status === "error") return <AlertCircle className="h-3.5 w-3.5 text-red-500" />;
  return <WifiOff className="h-3.5 w-3.5 text-slate-400" />;
};

export default function ProviderSettings() {
  const [providers, setProviders] = useState<Provider[]>(PROVIDERS);
  const [selectedProvider, setSelectedProvider] = useState("nvidia_nim");
  const [fallbackEnabled, setFallbackEnabled] = useState(true);
  const [loadBalancing, setLoadBalancing] = useState(false);
  const [loadBalancingStrategy, setLoadBalancingStrategy] = useState("round-robin");
  const [testingProvider, setTestingProvider] = useState<string | null>(null);

  const current = providers.find((p) => p.id === selectedProvider)!;

  const updateProvider = (id: string, updates: Partial<Provider>) => {
    setProviders((prev) => prev.map((p) => (p.id === id ? { ...p, ...updates } : p)));
  };

  const testConnection = (id: string) => {
    setTestingProvider(id);
    setTimeout(() => setTestingProvider(null), 2000);
  };

  const toggleProvider = (id: string) => {
    updateProvider(id, { enabled: !providers.find((p) => p.id === id)?.enabled });
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Provider Settings</h2>
        <p className="text-muted-foreground mt-1">Configure AI provider connections and routing.</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {providers.map((provider) => (
          <button
            key={provider.id}
            onClick={() => setSelectedProvider(provider.id)}
            className={`text-left p-4 rounded-xl border-2 transition-all hover:border-primary/50 ${
              selectedProvider === provider.id ? "border-primary bg-primary/5" : "border-border bg-card"
            }`}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{provider.logo}</span>
                <div>
                  <div className="font-semibold text-sm">{provider.name}</div>
                  <div className="flex items-center gap-1 mt-0.5">
                    <StatusIcon status={provider.status} />
                    <span className={`text-xs px-1.5 py-0.5 rounded-full ${statusColors[provider.status]}`}>
                      {provider.status}
                    </span>
                  </div>
                </div>
              </div>
              <Switch
                checked={provider.enabled}
                onCheckedChange={() => toggleProvider(provider.id)}
                onClick={(e: React.MouseEvent) => e.stopPropagation()}
              />
            </div>
            <div className="flex flex-wrap gap-1">
              {provider.features.slice(0, 4).map((f) => (
                <span key={f} className="text-xs bg-muted px-1.5 py-0.5 rounded">{f}</span>
              ))}
              {provider.features.length > 4 && (
                <span className="text-xs text-muted-foreground">+{provider.features.length - 4}</span>
              )}
            </div>
          </button>
        ))}
      </div>

      {current && (
        <Tabs defaultValue="config">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="config">Configuration</TabsTrigger>
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="config" className="space-y-4 mt-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <span className="text-xl">{current.logo}</span>
                  {current.name} Configuration
                </CardTitle>
                <CardDescription>Manage connection settings for {current.name}.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-1.5">
                  <Label>Base URL</Label>
                  <Input
                    value={current.baseUrl}
                    onChange={(e) => updateProvider(current.id, { baseUrl: e.target.value })}
                  />
                  <p className="text-xs text-muted-foreground">Override the default API endpoint.</p>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label>Timeout (s)</Label>
                      <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{current.timeout}s</span>
                    </div>
                    <Slider
                      min={5} max={120} step={5}
                      value={[current.timeout]}
                      onValueChange={([v]: number[]) => updateProvider(current.id, { timeout: v })}
                    />
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label>Retries</Label>
                      <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{current.retries}</span>
                    </div>
                    <Slider
                      min={0} max={10} step={1}
                      value={[current.retries]}
                      onValueChange={([v]: number[]) => updateProvider(current.id, { retries: v })}
                    />
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label>Priority</Label>
                      <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">#{current.priority}</span>
                    </div>
                    <Slider
                      min={1} max={providers.length} step={1}
                      value={[current.priority]}
                      onValueChange={([v]: number[]) => updateProvider(current.id, { priority: v })}
                    />
                  </div>
                </div>
                <Separator />
                <div className="flex flex-col sm:flex-row gap-3">
                  <Button
                    variant="outline"
                    size="sm"
                    className="gap-2"
                    onClick={() => testConnection(current.id)}
                    disabled={testingProvider === current.id}
                  >
                    {testingProvider === current.id ? (
                      <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Globe className="h-3.5 w-3.5" />
                    )}
                    {testingProvider === current.id ? "Testing..." : "Test Connection"}
                  </Button>
                  <Button variant="outline" size="sm" className="gap-2">
                    <ExternalLink className="h-3.5 w-3.5" />
                    View Documentation
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="models" className="space-y-4 mt-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Available Models</CardTitle>
                <CardDescription>Models available through {current.name}.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {current.models.map((model, i) => (
                    <div key={model} className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors">
                      <div className="flex items-center gap-3">
                        <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center text-xs font-bold text-primary">
                          {i + 1}
                        </div>
                        <div>
                          <div className="font-mono text-sm">{model}</div>
                          <div className="text-xs text-muted-foreground">{current.name}</div>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-xs">Active</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-4 mt-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Rate Limiting</CardTitle>
                <CardDescription>Configure request rate limits.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Requests per minute</Label>
                    <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{current.rateLimit.toLocaleString()}</span>
                  </div>
                  <Slider
                    min={100} max={100000} step={100}
                    value={[current.rateLimit]}
                    onValueChange={([v]: number[]) => updateProvider(current.id, { rateLimit: v })}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Routing & Fallback</CardTitle>
          <CardDescription>Configure how requests are routed between providers.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <Label>Automatic Fallback</Label>
              <p className="text-xs text-muted-foreground mt-0.5">Automatically switch to OpenAI if NVIDIA NIM is unavailable.</p>
            </div>
            <Switch checked={fallbackEnabled} onCheckedChange={setFallbackEnabled} />
          </div>
          <div className="flex items-center justify-between">
            <div>
              <Label>Load Balancing</Label>
              <p className="text-xs text-muted-foreground mt-0.5">Distribute requests across providers.</p>
            </div>
            <Switch checked={loadBalancing} onCheckedChange={setLoadBalancing} />
          </div>
          {loadBalancing && (
            <div className="space-y-1.5 pl-0">
              <Label>Strategy</Label>
              <Select value={loadBalancingStrategy} onValueChange={setLoadBalancingStrategy}>
                <SelectTrigger className="w-full sm:w-56">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="round-robin">Round Robin</SelectItem>
                  <SelectItem value="least-latency">Least Latency</SelectItem>
                  <SelectItem value="cost-optimized">Cost Optimized</SelectItem>
                  <SelectItem value="weighted">Weighted</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex justify-end gap-3">
        <Button variant="outline">Reset All</Button>
        <Button>Save Provider Settings</Button>
      </div>
    </div>
  );
}
