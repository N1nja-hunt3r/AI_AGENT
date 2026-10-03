import { useState } from "react";
import { Info, Zap, Brain, Clock, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Textarea } from "@/components/ui/textarea";

interface Model {
  id: string;
  name: string;
  role: string;
  provider: string;
  contextLength: number;
  speed: "fast" | "medium" | "slow";
  capabilities: string[];
  recommended?: boolean;
}

const MODELS: Model[] = [
  { id: "deepseek-ai/deepseek-v4-pro", name: "DeepSeek V4 Pro", role: "Chief", provider: "NVIDIA NIM", contextLength: 131072, speed: "medium", capabilities: ["text", "reasoning", "function-calling", "code-generation"], recommended: true },
  { id: "qwen/qwen3.5-122b-a10b", name: "Qwen 3.5 122B", role: "Chief (Fallback)", provider: "NVIDIA NIM", contextLength: 131072, speed: "medium", capabilities: ["text", "reasoning", "function-calling"] },
  { id: "meta/llama-3.3-70b-instruct", name: "Llama 3.3 70B", role: "Coder", provider: "NVIDIA NIM", contextLength: 128000, speed: "fast", capabilities: ["text", "code-generation", "function-calling"] },
  { id: "meta/llama-3.1-70b-instruct", name: "Llama 3.1 70B", role: "Coder (Fallback)", provider: "NVIDIA NIM", contextLength: 128000, speed: "fast", capabilities: ["text", "code-generation", "function-calling"] },
  { id: "qwen/qwen3.5-397b-a17b", name: "Qwen 3.5 397B", role: "Researcher", provider: "NVIDIA NIM", contextLength: 131072, speed: "slow", capabilities: ["text", "reasoning", "function-calling"] },
  { id: "qwen/qwen3-next-80b-a3b-instruct", name: "Qwen 3 Next 80B", role: "Researcher (Fallback)", provider: "NVIDIA NIM", contextLength: 131072, speed: "medium", capabilities: ["text", "reasoning", "function-calling"] },
  { id: "meta/llama-3.2-90b-vision-instruct", name: "Llama 3.2 90B Vision", role: "Vision", provider: "NVIDIA NIM", contextLength: 128000, speed: "medium", capabilities: ["text", "vision"] },
  { id: "meta/llama-3.2-11b-vision-instruct", name: "Llama 3.2 11B Vision", role: "Vision (Fallback)", provider: "NVIDIA NIM", contextLength: 128000, speed: "fast", capabilities: ["text", "vision"] },
  { id: "nvidia/nv-embed-v1", name: "NV-Embed-v1", role: "Embeddings", provider: "NVIDIA NIM", contextLength: 2048, speed: "fast", capabilities: ["embeddings"] },
  { id: "openai/whisper-large-v3", name: "Whisper Large v3", role: "STT", provider: "NVIDIA NIM", contextLength: 0, speed: "fast", capabilities: ["audio", "speech-to-text"] },
  { id: "parakeet-1.1b-rnnt-multilingual-asr", name: "Parakeet 1.1B", role: "STT (Fallback)", provider: "NVIDIA NIM", contextLength: 0, speed: "fast", capabilities: ["audio", "speech-to-text"] },
  { id: "magpie-tts-multilingual", name: "Magpie TTS", role: "TTS", provider: "NVIDIA NIM", contextLength: 0, speed: "fast", capabilities: ["audio", "text-to-speech"] },
  { id: "chatterbox-multilingual-tts", name: "Chatterbox TTS", role: "TTS (Fallback)", provider: "NVIDIA NIM", contextLength: 0, speed: "fast", capabilities: ["audio", "text-to-speech"] },
  { id: "gpt-4o", name: "GPT-4o", role: "Emergency Fallback", provider: "OpenAI (Fallback)", contextLength: 128000, speed: "fast", capabilities: ["text", "vision", "function-calling"] },
];

const speedIcon = (speed: string) => {
  if (speed === "fast") return <Zap className="h-3 w-3 text-yellow-500" />;
  if (speed === "medium") return <Clock className="h-3 w-3 text-blue-500" />;
  return <Brain className="h-3 w-3 text-purple-500" />;
};

const formatContext = (n: number) => {
  if (n === 0) return "N/A";
  if (n >= 1000000) return `${n / 1000000}M`;
  if (n >= 1000) return `${n / 1000}K`;
  return n.toString();
};

export default function ModelSettings() {
  const [selectedModel, setSelectedModel] = useState("deepseek-ai/deepseek-v4-pro");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [topP, setTopP] = useState(1.0);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0);
  const [presencePenalty, setPresencePenalty] = useState(0);
  const [systemPrompt, setSystemPrompt] = useState("You are Aspire, an intelligent AI assistant.");
  const [streamResponse, setStreamResponse] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [filterProvider, setFilterProvider] = useState("all");

  const current = MODELS.find((m) => m.id === selectedModel);

  const filteredModels = MODELS.filter(
    (m) => filterProvider === "all" || m.provider === filterProvider
  );

  const providers = Array.from(new Set(MODELS.map((m) => m.provider)));

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Model Settings</h2>
          <p className="text-muted-foreground mt-1">Configure AI model selection and parameters.</p>
        </div>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <CardTitle className="text-base">Select Model</CardTitle>
                <CardDescription>Choose the AI model for your conversations.</CardDescription>
              </div>
              <Select value={filterProvider} onValueChange={setFilterProvider}>
                <SelectTrigger className="w-full sm:w-40">
                  <SelectValue placeholder="All Providers" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Providers</SelectItem>
                  {providers.map((p) => (
                    <SelectItem key={p} value={p}>{p}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {filteredModels.map((model) => (
                <button
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className={`relative text-left p-3 rounded-lg border-2 transition-all hover:border-primary/50 ${
                    selectedModel === model.id
                      ? "border-primary bg-primary/5"
                      : "border-border bg-card"
                  }`}
                >
                  {model.recommended && (
                    <span className="absolute top-2 right-2 text-xs bg-primary text-primary-foreground px-1.5 py-0.5 rounded-full">
                      Recommended
                    </span>
                  )}
                  <div className="flex items-center gap-1.5 mb-1.5">
                    {speedIcon(model.speed)}
                    <span className="font-medium text-sm">{model.name}</span>
                  </div>
                  <div className="text-xs text-muted-foreground mb-2">{model.role} · {model.provider}</div>
                  <div className="flex flex-wrap gap-1 mb-2">
                    {model.capabilities.slice(0, 3).map((cap) => (
                      <span key={cap} className="text-xs bg-muted px-1.5 py-0.5 rounded">
                        {cap}
                      </span>
                    ))}
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{formatContext(model.contextLength)} ctx</span>
                    <span>{model.speed}</span>
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {current && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Selected: {current.name}</CardTitle>
              <CardDescription>Model details and capabilities.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: "Context", value: formatContext(current.contextLength) + " tokens" },
                  { label: "Speed", value: current.speed },
                  { label: "Role", value: current.role },
                  { label: "Provider", value: current.provider },
                ].map((item) => (
                  <div key={item.label} className="bg-muted/50 rounded-lg p-3">
                    <div className="text-xs text-muted-foreground mb-1">{item.label}</div>
                    <div className="font-semibold text-sm capitalize">{item.value}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Generation Parameters</CardTitle>
            <CardDescription>Adjust model behavior and output characteristics.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Label>Temperature</Label>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-3.5 w-3.5 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>Controls randomness. Lower = more focused, Higher = more creative.</TooltipContent>
                  </Tooltip>
                </div>
                <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{temperature.toFixed(1)}</span>
              </div>
              <Slider
                min={0} max={2} step={0.1}
                value={[temperature]}
                onValueChange={([v]: number[]) => setTemperature(v)}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Precise (0)</span>
                <span>Balanced (1)</span>
                <span>Creative (2)</span>
              </div>
            </div>

            <Separator />

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Label>Max Tokens</Label>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-3.5 w-3.5 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>Maximum number of tokens to generate.</TooltipContent>
                  </Tooltip>
                </div>
                <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{maxTokens}</span>
              </div>
              <Slider
                min={256} max={16384} step={256}
                value={[maxTokens]}
                onValueChange={([v]: number[]) => setMaxTokens(v)}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>256</span>
                <span>16,384</span>
              </div>
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div>
                <Label>Stream Response</Label>
                <p className="text-xs text-muted-foreground mt-0.5">Show tokens as they are generated.</p>
              </div>
              <Switch checked={streamResponse} onCheckedChange={setStreamResponse} />
            </div>

            <Button
              variant="ghost"
              size="sm"
              className="w-full gap-2 text-muted-foreground"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              {showAdvanced ? "Hide" : "Show"} Advanced Parameters
            </Button>

            {showAdvanced && (
              <div className="space-y-6 pt-2">
                <Separator />
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Top P</Label>
                    <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{topP.toFixed(2)}</span>
                  </div>
                  <Slider min={0} max={1} step={0.01} value={[topP]} onValueChange={([v]: number[]) => setTopP(v)} />
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Frequency Penalty</Label>
                    <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{frequencyPenalty.toFixed(1)}</span>
                  </div>
                  <Slider min={-2} max={2} step={0.1} value={[frequencyPenalty]} onValueChange={([v]: number[]) => setFrequencyPenalty(v)} />
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Presence Penalty</Label>
                    <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{presencePenalty.toFixed(1)}</span>
                  </div>
                  <Slider min={-2} max={2} step={0.1} value={[presencePenalty]} onValueChange={([v]: number[]) => setPresencePenalty(v)} />
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">System Prompt</CardTitle>
            <CardDescription>Set the default system message for all conversations.</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={4}
              className="resize-none font-mono text-sm"
              placeholder="You are Aspire, an intelligent AI assistant..."
            />
            <p className="text-xs text-muted-foreground mt-2">{systemPrompt.length} characters</p>
          </CardContent>
        </Card>

        <div className="flex justify-end gap-3">
          <Button variant="outline">Reset to Defaults</Button>
          <Button>Save Settings</Button>
        </div>
      </div>
    </TooltipProvider>
  );
}
