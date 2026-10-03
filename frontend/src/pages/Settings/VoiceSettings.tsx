import { useState } from "react";
import { Volume2, Play, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";

const VOICES = [
  { id: "whisper-large-v3", name: "Whisper Large v3", gender: "neutral", style: "STT", provider: "NVIDIA NIM" },
  { id: "parakeet-1.1b", name: "Parakeet 1.1B", gender: "neutral", style: "STT", provider: "NVIDIA NIM" },
  { id: "magpie-tts-multilingual", name: "Magpie TTS", gender: "neutral", style: "TTS", provider: "NVIDIA NIM" },
  { id: "chatterbox-multilingual", name: "Chatterbox TTS", gender: "neutral", style: "TTS", provider: "NVIDIA NIM" },
];

const LANGUAGES = [
  { code: "en-US", name: "English (US)" },
  { code: "en-GB", name: "English (UK)" },
  { code: "es-ES", name: "Spanish" },
  { code: "fr-FR", name: "French" },
  { code: "de-DE", name: "German" },
  { code: "ja-JP", name: "Japanese" },
  { code: "zh-CN", name: "Chinese (Simplified)" },
  { code: "pt-BR", name: "Portuguese (Brazil)" },
];

export default function VoiceSettings() {
  const [selectedVoice, setSelectedVoice] = useState("alloy");
  const [language, setLanguage] = useState("en-US");
  const [speed, setSpeed] = useState(1.0);
  const [volume, setVolume] = useState(0.8);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [noiseSuppress, setNoiseSuppress] = useState(true);
  const [echoCancellation, setEchoCancellation] = useState(true);
  const [autoGain, setAutoGain] = useState(true);
  const [wakeWord, setWakeWord] = useState(false);
  const [wakeWordValue, setWakeWordValue] = useState("Hey Aspire");
  const [inputDevice, setInputDevice] = useState("default");
  const [outputDevice, setOutputDevice] = useState("default");
  const [isPlaying, setIsPlaying] = useState(false);
  const [provider, setProvider] = useState("all");

  const filteredVoices = VOICES.filter((v) => provider === "all" || v.provider === provider);

  const togglePlay = () => {
    setIsPlaying((p) => !p);
    if (!isPlaying) {
      setTimeout(() => setIsPlaying(false), 3000);
    }
  };

  const selectedVoiceData = VOICES.find((v) => v.id === selectedVoice);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Voice Settings</h2>
        <p className="text-muted-foreground mt-1">Configure text-to-speech and speech recognition settings.</p>
      </div>

      <div className="flex items-center justify-between p-4 rounded-lg border bg-card">
        <div>
          <Label className="text-base font-semibold">Enable Voice Features</Label>
          <p className="text-sm text-muted-foreground mt-0.5">Turn on voice input and output capabilities.</p>
        </div>
        <Switch checked={voiceEnabled} onCheckedChange={setVoiceEnabled} />
      </div>

      <div className={voiceEnabled ? "" : "opacity-50 pointer-events-none"}>
        <div className="space-y-6">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-base">Voice Selection</CardTitle>
                  <CardDescription>Choose the voice for text-to-speech output.</CardDescription>
                </div>
                <Select value={provider} onValueChange={setProvider}>
                  <SelectTrigger className="w-full sm:w-36">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="NVIDIA NIM">NVIDIA NIM</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 mb-4">
                {filteredVoices.map((voice) => (
                  <button
                    key={voice.id}
                    onClick={() => setSelectedVoice(voice.id)}
                    className={`text-left p-3 rounded-lg border-2 transition-all hover:border-primary/50 ${
                      selectedVoice === voice.id ? "border-primary bg-primary/5" : "border-border"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium text-sm">{voice.name}</span>
                      <Volume2 className="h-3.5 w-3.5 text-muted-foreground" />
                    </div>
                    <div className="flex gap-1 flex-wrap">
                      <span className="text-xs bg-muted px-1.5 py-0.5 rounded capitalize">{voice.gender}</span>
                      <span className="text-xs bg-muted px-1.5 py-0.5 rounded capitalize">{voice.style}</span>
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">{voice.provider}</div>
                  </button>
                ))}
              </div>

              {selectedVoiceData && (
                <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 p-3 rounded-lg bg-muted/50">
                  <div className="flex-1">
                    <span className="text-sm font-medium">Preview: {selectedVoiceData.name}</span>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      "The quick brown fox jumps over the lazy dog."
                    </p>
                  </div>
                  <Button size="sm" variant="outline" className="gap-2 shrink-0" onClick={togglePlay}>
                    {isPlaying ? <Square className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                    {isPlaying ? "Stop" : "Preview"}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Speech Output</CardTitle>
              <CardDescription>Adjust voice output parameters.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Speed</Label>
                    <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{speed.toFixed(1)}x</span>
                  </div>
                  <Slider min={0.25} max={4} step={0.25} value={[speed]} onValueChange={([v]: number[]) => setSpeed(v)} />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>0.25x</span>
                    <span>1.0x</span>
                    <span>4.0x</span>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Volume</Label>
                    <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{Math.round(volume * 100)}%</span>
                  </div>
                  <Slider min={0} max={1} step={0.05} value={[volume]} onValueChange={([v]: number[]) => setVolume(v)} />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Mute</span>
                    <span>Max</span>
                  </div>
                </div>
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <Label>Auto-speak responses</Label>
                  <p className="text-xs text-muted-foreground mt-0.5">Automatically read AI responses aloud.</p>
                </div>
                <Switch checked={autoSpeak} onCheckedChange={setAutoSpeak} />
              </div>
              <div className="space-y-1.5">
                <Label>Language</Label>
                <Select value={language} onValueChange={setLanguage}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {LANGUAGES.map((lang) => (
                      <SelectItem key={lang.code} value={lang.code}>{lang.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Microphone & Input</CardTitle>
              <CardDescription>Configure voice input and noise settings.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <Label>Input Device</Label>
                  <Select value={inputDevice} onValueChange={setInputDevice}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">Default Microphone</SelectItem>
                      <SelectItem value="headset">Headset Microphone</SelectItem>
                      <SelectItem value="external">External Microphone</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label>Output Device</Label>
                  <Select value={outputDevice} onValueChange={setOutputDevice}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="default">Default Speakers</SelectItem>
                      <SelectItem value="headphones">Headphones</SelectItem>
                      <SelectItem value="external">External Speakers</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <Separator />
              <div className="space-y-4">
                {[
                  { label: "Noise Suppression", desc: "Reduce background noise during recording.", value: noiseSuppress, onChange: setNoiseSuppress },
                  { label: "Echo Cancellation", desc: "Prevent audio feedback and echo.", value: echoCancellation, onChange: setEchoCancellation },
                  { label: "Auto Gain Control", desc: "Automatically adjust microphone sensitivity.", value: autoGain, onChange: setAutoGain },
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
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <Label>Wake Word Detection</Label>
                  <p className="text-xs text-muted-foreground mt-0.5">Activate with a voice command.</p>
                </div>
                <Switch checked={wakeWord} onCheckedChange={setWakeWord} />
              </div>
              {wakeWord && (
                <div className="space-y-1.5 pl-0">
                  <Label>Wake Word</Label>
                  <Select value={wakeWordValue} onValueChange={setWakeWordValue}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Hey Aspire">Hey Aspire</SelectItem>
                      <SelectItem value="OK Aspire">OK Aspire</SelectItem>
                      <SelectItem value="Hello Aspire">Hello Aspire</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="flex justify-end gap-3">
            <Button variant="outline">Reset to Defaults</Button>
            <Button>Save Settings</Button>
          </div>
        </div>
      </div>
    </div>
  );
}