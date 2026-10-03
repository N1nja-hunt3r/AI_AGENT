import { useState } from "react";
import { Sun, Moon, Monitor, Check, Palette, Type, Layout } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

const COLOR_SCHEMES = [
  { id: "zinc", name: "Zinc", primary: "#71717a", preview: "bg-zinc-500" },
  { id: "slate", name: "Slate", primary: "#64748b", preview: "bg-slate-500" },
  { id: "blue", name: "Blue", primary: "#3b82f6", preview: "bg-blue-500" },
  { id: "violet", name: "Violet", primary: "#8b5cf6", preview: "bg-violet-500" },
  { id: "rose", name: "Rose", primary: "#f43f5e", preview: "bg-rose-500" },
  { id: "orange", name: "Orange", primary: "#f97316", preview: "bg-orange-500" },
  { id: "green", name: "Green", primary: "#22c55e", preview: "bg-green-500" },
  { id: "teal", name: "Teal", primary: "#14b8a6", preview: "bg-teal-500" },
];

const FONTS = [
  { id: "inter", name: "Inter", preview: "Modern & Clean" },
  { id: "geist", name: "Geist", preview: "Vercel's Choice" },
  { id: "mono", name: "JetBrains Mono", preview: "Developer Friendly" },
  { id: "serif", name: "Playfair Display", preview: "Elegant & Classic" },
];

const LAYOUTS = [
  { id: "default", name: "Default", description: "Standard sidebar layout" },
  { id: "compact", name: "Compact", description: "Minimized sidebar" },
  { id: "wide", name: "Wide", description: "Full-width content" },
  { id: "centered", name: "Centered", description: "Centered content column" },
];

const DENSITY = [
  { id: "compact", name: "Compact" },
  { id: "comfortable", name: "Comfortable" },
  { id: "spacious", name: "Spacious" },
];

export default function ThemeSettings() {
  const [theme, setTheme] = useState<"light" | "dark" | "system">("system");
  const [colorScheme, setColorScheme] = useState("blue");
  const [font, setFont] = useState("inter");
  const [fontSize, setFontSize] = useState(14);
  const [borderRadius, setBorderRadius] = useState(8);
  const [layout, setLayout] = useState("default");
  const [density, setDensity] = useState("comfortable");
  const [animations, setAnimations] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [customCSS, setCustomCSS] = useState(false);
  const [sidebarPosition, setSidebarPosition] = useState("left");

  const themeOptions = [
    { id: "light", label: "Light", icon: Sun },
    { id: "dark", label: "Dark", icon: Moon },
    { id: "system", label: "System", icon: Monitor },
  ] as const;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Theme Settings</h2>
        <p className="text-muted-foreground mt-1">Customize the appearance of your interface.</p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Sun className="h-4 w-4" />
            <CardTitle className="text-base">Appearance Mode</CardTitle>
          </div>
          <CardDescription>Choose between light, dark, or system theme.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-3">
            {themeOptions.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setTheme(id)}
                className={`relative flex flex-col items-center gap-3 p-4 rounded-xl border-2 transition-all hover:border-primary/50 ${
                  theme === id ? "border-primary bg-primary/5" : "border-border"
                }`}
              >
                {theme === id && (
                  <span className="absolute top-2 right-2 h-4 w-4 rounded-full bg-primary flex items-center justify-center">
                    <Check className="h-2.5 w-2.5 text-primary-foreground" />
                  </span>
                )}
                <div className={`p-3 rounded-lg ${theme === id ? "bg-primary/10" : "bg-muted"}`}>
                  <Icon className={`h-5 w-5 ${theme === id ? "text-primary" : "text-muted-foreground"}`} />
                </div>
                <span className={`text-sm font-medium ${theme === id ? "text-primary" : ""}`}>{label}</span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Palette className="h-4 w-4" />
            <CardTitle className="text-base">Color Scheme</CardTitle>
          </div>
          <CardDescription>Select your preferred accent color.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
            {COLOR_SCHEMES.map((scheme) => (
              <button
                key={scheme.id}
                onClick={() => setColorScheme(scheme.id)}
                className="flex flex-col items-center gap-1.5 group"
                title={scheme.name}
              >
                <div className={`relative w-10 h-10 rounded-full ${scheme.preview} transition-transform group-hover:scale-110`}>
                  {colorScheme === scheme.id && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Check className="h-4 w-4 text-white drop-shadow" />
                    </div>
                  )}
                </div>
                <span className={`text-xs ${colorScheme === scheme.id ? "font-semibold" : "text-muted-foreground"}`}>
                  {scheme.name}
                </span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Type className="h-4 w-4" />
            <CardTitle className="text-base">Typography</CardTitle>
          </div>
          <CardDescription>Configure font family and size.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {FONTS.map((f) => (
              <button
                key={f.id}
                onClick={() => setFont(f.id)}
                className={`text-left p-3 rounded-lg border-2 transition-all hover:border-primary/50 ${
                  font === f.id ? "border-primary bg-primary/5" : "border-border"
                }`}
              >
                <div className="flex items-center justify-between mb-0.5">
                  <span className="font-medium text-sm">{f.name}</span>
                  {font === f.id && <Check className="h-4 w-4 text-primary" />}
                </div>
                <span className="text-xs text-muted-foreground">{f.preview}</span>
              </button>
            ))}
          </div>
          <Separator />
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Font Size</Label>
              <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded">{fontSize}px</span>
            </div>
            <Slider min={10} max={24} step={1} value={[fontSize]} onValueChange={([v]: number[]) => setFontSize(v)} />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Small (10px)</span>
              <span>Large (24px)</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Layout className="h-4 w-4" />
            <CardTitle className="text-base">Layout & Spacing</CardTitle>
          </div>
          <CardDescription>Adjust layout options and component density.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label>Border Radius</Label>
            <div className="flex items-center gap-4">
              <Slider min={0} max={20} step={2} value={[borderRadius]} onValueChange={([v]: number[]) => setBorderRadius(v)} className="flex-1" />
              <span className="text-sm font-mono bg-muted px-2 py-0.5 rounded w-14 text-center shrink-0">{borderRadius}px</span>
            </div>
            <div className="flex gap-2 pt-1">
              {[0, 4, 8, 12, 16, 20].map((r) => (
                <button
                  key={r}
                  onClick={() => setBorderRadius(r)}
                  className={`w-8 h-8 border-2 bg-muted transition-all ${borderRadius === r ? "border-primary" : "border-border"}`}
                  style={{ borderRadius: `${r}px` }}
                />
              ))}
            </div>
          </div>
          <Separator />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label>Layout</Label>
              <RadioGroup value={layout} onValueChange={setLayout} className="space-y-2">
                {LAYOUTS.map((l) => (
                  <div key={l.id} className="flex items-start gap-2">
                    <RadioGroupItem value={l.id} id={`layout-${l.id}`} className="mt-0.5" />
                    <div>
                      <Label htmlFor={`layout-${l.id}`} className="font-medium cursor-pointer">{l.name}</Label>
                      <p className="text-xs text-muted-foreground">{l.description}</p>
                    </div>
                  </div>
                ))}
              </RadioGroup>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Density</Label>
                <RadioGroup value={density} onValueChange={setDensity} className="space-y-2">
                  {DENSITY.map((d) => (
                    <div key={d.id} className="flex items-center gap-2">
                      <RadioGroupItem value={d.id} id={`density-${d.id}`} />
                      <Label htmlFor={`density-${d.id}`} className="cursor-pointer">{d.name}</Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>
              <div className="space-y-2">
                <Label>Sidebar Position</Label>
                <Select value={sidebarPosition} onValueChange={setSidebarPosition}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="left">Left</SelectItem>
                    <SelectItem value="right">Right</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
          <Separator />
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <Label>Animations</Label>
                <p className="text-xs text-muted-foreground mt-0.5">Enable transition animations.</p>
              </div>
              <Switch checked={animations} onCheckedChange={setAnimations} />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <Label>Reduced Motion</Label>
                <p className="text-xs text-muted-foreground mt-0.5">Minimize motion for accessibility.</p>
              </div>
              <Switch checked={reducedMotion} onCheckedChange={setReducedMotion} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">Custom CSS</CardTitle>
              <CardDescription>Advanced: inject custom styles.</CardDescription>
            </div>
            <Switch checked={customCSS} onCheckedChange={setCustomCSS} />
          </div>
        </CardHeader>
        {customCSS && (
          <CardContent>
            <textarea
              rows={6}
              className="w-full font-mono text-xs bg-muted rounded-lg p-3 border border-border focus:outline-none focus:ring-2 focus:ring-primary resize-none"
              placeholder=":root { --custom-color: #ff0000; }"
            />
          </CardContent>
        )}
      </Card>

      <div className="flex justify-end gap-3">
        <Button variant="outline">Reset to Defaults</Button>
        <Button>Apply Theme</Button>
      </div>
    </div>
  );
}