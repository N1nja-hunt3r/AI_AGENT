import React, { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Save, Plus, Trash2 } from "lucide-react";
import { getSettings, updateSettings, listApiKeys, createApiKey, deleteApiKey } from "@/api/settings";
import type { ModelSettings, MemorySettings, RAGSettings, FeatureFlagSettings } from "@/api/settings";

const defaultModel: ModelSettings = { provider: "", model: "", temperature: 0.7, max_tokens: 4096, top_p: 1 };
const defaultMemory: MemorySettings = { enabled: true, max_items: 10000, ttl_seconds: null, retrieval_top_k: 5 };
const defaultRag: RAGSettings = { enabled: true, default_namespace: "default", chunk_size: 1000, chunk_overlap: 200, top_k: 5 };
const defaultFeatures: FeatureFlagSettings = { enable_streaming: true, enable_function_calling: true, enable_multi_agent: false, enable_vector_search: true, enable_experimental: false };

const SettingsPage: React.FC = () => {
  const queryClient = useQueryClient();
  const [model, setModel] = useState<ModelSettings>(defaultModel);
  const [memory, setMemory] = useState<MemorySettings>(defaultMemory);
  const [rag, setRag] = useState<RAGSettings>(defaultRag);
  const [features, setFeatures] = useState<FeatureFlagSettings>(defaultFeatures);
  const [dirty, setDirty] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");

  const { data: settings, isLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: () => getSettings(),
    staleTime: 60_000,
  });

  const { data: apiKeys } = useQuery({
    queryKey: ["settings", "api-keys"],
    queryFn: () => listApiKeys(),
  });

  useEffect(() => {
    if (settings) {
      setModel(settings.model);
      setMemory(settings.memory);
      setRag(settings.rag);
      setFeatures(settings.features);
    }
  }, [settings]);

  const saveMutation = useMutation({
    mutationFn: () => updateSettings({ model, memory, rag, features }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["settings"] }); setDirty(false); },
  });

  const createKeyMutation = useMutation({
    mutationFn: (name: string) => createApiKey({ name, scopes: ["chat:read", "chat:write"] }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ["settings", "api-keys"] }); setNewKeyName(""); },
  });

  const deleteKeyMutation = useMutation({
    mutationFn: (id: string) => deleteApiKey(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["settings", "api-keys"] }),
  });

  if (isLoading) {
    return <div className="flex h-full w-full items-center justify-center"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>;
  }

  return (
    <div className="flex h-full w-full flex-col gap-6 overflow-y-auto p-6 md:p-10">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Settings</h1>
          <p className="text-sm text-muted-foreground">Configure application preferences, integrations, and account options.</p>
        </div>
        {dirty && <Button onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}><Save className="mr-1 h-4 w-4" />{saveMutation.isPending ? "Saving..." : "Save"}</Button>}
      </div>

      <Tabs defaultValue="model" className="w-full">
        <TabsList>
          <TabsTrigger value="model">Model</TabsTrigger>
          <TabsTrigger value="memory">Memory</TabsTrigger>
          <TabsTrigger value="rag">RAG</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="api-keys">API Keys</TabsTrigger>
        </TabsList>

        <TabsContent value="model" className="space-y-4 pt-4">
          <Card>
            <CardHeader><CardTitle className="text-lg">Model Configuration</CardTitle></CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2"><Label>Provider</Label><Input value={model.provider} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setModel({ ...model, provider: e.target.value }); setDirty(true); }} /></div>
                <div className="space-y-2"><Label>Model</Label><Input value={model.model} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setModel({ ...model, model: e.target.value }); setDirty(true); }} /></div>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2"><Label>Temperature</Label><Input type="number" step={0.1} min={0} max={2} value={model.temperature} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setModel({ ...model, temperature: +e.target.value }); setDirty(true); }} /></div>
                <div className="space-y-2"><Label>Max Tokens</Label><Input type="number" step={1} min={1} value={model.max_tokens} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setModel({ ...model, max_tokens: +e.target.value }); setDirty(true); }} /></div>
                <div className="space-y-2"><Label>Top P</Label><Input type="number" step={0.05} min={0} max={1} value={model.top_p} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setModel({ ...model, top_p: +e.target.value }); setDirty(true); }} /></div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="memory" className="space-y-4 pt-4">
          <Card>
            <CardHeader><CardTitle className="text-lg">Memory Settings</CardTitle></CardHeader>
            <CardContent className="grid gap-4">
              <div className="flex items-center justify-between"><Label>Enable Memory</Label><Switch checked={memory.enabled} onCheckedChange={(v: boolean) => { setMemory({ ...memory, enabled: v }); setDirty(true); }} /></div>
              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2"><Label>Max Items</Label><Input type="number" value={memory.max_items} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setMemory({ ...memory, max_items: +e.target.value }); setDirty(true); }} /></div>
                <div className="space-y-2"><Label>TTL (seconds)</Label><Input type="number" value={memory.ttl_seconds ?? ""} placeholder="Never" onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setMemory({ ...memory, ttl_seconds: e.target.value ? +e.target.value : null }); setDirty(true); }} /></div>
                <div className="space-y-2"><Label>Retrieval Top K</Label><Input type="number" value={memory.retrieval_top_k} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setMemory({ ...memory, retrieval_top_k: +e.target.value }); setDirty(true); }} /></div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="rag" className="space-y-4 pt-4">
          <Card>
            <CardHeader><CardTitle className="text-lg">RAG Settings</CardTitle></CardHeader>
            <CardContent className="grid gap-4">
              <div className="flex items-center justify-between"><Label>Enable RAG</Label><Switch checked={rag.enabled} onCheckedChange={(v: boolean) => { setRag({ ...rag, enabled: v }); setDirty(true); }} /></div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2"><Label>Default Namespace</Label><Input value={rag.default_namespace} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setRag({ ...rag, default_namespace: e.target.value }); setDirty(true); }} /></div>
                <div className="space-y-2"><Label>Top K</Label><Input type="number" value={rag.top_k} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setRag({ ...rag, top_k: +e.target.value }); setDirty(true); }} /></div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2"><Label>Chunk Size</Label><Input type="number" value={rag.chunk_size} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setRag({ ...rag, chunk_size: +e.target.value }); setDirty(true); }} /></div>
                <div className="space-y-2"><Label>Chunk Overlap</Label><Input type="number" value={rag.chunk_overlap} onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setRag({ ...rag, chunk_overlap: +e.target.value }); setDirty(true); }} /></div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="features" className="space-y-4 pt-4">
          <Card>
            <CardHeader><CardTitle className="text-lg">Feature Flags</CardTitle></CardHeader>
            <CardContent className="grid gap-4">
              {(Object.entries(features) as [string, boolean][]).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between">
                  <Label className="capitalize">{key.replace(/_/g, " ")}</Label>
                  <Switch checked={val} onCheckedChange={(v: boolean) => { setFeatures({ ...features, [key]: v }); setDirty(true); }} />
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api-keys" className="space-y-4 pt-4">
          <Card>
            <CardHeader><CardTitle className="text-lg">API Keys</CardTitle></CardHeader>
            <CardContent className="grid gap-4">
              <div className="flex gap-2">
                <Input placeholder="New key name..." value={newKeyName} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewKeyName(e.target.value)} />
                <Button onClick={() => createKeyMutation.mutate(newKeyName)} disabled={!newKeyName.trim() || createKeyMutation.isPending}>
                  <Plus className="mr-1 h-4 w-4" /> Create
                </Button>
              </div>
              {apiKeys?.keys.map((key) => (
                <div key={key.id} className="flex items-center justify-between rounded-md border p-3">
                  <div>
                    <p className="text-sm font-medium">{key.name}</p>
                    <p className="text-xs text-muted-foreground">{key.prefix}... · Created {new Date(key.createdAt).toLocaleDateString()}</p>
                  </div>
                  <Button variant="ghost" size="icon" onClick={() => deleteKeyMutation.mutate(key.id)}><Trash2 className="h-4 w-4 text-destructive" /></Button>
                </div>
              ))}
              {(!apiKeys || apiKeys.keys.length === 0) && <p className="text-sm text-muted-foreground">No API keys created yet.</p>}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SettingsPage;
