import { useState, useEffect, type ChangeEvent } from "react";
import { Eye, EyeOff, Copy, Trash2, Plus, Check, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { listApiKeys, createApiKey, deleteApiKey } from "@/api/settings";
import type { ApiKey } from "@/api/settings";

export default function ApiKeys() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [visibleKeys, setVisibleKeys] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyScopes, setNewKeyScopes] = useState("read,write");
  const [searchTerm, setSearchTerm] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [secretKey, setSecretKey] = useState<string | null>(null);

  useEffect(() => {
    const fetchKeys = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await listApiKeys();
        setKeys(response.keys);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load API keys");
      } finally {
        setLoading(false);
      }
    };
    fetchKeys();
  }, []);

  const toggleVisibility = (id: string) => {
    setVisibleKeys((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const copyKey = (id: string, key: string) => {
    navigator.clipboard.writeText(key);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const deleteKey = async (id: string) => {
    setDeleting(id);
    try {
      await deleteApiKey(id);
      setKeys((prev) => prev.filter((k) => k.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete API key");
    } finally {
      setDeleting(null);
    }
  };

  const addKey = async () => {
    if (!newKeyName.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const scopes = newKeyScopes.split(",").map((s) => s.trim()).filter(Boolean);
      const response = await createApiKey({ name: newKeyName.trim(), scopes });
      setKeys((prev) => [response.key, ...prev]);
      setSecretKey(response.secret);
      setNewKeyName("");
      setNewKeyScopes("read,write");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create API key");
    } finally {
      setCreating(false);
    }
  };

  const maskKey = (prefix: string) => {
    if (!prefix) return "••••••••";
    return prefix + "••••••••••••••••••••";
  };

  const filteredKeys = keys.filter((k) =>
    k.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">API Keys</h2>
          <p className="text-muted-foreground mt-1">Manage your API keys.</p>
        </div>
        <div className="flex items-center justify-center py-16">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">API Keys</h2>
        <p className="text-muted-foreground mt-1">Manage your API keys.</p>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50 dark:bg-red-900/10 dark:border-red-800">
          <AlertCircle className="h-4 w-4 text-red-500" />
          <AlertDescription className="text-red-700 dark:text-red-400 text-sm ml-2">{error}</AlertDescription>
        </Alert>
      )}

      <Alert className="border-amber-200 bg-amber-50 dark:bg-amber-900/10 dark:border-amber-800">
        <AlertDescription className="text-amber-700 dark:text-amber-400 text-sm">
          Keep your API keys secure. Never share them in public repositories or client-side code.
        </AlertDescription>
      </Alert>

      {secretKey && (
        <Alert className="border-emerald-200 bg-emerald-50 dark:bg-emerald-900/10 dark:border-emerald-800">
          <AlertDescription className="text-emerald-700 dark:text-emerald-400 text-sm space-y-2">
            <p className="font-semibold">API key created successfully!</p>
            <p>Copy this key now. You won't be able to see it again.</p>
            <div className="flex items-center gap-2">
              <code className="text-xs font-mono bg-emerald-100 dark:bg-emerald-900/30 px-2 py-1 rounded break-all flex-1">{secretKey}</code>
              <Button size="sm" variant="outline" className="gap-1.5 shrink-0" onClick={() => { navigator.clipboard.writeText(secretKey); setCopiedId("secret"); setTimeout(() => setCopiedId(null), 2000); }}>
                {copiedId === "secret" ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                {copiedId === "secret" ? "Copied" : "Copy"}
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}

      <div className="flex flex-col sm:flex-row gap-3">
        <Input
          placeholder="Search keys..."
          value={searchTerm}
          onChange={(e: ChangeEvent<HTMLInputElement>) => setSearchTerm(e.target.value)}
          className="flex-1"
        />
        <Dialog open={isAddOpen} onOpenChange={(open: boolean) => { setIsAddOpen(open); if (!open) setSecretKey(null); }}>
          <DialogTrigger asChild>
            <Button className="gap-2 whitespace-nowrap">
              <Plus className="h-4 w-4" />
              Add Key
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Add API Key</DialogTitle>
              <DialogDescription>Create a new API key.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              <div className="space-y-1.5">
                <Label>Name</Label>
                <Input
                  placeholder="e.g. Production key"
                  value={newKeyName}
                  onChange={(e: ChangeEvent<HTMLInputElement>) => setNewKeyName(e.target.value)}
                />
              </div>
              <div className="space-y-1.5">
                <Label>Scopes (comma-separated)</Label>
                <Input
                  placeholder="read,write"
                  value={newKeyScopes}
                  onChange={(e: ChangeEvent<HTMLInputElement>) => setNewKeyScopes(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">Permissions for this key, e.g. read, write</p>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsAddOpen(false)}>Cancel</Button>
              <Button onClick={addKey} disabled={!newKeyName.trim() || creating}>
                {creating && <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />}
                Create Key
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="space-y-3">
        {filteredKeys.length === 0 && (
          <div className="text-center py-12 text-muted-foreground">
            No API keys found.
          </div>
        )}
        {filteredKeys.map((apiKey) => (
          <Card key={apiKey.id} className="overflow-hidden">
            <CardContent className="p-4 sm:p-6">
              <div className="flex flex-col sm:flex-row sm:items-start gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2 mb-2">
                    <span className="font-semibold text-sm">{apiKey.name}</span>
                    <Badge variant="outline" className="text-xs font-mono">{apiKey.prefix}...</Badge>
                  </div>
                  <div className="flex items-center gap-2 mb-3">
                    <code className="text-xs sm:text-sm font-mono bg-muted px-2 py-1 rounded text-muted-foreground flex-1 truncate">
                      {visibleKeys.has(apiKey.id) ? `${apiKey.prefix}••••••••••••••••••••` : maskKey(apiKey.prefix)}
                    </code>
                    <Button size="icon" variant="ghost" className="h-7 w-7 shrink-0" onClick={() => toggleVisibility(apiKey.id)}>
                      {visibleKeys.has(apiKey.id) ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                    </Button>
                    <Button size="icon" variant="ghost" className="h-7 w-7 shrink-0" onClick={() => copyKey(apiKey.id, `${apiKey.prefix}...`)}>
                      {copiedId === apiKey.id ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                    <span>Created: {new Date(apiKey.createdAt).toLocaleDateString()}</span>
                    <span>Last used: {apiKey.lastUsedAt ? new Date(apiKey.lastUsedAt).toLocaleDateString() : "Never"}</span>
                    <span>Scopes: {apiKey.scopes.join(", ")}</span>
                    {apiKey.expiresAt && <span>Expires: {new Date(apiKey.expiresAt).toLocaleDateString()}</span>}
                  </div>
                </div>
                <div className="flex items-center gap-2 sm:flex-col sm:items-end">
                  <div className="flex gap-1">
                    <Button size="icon" variant="ghost" className="h-8 w-8 text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20" onClick={() => deleteKey(apiKey.id)} disabled={deleting === apiKey.id}>
                      {deleting === apiKey.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Usage Summary</CardTitle>
          <CardDescription>Total API keys in your account.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              { label: "Total Keys", value: keys.length, color: "text-blue-500" },
              { label: "With Expiry", value: keys.filter((k) => k.expiresAt).length, color: "text-amber-500" },
              { label: "Never Used", value: keys.filter((k) => !k.lastUsedAt).length, color: "text-slate-500" },
            ].map((stat) => (
              <div key={stat.label} className="text-center p-3 rounded-lg bg-muted/50">
                <div className={`text-2xl font-bold ${stat.color}`}>{stat.value}</div>
                <div className="text-xs text-muted-foreground mt-0.5">{stat.label}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
