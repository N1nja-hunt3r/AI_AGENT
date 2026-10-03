import { useState } from "react";
import { Shield, Key, Smartphone, AlertTriangle, Eye, EyeOff, Lock, Unlock, LogOut, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

const SESSIONS = [
  { id: "1", device: "MacBook Pro", browser: "Chrome 121", location: "New York, US", ip: "192.168.1.1", lastActive: "Now", current: true },
  { id: "2", device: "iPhone 15", browser: "Safari Mobile", location: "New York, US", ip: "192.168.1.2", lastActive: "2 hours ago", current: false },
  { id: "3", device: "Windows PC", browser: "Firefox 122", location: "London, UK", ip: "10.0.0.1", lastActive: "3 days ago", current: false },
];

const passwordStrength = (pass: string): { score: number; label: string; color: string } => {
  if (!pass) return { score: 0, label: "None", color: "bg-muted" };
  const score = [/.{8,}/, /[A-Z]/, /[a-z]/, /\d/, /[^A-Za-z0-9]/].filter((r) => r.test(pass)).length;
  const levels = [
    { score: 0, label: "None", color: "bg-muted" },
    { score: 1, label: "Very Weak", color: "bg-red-500" },
    { score: 2, label: "Weak", color: "bg-orange-500" },
    { score: 3, label: "Fair", color: "bg-yellow-500" },
    { score: 4, label: "Strong", color: "bg-blue-500" },
    { score: 5, label: "Very Strong", color: "bg-emerald-500" },
  ];
  return levels[score] || levels[0];
};

export default function SecuritySettings() {
  const [twoFactor, setTwoFactor] = useState(false);
  const [loginNotifs, setLoginNotifs] = useState(true);
  const [apiAccess, setApiAccess] = useState(true);
  const [sessionTimeout, setSessionTimeout] = useState("60");
  const [ipWhitelist, setIpWhitelist] = useState(false);
  const [auditLog, setAuditLog] = useState(true);
  const [dataEncryption, setDataEncryption] = useState(true);
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [showCurrentPw, setShowCurrentPw] = useState(false);
  const [showNewPw, setShowNewPw] = useState(false);
  const [sessions, setSessions] = useState(SESSIONS);
  const [is2FAOpen, setIs2FAOpen] = useState(false);
  const [verifyCode, setVerifyCode] = useState("");

  const strength = passwordStrength(newPw);
  const passwordMatch = newPw && confirmPw && newPw === confirmPw;

  const revokeSession = (id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
  };

  const securityScore = [twoFactor, loginNotifs, dataEncryption, auditLog, ipWhitelist].filter(Boolean).length;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Security Settings</h2>
        <p className="text-muted-foreground mt-1">Manage your account security and privacy.</p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Security Score
          </CardTitle>
          <CardDescription>Your current security configuration.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 mb-3">
            <div className="text-4xl font-bold">{securityScore * 20}%</div>
            <div className="flex-1">
              <Progress value={securityScore * 20} className="h-3" />
              <p className="text-xs text-muted-foreground mt-1">
                {securityScore === 5 ? "Excellent security!" : `Enable ${5 - securityScore} more feature${5 - securityScore === 1 ? "" : "s"} to improve your security.`}
              </p>
            </div>
          </div>
          {!twoFactor && (
            <Alert className="border-orange-200 bg-orange-50 dark:bg-orange-900/10 dark:border-orange-800">
              <AlertTriangle className="h-4 w-4 text-orange-500" />
              <AlertDescription className="text-orange-700 dark:text-orange-400 text-sm ml-2">
                Two-factor authentication is not enabled. We strongly recommend enabling it.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Key className="h-4 w-4" />
            Change Password
          </CardTitle>
          <CardDescription>Update your account password.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label>Current Password</Label>
            <div className="relative">
              <Input
                type={showCurrentPw ? "text" : "password"}
                value={currentPw}
                onChange={(e) => setCurrentPw(e.target.value)}
                placeholder="Enter current password"
                className="pr-10"
              />
              <Button size="icon" variant="ghost" className="absolute right-1 top-1 h-7 w-7" onClick={() => setShowCurrentPw(!showCurrentPw)}>
                {showCurrentPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
              </Button>
            </div>
          </div>
          <div className="space-y-1.5">
            <Label>New Password</Label>
            <div className="relative">
              <Input
                type={showNewPw ? "text" : "password"}
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                placeholder="Enter new password"
                className="pr-10"
              />
              <Button size="icon" variant="ghost" className="absolute right-1 top-1 h-7 w-7" onClick={() => setShowNewPw(!showNewPw)}>
                {showNewPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
              </Button>
            </div>
            {newPw && (
              <div className="space-y-1.5 mt-2">
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className={`h-1.5 flex-1 rounded-full transition-colors ${i <= strength.score ? strength.color : "bg-muted"}`} />
                  ))}
                </div>
                <p className={`text-xs ${strength.score >= 4 ? "text-emerald-600" : strength.score >= 3 ? "text-yellow-600" : "text-red-600"}`}>
                  {strength.label}
                </p>
              </div>
            )}
          </div>
          <div className="space-y-1.5">
            <Label>Confirm Password</Label>
            <Input
              type="password"
              value={confirmPw}
              onChange={(e) => setConfirmPw(e.target.value)}
              placeholder="Confirm new password"
              className={confirmPw ? (passwordMatch ? "border-emerald-500" : "border-red-500") : ""}
            />
            {confirmPw && !passwordMatch && <p className="text-xs text-red-500">Passwords don't match.</p>}
            {passwordMatch && <p className="text-xs text-emerald-600">Passwords match!</p>}
          </div>
          <div className="text-xs text-muted-foreground space-y-1">
            {["At least 8 characters", "One uppercase letter", "One lowercase letter", "One number", "One special character"].map((req) => (
              <div key={req} className="flex items-center gap-1.5">
                <div className={`h-1.5 w-1.5 rounded-full ${newPw.length > 0 ? "bg-emerald-500" : "bg-muted"}`} />
                <span>{req}</span>
              </div>
            ))}
          </div>
          <Button disabled={!currentPw || !passwordMatch} className="w-full sm:w-auto">
            Update Password
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Smartphone className="h-4 w-4" />
            Two-Factor Authentication
          </CardTitle>
          <CardDescription>Add an extra layer of security to your account.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-sm">2FA Status</p>
              <p className="text-xs text-muted-foreground">
                {twoFactor ? "Two-factor authentication is enabled." : "Two-factor authentication is disabled."}
              </p>
            </div>
            <Badge className={twoFactor ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400" : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"} variant="outline">
              {twoFactor ? "Enabled" : "Disabled"}
            </Badge>
          </div>
          <Dialog open={is2FAOpen} onOpenChange={setIs2FAOpen}>
            <DialogTrigger asChild>
              <Button variant={twoFactor ? "destructive" : "default"} size="sm" className="gap-2">
                {twoFactor ? <Unlock className="h-4 w-4" /> : <Lock className="h-4 w-4" />}
                {twoFactor ? "Disable 2FA" : "Enable 2FA"}
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>{twoFactor ? "Disable" : "Enable"} Two-Factor Authentication</DialogTitle>
                <DialogDescription>
                  {twoFactor ? "Enter your verification code to disable 2FA." : "Scan the QR code with your authenticator app."}
                </DialogDescription>
              </DialogHeader>
              {!twoFactor && (
                <div className="flex justify-center py-4">
                  <div className="w-40 h-40 bg-muted rounded-lg flex items-center justify-center text-muted-foreground text-sm">
                    QR Code Here
                  </div>
                </div>
              )}
              <div className="space-y-2">
                <Label>Verification Code</Label>
                <Input placeholder="000000" maxLength={6} value={verifyCode} onChange={(e) => setVerifyCode(e.target.value)} />
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIs2FAOpen(false)}>Cancel</Button>
                <Button onClick={() => { setTwoFactor(!twoFactor); setIs2FAOpen(false); setVerifyCode(""); }}>
                  Confirm
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Clock className="h-4 w-4" />
            Active Sessions
          </CardTitle>
          <CardDescription>Manage your active login sessions.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {sessions.map((session) => (
            <div key={session.id} className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 rounded-lg border bg-card">
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="font-medium text-sm">{session.device}</span>
                  {session.current && (
                    <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 text-xs" variant="outline">
                      Current
                    </Badge>
                  )}
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
                  <span>{session.browser}</span>
                  <span>{session.location}</span>
                  <span>IP: {session.ip}</span>
                  <span>Active: {session.lastActive}</span>
                </div>
              </div>
              {!session.current && (
                <Button size="sm" variant="outline" className="gap-1.5 text-red-500 border-red-200 hover:bg-red-50 dark:hover:bg-red-900/20 shrink-0" onClick={() => revokeSession(session.id)}>
                  <LogOut className="h-3.5 w-3.5" />
                  Revoke
                </Button>
              )}
            </div>
          ))}
          <Button variant="outline" size="sm" className="w-full text-red-500" onClick={() => setSessions(sessions.filter((s) => s.current))}>
            Revoke All Other Sessions
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Security Preferences</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {[
            { label: "Login Notifications", desc: "Get notified of new login attempts.", value: loginNotifs, onChange: setLoginNotifs },
            { label: "API Access Logging", desc: "Log all API key usage.", value: apiAccess, onChange: setApiAccess },
            { label: "Audit Log", desc: "Keep a record of all account actions.", value: auditLog, onChange: setAuditLog },
            { label: "Data Encryption at Rest", desc: "Encrypt stored data and conversations.", value: dataEncryption, onChange: setDataEncryption },
            { label: "IP Whitelist", desc: "Only allow access from specified IPs.", value: ipWhitelist, onChange: setIpWhitelist },
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
          <div className="space-y-1.5">
            <Label>Session Timeout</Label>
            <Select value={sessionTimeout} onValueChange={setSessionTimeout}>
              <SelectTrigger className="w-full sm:w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="15">15 minutes</SelectItem>
                <SelectItem value="30">30 minutes</SelectItem>
                <SelectItem value="60">1 hour</SelectItem>
                <SelectItem value="240">4 hours</SelectItem>
                <SelectItem value="480">8 hours</SelectItem>
                <SelectItem value="0">Never</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end gap-3">
        <Button variant="outline">Cancel</Button>
        <Button>Save Security Settings</Button>
      </div>
    </div>
  );
}