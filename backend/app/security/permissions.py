from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Capability(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    APPROVE = "approve"
    ADMIN = "admin"
    NETWORK_ACCESS = "network_access"
    FILE_ACCESS = "file_access"
    SECRETS_ACCESS = "secrets_access"
    AUDIT_READ = "audit_read"


class PermissionError_(Exception):
    """Raised on a denied authorization check. Named to avoid shadowing builtins.PermissionError."""


class RoleNotFoundError(Exception):
    pass


class UserNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class Role:
    name: str
    capabilities: frozenset[Capability]
    description: str = ""
    inherits: frozenset[str] = field(default_factory=frozenset)


@dataclass
class PolicyRule:
    name: str
    predicate: Callable[["AccessContext"], bool]
    effect: str = "deny"

    def __post_init__(self) -> None:
        if self.effect not in ("allow", "deny"):
            raise ValueError("effect must be 'allow' or 'deny'")


@dataclass
class AccessContext:
    user_id: str
    capability: Capability
    resource: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessDecision:
    allowed: bool
    user_id: str
    capability: Capability
    resource: Optional[str] = None
    reason: str = ""
    matched_rule: Optional[str] = None


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


DEFAULT_ROLES: dict[str, Role] = {
    "viewer": Role(
        name="viewer",
        capabilities=frozenset({Capability.READ, Capability.AUDIT_READ}),
        description="Read-only access",
    ),
    "operator": Role(
        name="operator",
        capabilities=frozenset({Capability.READ, Capability.WRITE, Capability.EXECUTE, Capability.FILE_ACCESS}),
        description="Standard operational access",
        inherits=frozenset({"viewer"}),
    ),
    "approver": Role(
        name="approver",
        capabilities=frozenset({Capability.APPROVE}),
        description="Can approve/reject pending requests",
        inherits=frozenset({"viewer"}),
    ),
    "admin": Role(
        name="admin",
        capabilities=frozenset({
            Capability.READ, Capability.WRITE, Capability.DELETE, Capability.EXECUTE,
            Capability.APPROVE, Capability.ADMIN, Capability.NETWORK_ACCESS,
            Capability.FILE_ACCESS, Capability.SECRETS_ACCESS, Capability.AUDIT_READ,
        }),
        description="Full administrative access",
    ),
}


class PermissionManager:
    """Role-based access control with capability checks and policy rules."""

    def __init__(
        self,
        *,
        roles: Optional[dict[str, Role]] = None,
        default_deny: bool = True,
    ) -> None:
        self.roles: dict[str, Role] = dict(roles) if roles is not None else dict(DEFAULT_ROLES)
        self.default_deny = default_deny
        self._user_roles: dict[str, set[str]] = {}
        self._user_direct_capabilities: dict[str, set[Capability]] = {}
        self._policy_rules: list[PolicyRule] = []
        self._created_at = time.time()
        self._check_count = 0
        self._deny_count = 0

    def register_role(self, role: Role) -> None:
        self.roles[role.name] = role

    def remove_role(self, role_name: str) -> None:
        self.roles.pop(role_name, None)
        for roles in self._user_roles.values():
            roles.discard(role_name)

    def add_policy_rule(self, rule: PolicyRule) -> None:
        self._policy_rules.append(rule)

    def assign_role(self, user_id: str, role_name: str) -> None:
        if role_name not in self.roles:
            raise RoleNotFoundError(f"role '{role_name}' is not registered")
        self._user_roles.setdefault(user_id, set()).add(role_name)

    def revoke_role(self, user_id: str, role_name: str) -> None:
        if user_id in self._user_roles:
            self._user_roles[user_id].discard(role_name)

    def grant_capability(self, user_id: str, capability: Capability) -> None:
        self._user_direct_capabilities.setdefault(user_id, set()).add(capability)

    def revoke_capability(self, user_id: str, capability: Capability) -> None:
        if user_id in self._user_direct_capabilities:
            self._user_direct_capabilities[user_id].discard(capability)

    def _resolve_role_capabilities(self, role_name: str, *, _seen: Optional[set[str]] = None) -> set[Capability]:
        seen = _seen or set()
        if role_name in seen or role_name not in self.roles:
            return set()
        seen.add(role_name)
        role = self.roles[role_name]
        caps = set(role.capabilities)
        for parent in role.inherits:
            caps |= self._resolve_role_capabilities(parent, _seen=seen)
        return caps

    def get_user_capabilities(self, user_id: str) -> set[Capability]:
        caps: set[Capability] = set(self._user_direct_capabilities.get(user_id, set()))
        for role_name in self._user_roles.get(user_id, set()):
            caps |= self._resolve_role_capabilities(role_name)
        return caps

    def get_user_roles(self, user_id: str) -> set[str]:
        return set(self._user_roles.get(user_id, set()))

    def has_capability(self, user_id: str, capability: Capability) -> bool:
        return capability in self.get_user_capabilities(user_id)

    def check(
        self,
        user_id: str,
        capability: Capability,
        *,
        resource: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> AccessDecision:
        self._check_count += 1
        context = AccessContext(user_id=user_id, capability=capability, resource=resource, attributes=attributes or {})

        for rule in self._policy_rules:
            try:
                if rule.predicate(context):
                    allowed = rule.effect == "allow"
                    if not allowed:
                        self._deny_count += 1
                    return AccessDecision(
                        allowed=allowed, user_id=user_id, capability=capability, resource=resource,
                        reason=f"policy rule '{rule.name}' matched ({rule.effect})", matched_rule=rule.name,
                    )
            except Exception as exc:
                self._deny_count += 1
                return AccessDecision(
                    allowed=False, user_id=user_id, capability=capability, resource=resource,
                    reason=f"policy rule '{rule.name}' raised an error: {exc}", matched_rule=rule.name,
                )

        has_cap = self.has_capability(user_id, capability)
        if has_cap:
            return AccessDecision(
                allowed=True, user_id=user_id, capability=capability, resource=resource,
                reason="capability granted via role or direct grant",
            )

        self._deny_count += 1
        reason = "capability not granted" if self.default_deny else "no matching rule, default allow"
        return AccessDecision(allowed=not self.default_deny, user_id=user_id, capability=capability, resource=resource, reason=reason)

    def require(
        self,
        user_id: str,
        capability: Capability,
        *,
        resource: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> AccessDecision:
        decision = self.check(user_id, capability, resource=resource, attributes=attributes)
        if not decision.allowed:
            raise PermissionError_(
                f"user '{user_id}' denied capability '{capability.value}' on resource '{resource}': {decision.reason}"
            )
        return decision

    def health_check(self) -> HealthStatus:
        try:
            probe_user = "__health_check_user__"
            self.assign_role(probe_user, "viewer") if "viewer" in self.roles else None
            self.check(probe_user, Capability.READ)
            self.revoke_role(probe_user, "viewer")
            self._user_roles.pop(probe_user, None)
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "registered_roles": list(self.roles.keys()),
                "policy_rule_count": len(self._policy_rules),
                "tracked_users": len(self._user_roles),
                "check_count": self._check_count,
                "deny_count": self._deny_count,
                "default_deny": self.default_deny,
            }
            return HealthStatus(healthy=True, component="permissions", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="permissions", details={"error": str(exc)})


__all__ = [
    "PermissionManager",
    "Role",
    "Capability",
    "PolicyRule",
    "AccessContext",
    "AccessDecision",
    "PermissionError_",
    "RoleNotFoundError",
    "UserNotFoundError",
    "HealthStatus",
    "DEFAULT_ROLES",
]
