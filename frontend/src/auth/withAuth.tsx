"use client";

import React, { type ComponentType } from "react";
import { ProtectedLayout, type ProtectedLayoutProps } from "./ProtectedLayout";

export type WithAuthOptions = Omit<ProtectedLayoutProps, "children">;

/**
 * Higher-order component that wraps any component with ProtectedLayout.
 * Supports full guard configuration and custom slot components.
 *
 * @example
 * const AdminPage = withAuth(AdminDashboard, {
 *   guard: { requiredRoles: ["admin"] },
 *   forbidden: <AccessDeniedPage />,
 * });
 */
export const withAuth = <TProps extends object>(
  Component: ComponentType<TProps>,
  options: WithAuthOptions = {}
): ComponentType<TProps> => {
  const displayName =
    Component.displayName ?? Component.name ?? "Component";

  const WrappedComponent = (props: TProps): React.JSX.Element => (
    <ProtectedLayout {...options}>
      <Component {...props} />
    </ProtectedLayout>
  );

  WrappedComponent.displayName = `withAuth(${displayName})`;
  return WrappedComponent;
};
